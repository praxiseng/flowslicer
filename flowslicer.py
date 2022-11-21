
from collections import defaultdict
from binaryninja.mediumlevelil import SSAVariable
from binaryninja import flowgraph, BranchType
try:
    from .dfil import *
except ImportError:
    from dfil import *
from collections.abc import Mapping


@dataclass(init=True, frozen=True)
class DataSlice:
    nodes: list[DataNode]
    edges: list[DataFlowEdge]
    expressions: list[TokenExpression]

    def display_verbose(self, dfx):
        # nids = set([n.node_id for n in self.nodes])
        slice_nids_txt = ','.join([str(n.node_id) for n in self.nodes])

        print(f'\nSlice {slice_nids_txt}')
        for n in self.nodes:
            dfx.print_verbose_node_line(n)

        for edge in self.edges:
            print(f'  {edge.in_node.node_id:2} -> {edge.out_node.node_id:2}  {edge.edge_type.name}')

        for expression in self.expressions:
            print(f'Node {expression.base_node.node_id:2} Expression {"".join(expression.get_text())}')


class ExpressionSlice:
    ''' To fold expressions, we need a different notion of nodes and edges that expands
        lists into descriptions. '''
    def __init__(self, nodes: list[DataNode]):
        self.nodes = nodes
        self.expressions: list[TokenExpression] = []
        self.xmap: dict[int, TokenExpression] = {}
        self.input_nodes = set()
        for n in nodes:
            expr = n.get_expression()
            self.expressions.append(expr)
            self.xmap[n.node_id] = expr

        for expr in self.expressions:
            for index, token in enumerate(expr.tokens):
                if isinstance(token, DataNode):
                    if token.node_id in self.xmap:
                        in_expr = self.xmap[token.node_id]
                        edge = ExpressionEdge(in_expr, expr)
                        in_expr.uses.append(edge)
                        expr.tokens[index] = edge
                    else:
                        self.input_nodes.add(token.node_id)

    def fold_node(self, node_id):
        assert(node_id in self.xmap)

    def fold_constants(self):
        new_expressions = []
        for expr in self.expressions:
            if expr.base_node.operation == DataFlowILOperation.DFIL_DECLARE_CONST:
                continue
            new_expressions.append(expr)
            for index, token in enumerate(expr.tokens):
                match token:
                    case DataNode() as dn:
                        if dn.operation == DataFlowILOperation.DFIL_DECLARE_CONST:
                            expr.tokens[index] = dn.il_expr
                            self.input_nodes.remove(dn.node_id)
                    case ExpressionEdge() as ee:
                        in_expr: TokenExpression = ee.in_expr
                        if in_expr.base_node.operation == DataFlowILOperation.DFIL_DECLARE_CONST:
                            expr.tokens[index] = in_expr.base_node.il_expr
        self.expressions = new_expressions

    def display_verbose(self, dfx):

        print(f'Input: {self.input_nodes}')
        #for n in self.nodes:
        #    dfx.print_verbose_node_line(n)

        for expression in self.expressions:
            print(f'Node {expression.base_node.node_id:2} Expression {"".join(expression.get_text())}')




def fold_const(data_slice: DataSlice) -> DataSlice:
    const_node_nids = [n.node_id for n in data_slice.nodes if n.operation == DataFlowILOperation.DFIL_DECLARE_CONST]
    new_nodes = [n for n in data_slice.nodes if n.operation != DataFlowILOperation.DFIL_DECLARE_CONST]
    new_edges = [e for e in data_slice.edges if e.in_node.node_id not in const_node_nids]
    new_expressions = []
    for expr in data_slice.expressions:
        if expr.tokens[0] == 'DFIL_DECLARE_CONST':
            continue
        folded_nids = expr.folded_nids
        new_tokens = []
        for tok in expr.tokens:
            if isinstance(tok, DataNode) and tok.operation == DataFlowILOperation.DFIL_DECLARE_CONST:
                new_tokens.extend(tok.get_expression().tokens)
                folded_nids = folded_nids | {tok.node_id}
            else:
                new_tokens.append(tok)
        new_expressions.append(TokenExpression(expr.base_node, new_tokens, folded_nids))

    return DataSlice(new_nodes, new_edges, new_expressions)



DEFAULT_SLICING_BLACKLIST=\
    frozenset({
        DataFlowILOperation.DFIL_CALL,
        DataFlowILOperation.DFIL_DEREF,
        DataFlowILOperation.DFIL_STORE,
        DataFlowILOperation.DFIL_ARRAY_INDEX,
        # Don't want to unify all references to the same constant (e.g. 0)
        DataFlowILOperation.DFIL_DECLARE_CONST,
        DataFlowILOperation.DFIL_LOGIC_AND,
        DataFlowILOperation.DFIL_LOGIC_OR,
    })


class DataFlowILFunction:
    def __init__(self,
                 basic_blocks: list[DataBasicBlock],
                 data_flow_edges: list[DataFlowEdge],
                 control_edges: list[ControlEdge]):

        self.basic_blocks = basic_blocks
        self.control_edges = control_edges

        self.all_nodes: dict[int, DataNode] = {}
        self.node_to_bb: dict[int, DataBasicBlock] = {}
        self.nodes_by_il = {}
        self.vars_to_nodes = {}
        self.bb_id_to_bb = {}
        for bb in basic_blocks:
            self.bb_id_to_bb[bb.block_id] = bb
            for dn in bb.data_nodes:
                self.all_nodes[dn.node_id] = dn
                self.node_to_bb[dn.node_id] = bb
                match dn.il_expr:
                    case commonil.BaseILInstruction() as il_instr:
                        self.nodes_by_il[il_instr.instr_index] = dn
                    case binaryninja.SSAVariable():
                        self.vars_to_nodes[repr(dn.il_expr)] = dn

        self.node_to_ce: dict[int, ControlEdge] = {}
        for ce in control_edges:
            if not ce.data_node:
                continue
            self.node_to_ce[ce.data_node.node_id] = ce

        self.in_edges: Mapping[int, list[DataFlowEdge]] = defaultdict(list)
        self.out_edges: Mapping[int, list[DataFlowEdge]] = defaultdict(list)
        for edge in data_flow_edges:
            a, b = edge.in_node, edge.out_node
            self.out_edges[a.node_id].append(edge)
            self.in_edges[b.node_id].append(edge)

    def get_node_edges_by_id(self, nid: int) -> \
            tuple[
                list[DataFlowEdge],
                list[DataFlowEdge],
                DataBasicBlock,
                ControlEdge]:
        return (self.in_edges.get(nid, []),
                self.out_edges.get(nid, []),
                self.node_to_bb.get(nid, None),
                self.node_to_ce.get(nid, None))

    def get_node_edges(self, n: DataNode) -> \
            tuple[
                list[DataFlowEdge],
                list[DataFlowEdge],
                DataBasicBlock,
                ControlEdge]:
        return self.get_node_edges_by_id(n.node_id)\


    def get_interior_edges(self, node_ids):
        interior = []
        for nid in node_ids:
            for edge in self.out_edges.get(nid, []):
                if edge.out_node.node_id in node_ids:
                    interior.append(edge)
        # for edge in self.all_edges:
        #     if edge.in_node.node_id in node_ids and edge.out_node.node_id in node_ids:
        #         interior.append(edge)
        return interior

    def _expand_slice(self,
                      start_nid: int,
                      op_blacklist: list[DataFlowILOperation] = DEFAULT_SLICING_BLACKLIST
                      ):

        print(f'\nExpanding {start_nid}')
        nids_in_slice = {start_nid}
        remaining_nodes = {start_nid}
        while remaining_nodes:
            nid = min(remaining_nodes)
            remaining_nodes.remove(nid)

            edges_in, edges_out, bb, ce = self.get_node_edges_by_id(nid)

            nodes_in = [e.in_node for e in edges_in]
            nodes_out = [e.out_node for e in edges_out]
            connected_nodes = nodes_in+nodes_out

            # connected_nids = [n.node_id for n in connected_nodes]
            nodes_in_txt = ','.join([str(n.node_id) for n in nodes_in])
            nodes_out_txt = ','.join([str(n.node_id) for n in nodes_out])
            nodes_inout_txt = f'in:{nodes_in_txt:8} out:{nodes_out_txt:12}'

            print(f'Expand {nid:4}  {str(remaining_nodes):30} {nodes_inout_txt} {nids_in_slice}')
            for n in connected_nodes:
                if n.node_id in nids_in_slice:
                    continue
                # if n.node_id in remaining_nodes:
                #     continue
                nids_in_slice.add(n.node_id)
                # We capture blacklisted nodes in the slice, but don't traverse their edges

                print(f'Node {n.node_id}  {"not " if n.operation not in op_blacklist else ""}in blacklist')
                if n.operation not in op_blacklist:
                    remaining_nodes.add(n.node_id)
        return nids_in_slice

    def expand_slice(self,
                     start_node: DataNode,
                     op_blacklist: list[DataFlowILOperation] = DEFAULT_SLICING_BLACKLIST):

        nids_in_slice = self._expand_slice(start_node.node_id, op_blacklist)

        return [self.all_nodes[nid] for nid in sorted(nids_in_slice)]

    def partition_basic_slices(self, op_blacklist: list[DataFlowILOperation] = DEFAULT_SLICING_BLACKLIST):
        remaining_nids = set(self.all_nodes.keys())
        slices = []

        while remaining_nids:
            nid = remaining_nids.pop()
            slice_nids = self._expand_slice(nid, op_blacklist)
            remaining_nids -= slice_nids

            data_slice = [self.all_nodes[nid] for nid in sorted(slice_nids)]
            slices.append(data_slice)

        return slices

    def get_bb_edges(self, bb: DataBasicBlock) -> tuple[list[ControlEdge], list[ControlEdge]]:
        in_edges = []
        out_edges = []
        for ce in self.control_edges:
            if ce.out_block.block_id == bb.block_id:
                in_edges.append(ce)
            if ce.in_block.block_id == bb.block_id:
                out_edges.append(ce)
        return in_edges, out_edges

    def graph(self):
        g = flowgraph.FlowGraph()
        graph_nodes = {}
        shown_nodes = set()
        for bb in self.basic_blocks:
            graph_node = flowgraph.FlowGraphNode(g)
            lines = []
            for dn in bb.base_data_nodes:
                il = dn.base_instr
                il_txt = ''
                if il:
                    il_txt = f'{il.instr_index:3}@{il.address:<5x}'

                lines.append(f'{dn.node_id:3} {il_txt:12} {dn.get_dfil_txt(shown_nodes)}')
                shown_nodes.add(dn.node_id)
            graph_node.lines = lines
            graph_nodes[bb.block_id] = graph_node
            graph_node.basic_block = bb.il_block
            g.append(graph_node)
        for edge in self.control_edges:
            branch_map = {
                ControlEdgeType.TrueBranch: binaryninja.BranchType.TrueBranch,
                ControlEdgeType.FalseBranch: binaryninja.BranchType.FalseBranch,
                ControlEdgeType.Unconditional: binaryninja.BranchType.UnconditionalBranch,
            }
            edge_type = branch_map.get(edge.edge_type, binaryninja.BranchType.UnconditionalBranch)
            in_node = graph_nodes[edge.in_block.block_id]
            out_node = graph_nodes[edge.out_block.block_id]
            in_node.add_outgoing_edge(edge_type, out_node)
        binaryninja.show_graph_report('Data Flow IL', g)

    def _collect_outgoing(self, nodes, current_node: DataNode):
        node_id = current_node.node_id
        if node_id in nodes:
            return
        nodes[node_id] = current_node
        for out_edge in self.out_edges[node_id]:
            self._collect_outgoing(nodes, out_edge.out_node)

    def graph_flows_from(self, dn: DataNode):
        g = flowgraph.FlowGraph()
        data_nodes = {}
        self._collect_outgoing(data_nodes, dn)
        graph_nodes = {}
        edges = []
        for data_node in data_nodes.values():
            graph_node = flowgraph.FlowGraphNode(g)
            graph_node.lines = [f'{data_node.node_id} {self.get_dfil_txt(data_node)}',
                                data_node.get_dfil_txt()]
            graph_nodes[data_node.node_id] = graph_node
            g.append(graph_node)

        # [edge for edge in self.out_edges[data_node.node_id] for data_node in data_nodes.values()]
        for data_node in data_nodes.values():
            node_id = data_node.node_id
            for edge in self.out_edges[node_id]:
                edges.append(edge)
        for edge in edges:
            a = graph_nodes[edge.in_node.node_id]
            b = graph_nodes[edge.out_node.node_id]
            load_style = binaryninja.flowgraph.EdgeStyle(binaryninja.flowgraph.EdgePenStyle.SolidLine, 5,
                                                         binaryninja.enums.ThemeColor.TrueBranchColor)
            flow_type_map = {
                EdgeType.Move: (BranchType.UnconditionalBranch, None),
                EdgeType.Load: (BranchType.UserDefinedBranch, load_style),
                EdgeType.Store: (BranchType.FalseBranch, None),
                EdgeType.Branch: (BranchType.IndirectBranch, None)
            }
            edge_type, edge_style = flow_type_map.get(edge.edge_type, (BranchType.UnconditionalBranch, None))
            a.add_outgoing_edge(edge_type, b, edge_style)

        binaryninja.show_graph_report(f'Data flow of {dn.node_id}', g)

    def get_dfil_txt(self, dn):
        in_edges = self.in_edges[dn.node_id]
        out_edges = self.out_edges[dn.node_id]

        op_txt = dn.get_operation_txt()

        arg_txt = ', '.join([e.short_in_txt() for e in in_edges])

        if 'DECLARE' in op_txt:
            arg_txt = dn.format_tokens()
        outputs_txt = ' '.join([e.short_out_txt() for e in out_edges])
        if outputs_txt:
            outputs_txt = ' -> ' + outputs_txt
        return f'{op_txt}({arg_txt}){outputs_txt}'

    def print_verbose_node_line(self, n):
        edges_in, edges_out, bb, ce = self.get_node_edges(n)

        in_txt = ','.join(f'{e.in_node.node_id}' for e in edges_in)
        out_txt = ','.join(f'{e.out_node.node_id}{e.edge_type.short()}' for e in edges_out)
        ce_txt = f'->B{ce.out_block.block_id}' if ce else ''

        hlil_txt = f'{n.get_il_index():2}@{n.get_il_address():<5x}'

        expr_txt = n.get_expression().get_text()

        dfil_txt = f'DFIL {n.node_id:2} B{bb.block_id}'
        hlil_txt = f'HLIL {hlil_txt:8}'
        edge_txt = f'Edges {in_txt:12} {out_txt:10} {ce_txt:5}'

        print(f'{dfil_txt} {hlil_txt} {edge_txt} {expr_txt:28} {n.format_tokens()}')


class ILParser:
    def __init__(self):
        self.next_node_id = 1
        self.data_blocks = []
        self.current_data_bb: DataBasicBlock
        self.current_base_instr = None

        self.il_bb_to_dbb = {}

        self.nodes_by_id = {}

        self.varnodes = {}

        # node_id -> list[Edge]
        self.out_edges = defaultdict(list)
        self.in_edges = defaultdict(list)

        self.all_data_edges = []

    def _node(self, expr, operands: [], out_var=None):
        node_id = self.next_node_id
        self.next_node_id += 1
        dfil_op = get_dfil_op(expr)
        bb_id = self.current_data_bb.block_id
        dn = DataNode(self.current_base_instr, expr, operands, node_id, bb_id, dfil_op)
        self.current_data_bb.data_nodes.append(dn)
        self.nodes_by_id[node_id] = dn
        if out_var:
            self.varnodes[out_var] = dn

        for operand_index, operand in enumerate(operands):
            edge_type = get_edge_type_from_dfil_op(dfil_op, operand_index)
            e = DataFlowEdge(operand, dn, edge_type, operand_index)
            self.out_edges[operand.node_id].append(e)
            self.in_edges[node_id].append(e)
            self.all_data_edges.append(e)

        return dn

    def _unimplemented(self, expr):
        node_id = self.next_node_id
        self.next_node_id += 1
        return DataNode(self.current_base_instr, expr, [], node_id, DataFlowILOperation.DFIL_UNKNOWN)

    def _var(self, var):
        if var in self.varnodes:
            return self.varnodes[var]

        # print(f' var {var}')
        dn = self._node(var, [])
        self.varnodes[var] = dn

        return dn

    def _recurse(self, expr):
        match expr:
            case commonil.Constant() as constant:
                node = self._var(constant.constant)
            case int() | float():
                node = self._var(expr)
            case binaryninja.variable.Variable() as var:
                node = self._var(var)
            case highlevelil.HighLevelILVarSsa(var=ssa):
                node = self._var(ssa)
            case highlevelil.HighLevelILVarPhi() as var_phi:
                operands = [self._recurse(operand) for operand in var_phi.src]
                node = self._node(expr, operands, var_phi.dest)
            case highlevelil.HighLevelILVarInitSsa() as var_ssa:
                operands = [self._recurse(var_ssa.src)]
                node = self._node(expr, operands, var_ssa.dest)
            case highlevelil.HighLevelILVarInit() as var_init:
                operands = [self._recurse(var_init.src)]
                node = self._node(expr, operands, var_init.dest)
            case highlevelil.HighLevelILCallSsa():
                dest, params, dest_mem, src_mem = expr.operands
                operands = [self._recurse(operand) for operand in [dest] + params]
                node = self._node(expr, operands)
            case highlevelil.HighLevelILArrayIndexSsa() as array_index_ssa:
                operands = [self._recurse(operand) for operand in [array_index_ssa.src, array_index_ssa.index]]
                node = self._node(expr, operands)
            case highlevelil.HighLevelILStructField() as struct_field:
                # .member_index has been None
                # TODO: consider unifying struct/array/deref under a generic GEP (a la LLVM) to unify dereferences.
                operands = [self._recurse(operand) for operand in [struct_field.src, struct_field.offset]]
                node = self._node(expr, operands)
            case highlevelil.HighLevelILAssignMemSsa() as assign_mem_ssa:
                # Note that expr.dest tends to contain the memory reference, not the HighLevelILAssignMemSsa.  expr.dest
                # can be HighLevelILDerefSsa or HighLevelILArrayIndexSsa, for example
                operands = [self._recurse(operand) for operand in [assign_mem_ssa.dest, assign_mem_ssa.src]]
                node = self._node(expr, operands)
            case highlevelil.HighLevelILTailcall() as tailcall:
                operands = [self._recurse(operand) for operand in tailcall.params]
                node = self._node(expr, operands, tailcall.dest)
            case highlevelil.HighLevelILDerefSsa() as deref_ssa:
                node = self._node(expr, [self._recurse(deref_ssa.src)])
            case highlevelil.HighLevelILMemPhi() | highlevelil.HighLevelILVarDeclare() | highlevelil.HighLevelILNoret():
                # These will be at the highest level, so we don't need to worry about returning None as an operand
                node = None
            case SSAVariable() as ssa:
                node = self._var(ssa)
            case commonil.BaseILInstruction():
                operands = [self._recurse(operand) for operand in expr.operands]
                node = self._node(expr, operands)
            case _:
                print(f'Type not handled: {type(expr)} for {expr}')
                node = self._unimplemented(expr)
                # node = self._node(expr, [])

        return node

    def parse(self, il: list[binaryninja.highlevelil.HighLevelILBasicBlock]) -> DataFlowILFunction:
        for index, il_bb in enumerate(il):
            dbb = DataBasicBlock(il_bb, [], index, [], [], [], None)
            self.data_blocks.append(dbb)
            self.il_bb_to_dbb[il_bb.index] = dbb

        for il_bb in il:
            dbb = self.il_bb_to_dbb[il_bb.index]
            self.current_data_bb = dbb
            for il_instr in il_bb:
                self.current_base_instr = il_instr
                dn = self._recurse(il_instr)
                if dn:
                    dbb.base_data_nodes.append(dn)

        all_control_edges = []
        for dbb in self.data_blocks:
            out_edge_node = None
            if dbb.data_nodes and dbb.data_nodes[-1].is_control_related():
                out_edge_node = dbb.data_nodes[-1]

            for out_edge in dbb.il_block.outgoing_edges:
                out_dbb = self.il_bb_to_dbb[out_edge.target.index]
                et = ControlEdgeType.from_basic_block_edge(out_edge)
                ce = ControlEdge(dbb, out_dbb, et, out_edge_node)
                all_control_edges.append(ce)
                dbb.edges_out.append(ce)
                out_dbb.edges_in.append(ce)

        return DataFlowILFunction(self.data_blocks, self.all_data_edges, all_control_edges)


def display_node_tree(dfil_fx, node, depth=0):
    print(f'{"  "*depth}{node.node_id:2} {dfil_fx.get_dfil_txt(node):60} {node.format_tokens()}')
    out_edges = dfil_fx.out_edges[node.node_id]
    for out_edge in out_edges:
        display_node_tree(dfil_fx, out_edge.out_node, depth=depth+1)


def print_dfil(dfil_fx: DataFlowILFunction):
    for block in dfil_fx.basic_blocks:
        range_txt = f'{block.il_block.start:3} {block.il_block.end}'
        print(f'Block {block.block_id} with {len(block.data_nodes)} nodes: {block.il_block}, {range_txt}')
        for dn in block.data_nodes:
            # dn.display()
            # optxt = dn.get_operation_txt()
            hlil_index = dn.base_instr.instr_index
            # in_txt = ' '.join(edge.short_in_txt() for edge in parser.in_edges[dn.node_id])
            # out_txt = ' '.join(edge.short_out_txt() for edge in parser.out_edges[dn.node_id])

            dfil_txt = dfil_fx.get_dfil_txt(dn)
            # print(f'{dn.node_id:2} {hlil_index:2} {optxt:20}  {in_txt:20} {out_txt:30} {dn.format_tokens():40}')
            print(f'{dn.node_id:2} {hlil_index:2} {dfil_txt:60} {dn.format_tokens():40}')


def analyze_function(bv: binaryninja.BinaryView,
                     fx: binaryninja.Function):
    parser = ILParser()
    dfil_fx = parser.parse(list(fx.hlil.ssa_form))

    print_dfil(dfil_fx)

    for block in parser.data_blocks:

        print(block.get_txt())
        for oe in block.edges_out:
            print(f'   {oe.edge_type.name:16} {oe.out_block.block_id} {oe.data_node.node_id if oe.data_node else ""}')

    test_node = parser.data_blocks[0].data_nodes[0]
    display_node_tree(dfil_fx, test_node)

    # dfil_fx.graph_flows_from(test_node)

    dfil_fx.graph()


def analyze_hlil_instruction(bv: binaryninja.BinaryView,
                             instr: highlevelil.HighLevelILInstruction):
    ssa = instr.ssa_form

    parser = ILParser()
    dfil_fx = parser.parse(list(ssa.function.ssa_form))
    dfil_node: DataNode

    import binaryninjaui
    ctx = binaryninjaui.UIContext.activeContext()
    h = ctx.contentActionHandler()
    a = h.actionContext()
    token_state = a.token
    var = binaryninja.Variable.from_identifier(ssa.function.ssa_form, token_state.token.value)
    ts : binaryninja.architecture.InstructionTextToken = token_state.token
    ssa_var = None
    # print(f' var = {var} {type(ts)} {ts.text}')
    if '#' in ts.text:
        var_name, var_version = ts.text.split('#')
        ssa_var = binaryninja.mediumlevelil.SSAVariable(var, int(var_version))

    if ssa_var:
        dfil_node = dfil_fx.vars_to_nodes.get(repr(ssa_var), None)
        print(f'Made SSA var, got node {dfil_node.node_id}')
    else:
        # No variable selected, just use the whole instruction
        dfil_node = dfil_fx.nodes_by_il.get(ssa.instr_index, None)

    if dfil_node != None:
        dfil_fx.graph_flows_from(dfil_node)
    else:
        print(f'Could not find instruction {ssa.instr_index} {ssa.expr_index}')


def handle_function(bv: binaryninja.BinaryView,
                    fx: binaryninja.Function):

    parser = ILParser()
    dfx: DataFlowILFunction = parser.parse(list(fx.hlil.ssa_form))
    #print_dfil(dfx)

    for bb in dfx.basic_blocks:
        in_edges, out_edges = dfx.get_bb_edges(bb)

        in_txt = ','.join(f'{e.in_block.block_id}' for e in in_edges)
        out_txt = ','.join(f'{e.out_block.block_id}{e.edge_type.short()}' for e in out_edges)

        print(f'Block {bb.block_id}    {in_txt:10} {out_txt:10}')

    for n in dfx.all_nodes.values():
        dfx.print_verbose_node_line(n)

    partition = dfx.partition_basic_slices()
    slice_objs = []
    for nodes in partition:

        nids = set([n.node_id for n in nodes])
        interior_edges = dfx.get_interior_edges(nids)

        expressions = [n.get_expression() for n in nodes]

        #data_slice = DataSlice(nodes, interior_edges, expressions)
        #slice_objs.append(data_slice)

        # s.display_verbose(dfx)

        # print('FOLDED: ********************')
        #folded = fold_const(data_slice)
        #folded.display_verbose(dfx)
        xslice = ExpressionSlice(nodes)
        xslice.fold_constants()
        xslice.display_verbose(dfx)

def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('binary')
    parser.add_argument('--function', metavar='NAME', nargs='+')

    args = parser.parse_args()

    with binaryninja.open_view(args.binary) as bv:
        print(f'bv has {len(bv.functions)} functions')
        for fxname in args.function:
            funcs = bv.get_functions_by_name(fxname)
            funcs = [func for func in funcs if not func.is_thunk]
            assert(len(funcs) == 1)
            fx = funcs[0]
            handle_function(bv, fx)


if __name__ == "__main__":
    main()