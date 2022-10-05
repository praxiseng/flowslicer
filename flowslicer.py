
from collections import defaultdict
from binaryninja.mediumlevelil import SSAVariable
from binaryninja import flowgraph, BranchType
from .dfil import *


class DataFlowILFunction:
    def __init__(self,
                 basic_blocks: list[DataBasicBlock],
                 data_flow_edges: list[DataFlowEdge],
                 control_edges: list[ControlEdge]):

        self.basic_blocks = basic_blocks
        self.control_edges = control_edges

        self.all_nodes = {}
        for bb in basic_blocks:
            for dn in bb.data_nodes:
                self.all_nodes[dn.node_id] = dn

        self.in_edges = defaultdict(list)
        self.out_edges = defaultdict(list)
        for edge in data_flow_edges:
            a, b = edge.in_node, edge.out_node
            self.out_edges[a.node_id].append(edge)
            self.in_edges[b.node_id].append(edge)

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


class ILParser:
    def __init__(self):
        self.next_node_id = 1
        self.data_blocks = []
        self.current_data_bb = None
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
        dn = DataNode(self.current_base_instr, expr, operands, node_id, dfil_op)
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
            case commonil.Constant():
                print(f'Constant {expr.constant}')
                node = self._var(expr.constant)
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
            case highlevelil.HighLevelILVarInit():
                operands = [self._recurse(expr.src)]
                node = self._node(expr, operands, expr.dest)
            case highlevelil.HighLevelILCallSsa():
                dest, params, dest_mem, src_mem = expr.operands
                operands = [self._recurse(operand) for operand in [dest] + params]
                node = self._node(expr, operands)
            case highlevelil.HighLevelILArrayIndexSsa():
                operands = [self._recurse(operand) for operand in [expr.src, expr.index]]
                node = self._node(expr, operands)
            case highlevelil.HighLevelILStructField():
                # .member_index has been None
                # TODO: consider unifying struct/array/deref under a generic GEP (a la LLVM) to unify dereferences.
                operands = [self._recurse(operand) for operand in [expr.src, expr.offset]]
                node = self._node(expr, operands)
            case highlevelil.HighLevelILAssignMemSsa():
                # Note that expr.dest tends to contain the memory reference, not the HighLevelILAssignMemSsa.  expr.dest
                # can be HighLevelILDerefSsa or HighLevelILArrayIndexSsa, for example
                operands = [self._recurse(operand) for operand in [expr.dest, expr.src]]
                node = self._node(expr, operands)
            case highlevelil.HighLevelILTailcall():
                operands = [self._recurse(operand) for operand in expr.params]
                node = self._node(expr, operands, expr.dest)
            case highlevelil.HighLevelILDerefSsa():
                node = self._node(expr, [self._recurse(expr.src)])
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

    def parse(self, il):
        for index, il_bb in enumerate(il):
            dbb = DataBasicBlock(il_bb, [], index, [], [], [], None)
            self.data_blocks.append(dbb)
            self.il_bb_to_dbb[il_bb] = dbb

        for il_bb in il:
            dbb = self.il_bb_to_dbb[il_bb]
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
                out_dbb = self.il_bb_to_dbb[out_edge.target]
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


def print_dfil(dfil_fx : DataFlowILFunction):
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
    print('Hello, world!')
    print(f'bv={bv}')
    print(f'fx={fx}')

    parser = ILParser()
    dfil_fx = parser.parse(fx.hlil.ssa_form)

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
    print(f'Analyze instr {instr.ssa_form}')