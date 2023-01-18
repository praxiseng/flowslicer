import json
import os.path
import sys

from collections import defaultdict
from typing import List, Generator

from multiprocessing import Pool, Value, Process, Queue
import cbor2

import binaryninja.highlevelil
from binaryninja.mediumlevelil import SSAVariable
from binaryninja import flowgraph, BranchType, HighLevelILOperation

from collections.abc import Mapping
from dfil import DataNode, DataFlowEdge, TokenExpression

try:
    from .dfil import *
except ImportError:
    from dfil import *

verbosity = 1

class DFILCFG:
    def __init__(self,
                 basic_blocks: list[DataBasicBlock],
                 control_edges: list[ControlEdge]):
        self.basic_blocks = basic_blocks
        self.control_edges = control_edges

        self.block_id_to_block = {}
        self.edges_out = defaultdict(list)

        for bb in basic_blocks:
            self.block_id_to_block[bb.block_id] = bb

        for edge in control_edges:
            self.edges_out[edge.in_block.block_id].append(edge)

        # self.display_cfg()

    def get_control_edges(self, block_ids: set[int]):
        return [ce for ce in self.control_edges if
                ce.in_block.block_id in block_ids or
                ce.out_block.block_id in block_ids]

    def get_outging_edges(self, block_id: int) -> list[ControlEdge]:
        return self.edges_out.get(block_id, [])

    def get_bb_from_id(self, bb_id):
        return self.block_id_to_block.get(bb_id, None)

    def display_cfg(self):
        print('CFG:')
        for ce in self.control_edges:
            print(ce.get_txt())
            #data_node_id = f'N{ce.data_node.node_id}' if ce.data_node else ''
            #print(f'{ce.edge_type.name:14} BB{ce.in_block.block_id}->BB{ce.out_block.block_id} {data_node_id}')
        print('END CFG')

@dataclass(init=True, frozen=True)
class DataSlice:
    # TODO: check if this class is still used.  It may be superseded by ExpressionSlice.
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
            n = expression.base_node
            print(f'BB {n.block_id:2} Node {n.node_id:2} Expression {"".join(expression.get_text())}')

class LimitExceededException(Exception):
    """ Raised when a particular analysis has been run too many times """
    pass


class ExpressionSlice:
    ''' To fold expressions, we need a different notion of nodes and edges that expands
        lists into descriptions. '''
    def __init__(self, nodes: list[DataNode], analysis_limit=1000000):
        self.nodes = nodes
        self.expressions: list[TokenExpression] = []
        self.xmap: dict[int, TokenExpression] = {}
        self.input_nodes = set()
        self.counters = defaultdict(int)
        self.analysis_limit = analysis_limit

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

    def _limit_count(self, function_name):
        count = self.counters[function_name]
        self.counters[function_name] = count+1
        if count > self.analysis_limit:
            print(f'Iteration reached limit: {function_name} is {count}.  {len(self.nodes)} nodes, {len(self.expressions)} expressions')
            for name, count in self.counters.items():
                print(f'  {name:20} {count}')
            raise LimitExceededException()

    def remove_ints(self, min_value=0x1000):
        for expression in self.expressions:
            expression.remove_ints(min_value)

    def fold_node(self, node_id):
        assert(node_id in self.xmap)

    def bypass_edges(self, current_expr, out_expr):
        ''' Creates a new token list bypassing the current expression.
            The resulting token list can then be embedded in out_expr.
            If all uses of an expression are eliminated, then current_expr
            can be eliminated. '''

        self._limit_count('bypass_edges')

        tokens = current_expr.tokens[:]

        incoming: list[ExpressionEdge] = current_expr.getIncoming()

        ''' Make new ExpressionEdge objects that skip the current expression '''
        for in_edge in incoming:
            # print(f'\nTokenA {in_edge.in_expr.base_node.node_id:2} {in_edge.in_expr.get_text()}')
            # print(f'TokenB {expr.base_node.node_id:2} {expr.get_text()}')
            # print(f'TokenC {out_edge.out_expr.base_node.node_id:2} {out_edge.out_expr.get_text()}')

            new_edge = ExpressionEdge(in_edge.in_expr, out_expr)
            for index, token in enumerate(current_expr.tokens):
                if in_edge == token:
                    tokens[index] = new_edge
            in_edge.in_expr.uses.append(new_edge)

        return tokens

    def embed_edge_tokens(self,
                          edge : ExpressionEdge,
                          embed_tokens):
        for index, token in enumerate(edge.out_expr.tokens):
            if edge == token:
                edge.out_expr.tokens[index:index + 1] = embed_tokens
                break
        else:
            print(f'WARNING: could not find token for out edge')


    def getAddressSet(self):
        address_set = set()

        for node in self.nodes:
            match node.il_expr:
                case binaryninja.highlevelil.HighLevelILInstruction() as il_op:
                    address_set.add(il_op.address)
        return address_set

    def fold_expression(self, expr):
        """ For every use, embed a copy of the current expression, but with incoming ExpressionEdges
            substituted

            For example:
               To fold the following expression 20 into expression 30:

               20: DFIL_ADD(10->20, 11->20)
               30: DFIL_DEREF(20->30)

               We must create new edges that skip 20, resulting in the following folded expression:

               30: DFIL_DEREF(DFIL_ADD(10->30, 11->30))

        """

        self._limit_count('fold_expression')
        # note: cannot clean incoming edges in bypass_edges because it is used multiple times.
        incoming: list[ExpressionEdge] = expr.getIncoming()

        # print(f'Edges {expr.base_node.node_id} out {expr.uses}, {incoming}')
        for out_edge in expr.uses:
            ''' Make new ExpressionEdge objects that skip the current expression '''
            tokens = self.bypass_edges(expr, out_edge.out_expr)
            self.embed_edge_tokens(out_edge, tokens)

        for in_edge in incoming:
            in_edge.in_expr.uses.remove(in_edge)

    def fold_remove_nodes(self, remove_condition):
        self._limit_count('fold_remove_nodes')

        new_expressions = []
        for expr in self.expressions:
            if remove_condition(expr):
                self.fold_expression(expr)
            else:
                new_expressions.append(expr)

        self.expressions = new_expressions


    def fold_single_use(self):
        return self.fold_remove_nodes(lambda expr: len(expr.uses) == 1)

    def fold_const(self):
        return self.fold_remove_nodes(lambda expr: expr.base_node.operation==DataFlowILOperation.DFIL_DECLARE_CONST)

    def display_verbose(self, dfx):
        print(f'Input: {self.input_nodes}')
        #for n in self.nodes:
        #    dfx.print_verbose_node_line(n)

        for expression in self.expressions:
            use_nodes = [use.out_expr.base_node.node_id for use in expression.uses]
            use_txt = ','.join(str(u) for u in use_nodes)
            node = expression.base_node
            print(f'BB {node.block_id:2} Node {node.node_id:2} Use {use_txt:10} Expression {"".join(expression.get_text())}')


class Canonicalizer:
    """
    The Canonicalizer attempts to convert a DFIL ExpressionSlice into a common representation.  This is challenging
    due to the following considerations:

    1. Expressions within the same basic block may be re-ordered arbitrarily.
    1a.  We can force an ordering based on operation using a sort operation.
    1b.  We need to be sure to not include any arbitrary ID numbers such as DataNode or DataFlowBasicBlock IDs, as they
         may not be the same value or even same sort order in other matching instances.
    1c.  We can remove identifiers such as Node IDs, but preserve some flags/attributes such as whether the value is
         an input to the ExpressionSlice or another expression in the slice, or perhaps storage attributes (e.g.
         local/global, or program section for globals).  These are removed for the purposes of sort order.
    2. The graphs themselves can provide some ordering.
    2a.  We could potentially use the graph information for a topological sort.
    2b.  However, when the graphs (data and control flow) are cyclic, they cannot be used for a topological sort since
         they do not form a poset.
    2c.  With the cyclic paths, we could arbitrarily nominate "root" nodes to anchor the cycle, and permute
         possibilities with the database storage and matching.  But that seems complicated, so I will skip this
         approach for now in favor of expression folding.
         * To break CFG cycles, we could use the contents of each basic block to sort and pick the "root" nodes.
    3. Expression folding can add more ordering opportunities
    3a.  Take the following set of example expressions, which dereference two struct members:

            1.  var1 = x+8
            2.  var2 = *var1
            3.  var3 = x+0x20
            4.  var4 = *var3

         Since we remove the var numbers, expressions 2 and 4 would be identical and not have a sort order.  If we
         fold the expressions to the following, we can capture the graph ordering information and use the more complex
         expressions to increase the likelihood of a unique sort order:

            1. var2 = *(x+8)
            2. var4 = *(x+0x20)
    3b.  This approach still has issues with cyclic graphs, so care must be taken when selecting which nodes to fold


    Approach

    * It is assumed that some folding for the input ExpressionSlice is already done to increase the chance of a
      consistent total ordering
    * Canonicalizer starts by sorting expressions with all ID numbers removed.
    * Then Canonicalizer assigns canonical ID numbers to the expressions and basic blocks based on the order
      of appearance in the list of expressions.
    * Next, Canonicalizer sorts again with the ID numbers and outputs.  This should capture additional ordering
      based on actual parameters (e.g. two expressions using the same value produced earlier vs. a different one).
    * Finally, Canonicalizer outputs a canonical representation for matching.  This form includes canonical IDs, as well
      as basic block edges.

    Considerations

    * The above approach is very rough and could use a lot of work.  It is hard to combine partial ordering information
      with the dual overlaid graphs with the other fields.
    * Data slices may sit on basic blocks that are not directly connected, but have various paths.  We may want to do
      graph analysis to characterize the nature of the paths between the basic blocks.  For example, we should query
      whether one dominates another, or whether all paths from one lead to that other.
    * It might be easier to write a custom comparison-based sort.  That way the comparator can check the partial
      orderings to see if they give a result before moving on.  No great way to do that with Python's key-based sort.

    """
    def __init__(self,
                 slice: ExpressionSlice,
                 cfg: DFILCFG,
                 includeBB: bool = True,
                 verbose=False):
        self.slice = slice
        self.cfg = cfg

        self.includeBB = includeBB

        # Map expression indicies to canonical ID numbers
        self.canonical_expressions = {}

        # Map input nodes (NIDs) to canonical ID numbers
        self.canonical_nids = {}

        # Map basic block ID to canonical block ID
        self.canonical_block_ids = {}

        self.expressions = self.slice.expressions[:]

        # We first sort using a key agnostic to exact node IDs.
        # This may not be the best - multiple expressions may sort similarly, and have
        # an arbitrary ordering.  We could maybe pull graph dependencies for the ordering, but
        # it is unclear how to break cycles.
        self.expressions = sorted(self.expressions, key=self.expression_sort_key())
        if verbose:
            self.display_canonical_state('Stage 1: ')

        # Give canonical numbers based on order of appearance.
        self.canonical_numeralize()

        self.expressions = sorted(self.expressions, key=self.expression_sort_key())
        if verbose:
            self.display_canonical_state('Stage 2: ')

    def get_canonical_expression_id(self, expr: TokenExpression):
        nid = expr.base_node.node_id
        return self.canonical_expressions.get(nid, None)

    def get_canonical_block_id(self, expr: TokenExpression):
        bb_id = expr.base_node.block_id
        return self.canonical_block_ids.get(bb_id, None)

    def get_canonical_node_id(self, node : DataNode):
        return self.canonical_nids.get(node.node_id, None)

    def get_canonical_token_value(self, token):
        match token:
            case ExpressionEdge() as ee:
                cnid = self.get_canonical_expression_id(ee.in_expr)
                return "EXPRESSION" if cnid is None else f"X{cnid}"
            case TokenExpression() as exp:
                cnid = self.get_canonical_expression_id(exp)
                return "EXPRESSION" if cnid is None else f"X{cnid}"
            case DataNode() as dn:
                cnid = self.get_canonical_node_id(dn)
                return "INPUT" if cnid is None else f"N{cnid}"
            case DataBasicBlock() as bb:
                cbb_id = self.canonical_block_ids.get(bb.block_id)
                return "BLOCK" if cbb_id is None else f'BB{cbb_id}'
            case float():
                return f'float:{token}'
            case int():
                return f'0x{token:x}'
            case str():
                return token
            case _:
                return None

    def get_canonical_tokens(self, expr: TokenExpression):
        tokens = []

        cbid = self.get_canonical_block_id(expr)
        if cbid is not None:
            tokens.append(f'BB{cbid}')
            tokens.append(' ')

        cnid = self.canonical_expressions.get(expr.base_node.node_id, None)
        if cnid is not None:
            tokens.append(f'X{cnid}')
            tokens.append(' ')

        for token in expr.tokens:
            value = self.get_canonical_token_value(token)
            if value is None:
                print(f"Unknown type: {type(token)}: {token}")
            else:
                tokens.append(value)

        return tokens

    def expression_sort_key(self):
        def expression_sort_key_fx(expr: TokenExpression):
            key = []

            for token in expr.tokens:
                value = self.get_canonical_token_value(token)
                if value is None:
                    print(f"Unknown type: {type(token)}: {token}")
                else:
                    key.append(value)

            return key
        return expression_sort_key_fx

    def canonical_numeralize(self):
        ''' Assigned canonical ID numbers to data nodes, expressions, and basic blocks that appear in the current
            set of expressions. '''
        cexp_counter = 0
        node_counter = 0
        block_id_counter = 0

        for expr in self.expressions:
            node = expr.base_node
            nid = node.node_id
            if nid not in self.canonical_expressions:
                self.canonical_expressions[nid] = cexp_counter
                cexp_counter += 1

            bb_id = node.block_id
            if bb_id not in self.canonical_block_ids:
                self.canonical_block_ids[bb_id] = block_id_counter
                block_id_counter += 1

            for token in expr.tokens:
                match token:
                    case DataNode() as dn:
                        if dn.node_id not in self.canonical_nids:
                            self.canonical_nids[dn.node_id] = node_counter
                            node_counter += 1

    def get_canonical_edge_tokens(self, edge: ControlEdge):
        in_bb = self.get_canonical_token_value(edge.in_block)
        out_bb = self.get_canonical_token_value(edge.out_block)

        tokens = [in_bb, '->', out_bb, ' ', edge.edge_type.name]

        if edge.data_node:
            cnid = self.canonical_expressions.get(edge.data_node.node_id, None)
            node_txt = ''
            if cnid is not None:
                tokens.append(' ')
                tokens.append(f'X{cnid}')

        return tokens

    def get_relevant_cfg_edges(self) -> list[ControlEdge]:
        block_ids = set(self.canonical_block_ids.keys())

        edges = []

        sorted_block_ids = [bid for bid, cbid in sorted(self.canonical_block_ids.items(), key=lambda x:x[1])]
        for block_id in sorted_block_ids:
            out_edges = self.cfg.get_outging_edges(block_id)
            for edge in out_edges:
                has_related_data_node = False
                if edge.data_node:
                    has_related_data_node = self.canonical_expressions.get(edge.data_node.node_id, None) is not None
                if has_related_data_node or edge.out_block.block_id in block_ids:
                    edges.append(edge)

        return edges


    def get_canonical_lines(self):
        lines = []
        for expr in self.expressions:
            tokens = self.get_canonical_tokens(expr)
            lines.append(tokens)

        edges = self.get_relevant_cfg_edges()
        for edge in edges:
            tokens = self.get_canonical_edge_tokens(edge)
            lines.append(tokens)

        return sorted(lines)

    def get_canonical_text(self):
        lines = self.get_canonical_lines()
        return '\n'.join(["".join(str(token) for token in tokens) for tokens in lines])

    def get_simple_canonical_text(self):
        return self.get_canonical_text().replace("DFIL_", "").replace("DECLARE_CONST", "")

    def display_canonical_state(self, prefix=''):
        print(f'{prefix}')

        print(self.get_simple_canonical_text())


def fold_const(data_slice: DataSlice) -> DataSlice:
    const_node_nids = [n.node_id for n in data_slice.nodes if n.operation == DataFlowILOperation.DFIL_DECLARE_CONST]
    new_nodes: list[DataNode] = [n for n in data_slice.nodes if n.operation != DataFlowILOperation.DFIL_DECLARE_CONST]
    new_edges: list[DataFlowEdge] = [e for e in data_slice.edges if e.in_node.node_id not in const_node_nids]
    new_expressions: list[TokenExpression] = []
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


# These ops define partition boundaries.
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

        self.cfg = DFILCFG(basic_blocks, control_edges)

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

        # print(f'\nExpanding {start_nid}')
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

            # print(f'Expand {nid:4}  {str(remaining_nodes):30} {nodes_inout_txt} {nids_in_slice}')
            for n in connected_nodes:
                if n.node_id in nids_in_slice:
                    continue
                # if n.node_id in remaining_nodes:
                #     continue
                # assert (n.node_id in self.all_nodes)
                if n.node_id in self.all_nodes:
                    nids_in_slice.add(n.node_id)
                    if n.operation not in op_blacklist:
                        remaining_nodes.add(n.node_id)
                else:
                    if verbosity >= 2:
                        print(f'Node {n.node_id} not in all_nodes: {n.get_expression().get_text()}')

                # We capture blacklisted nodes in the slice, but don't traverse their edges

                # print(f'Node {n.node_id}  {"not " if n.operation not in op_blacklist else ""}in blacklist')

        assert (all(nid in self.all_nodes for nid in nids_in_slice))
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

            assert(all(nid in self.all_nodes for nid in slice_nids))

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
        if not all(operands):
            print(f'ERROR - Node has null operands')
            print(f'Expression: {expr.address:x} {expr}')
            print(f'Operands: {operands}')
            return


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

    def _unimplemented(self, expr, operands):
        node_id = self.next_node_id
        self.next_node_id += 1
        return DataNode(self.current_base_instr, expr, operands, node_id,
                        self.current_data_bb.block_id,
                        DataFlowILOperation.DFIL_UNKNOWN)

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
            case highlevelil.HighLevelILDerefFieldSsa() as deref_field_ssa:
                node = self._node(expr, [self._recurse(deref_field_ssa.src)])
            case highlevelil.HighLevelILMemPhi() | highlevelil.HighLevelILVarDeclare() | highlevelil.HighLevelILNoret():
                # These will be at the highest level, so we don't need to worry about returning None as an operand
                node = None
            case highlevelil.HighLevelILCase() as hlil_case:
                assert(len(hlil_case.operands) == 1)
                op1 = hlil_case.operands[0]
                operands = [self._recurse(operand) for operand in op1]
                node = self._node(expr, operands)
            case SSAVariable() as ssa:
                node = self._var(ssa)
            case highlevelil.HighLevelILGoto() | highlevelil.HighLevelILLabel():
                # The goto is already represented by the basic block edges
                node = None
                raise Exception()
            case highlevelil.HighLevelILWhileSsa() | highlevelil.HighLevelILDoWhileSsa():
                # TODO: check that the block in the first operand gets processed
                operands = [self._recurse(operand) for operand in expr.operands[1:]]
                node = self._node(expr, operands)
            case commonil.BaseILInstruction():
                operands = [self._recurse(operand) for operand in expr.operands]
                node = self._node(expr, operands)
            case list():
                operands = [self._recurse(operand) for operand in expr]

                assert(all(operands))
                node = self._node(expr, operands)
            case binaryninja.lowlevelil.ILIntrinsic() as intrinsic:
                iname = self._var(str(intrinsic.name))
                try:
                    outputs = self._recurse(intrinsic.outputs)
                except:
                    outputs = []
                try:
                    inputs = self._recurse(intrinsic.inputs)
                except:
                    inputs = []
                #operands = [iname, outputs, inputs]
                operands = [iname] + inputs
                node = self._node(expr, operands)
            case _:
                if verbosity >= 1:
                    print(f'{self.current_base_instr.address:x} Unimplemented: {expr}, in {self.current_base_instr}')
                    print(f'{type(expr)}')
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
                match il_instr:
                    case highlevelil.HighLevelILGoto() | highlevelil.HighLevelILLabel():
                        # The goto is already represented by the basic block edges
                        continue

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

def process_function(args,
                     bv: binaryninja.BinaryView,
                     fx: binaryninja.Function,
                     output: Generator[None, dict, None]):

    if not fx:
        print(f'Passed NULL function to process!')
        return
    if not fx.hlil:
        print(f'HLIL is null for function {fx}')
        return

    parser = ILParser()
    dfx: DataFlowILFunction = parser.parse(list(fx.hlil.ssa_form))
    #print_dfil(dfx)

    partition = dfx.partition_basic_slices()
    if verbosity >= 3:
        print(f'Function has {len(partition):3} partitions {fx}')

    for nodes in partition:
        option_set_texts = []
        for option in args.option_permutations:
            xslice = ExpressionSlice(nodes)
            if verbosity >= 4:
                print('Expression Slice:')
                xslice.display_verbose(dfx)

            if option.get('removeInt', None) is not None:
                xslice.remove_ints(option['removeInt'])

            try:
                xslice.fold_const()
            except LimitExceededException:
                pass

            try:
                xslice.fold_single_use()
            except LimitExceededException:
                pass

            canonical = Canonicalizer(xslice, dfx.cfg)
            canonical_text = canonical.get_canonical_text()

            if canonical_text in option_set_texts:
                # Same slice was produced with a different option set
                # No need to duplicate
                continue

            if verbosity >= 3:
                print(f'Canonical text\n{canonical_text}')

            slice_data = dict(
                option=option,
                file=dict(
                    name=os.path.basename(bv.file.filename),
                    path=bv.file.filename,
                ),
                function=dict(
                    name=fx.name,
                    address=fx.start
                ),
                addressSet=sorted(xslice.getAddressSet()),
                canonicalText=canonical_text,
            )

            option_set_texts.append(canonical_text)
            output.send(slice_data)


def handle_function(args,
                    bv: binaryninja.BinaryView,
                    fx: binaryninja.Function):

    display = display_json()
    display.send(None)

    process_function(args, bv, fx, display)


def display_json():
    try:
        while True:
            data = yield
            print(json.dumps(data, indent=4, default=str))
    finally:
        print("Iteration stopped")


def handle_functions_by_name(args,
                             bv: binaryninja.BinaryView,
                             names: list[str],
                             output: Generator[None, dict, None]):
    for fxname in names:
        funcs = bv.get_functions_by_name(fxname)
        funcs = [func for func in funcs if not func.is_thunk]
        assert (len(funcs) == 1)
        fx = funcs[0]
        process_function(args, bv, fx, output)


def _handle_binary(args,
                   binary_path: str,
                   output: Generator[None, dict, None]):

    with binaryninja.open_view(binary_path) as bv:
        if verbosity >= 2:
            print(f'bv has {len(bv.functions)} functions')
        if args.function:
            handle_functions_by_name(args, bv, args.function, output)
        else:
            for fx in bv.functions:
                if verbosity >= 2:
                    print(f'Analyzing {fx}')
                process_function(args, bv, fx, output)

class Logger:
    def __init__(self, output_streams, current_file_path):
        self.output_streams = output_streams
        self.current_file_path = current_file_path

    def write(self, buf):
        if not self.output_streams:
            return
        file_name = os.path.basename(self.current_file_path)
        for line in buf.rstrip().splitlines():
            txt = f'{file_name:20}{line}'
            for stream in self.output_streams:
                print(txt, file=stream)

    def flush(self):
        pass

    def isatty(self):
        return False

    def closed(self):
        return False


def dump_cbor(out_fd):
    try:
        while True:
            data = yield
            out_fd.write(cbor2.dumps(data))
    finally:
        pass
        #print("Iteration stopped")

class Main:
    def __init__(self):
        import argparse

        parser = argparse.ArgumentParser()
        parser.add_argument('binary')
        parser.add_argument('--function', metavar='NAME', nargs='+')
        parser.add_argument('--output', default='data_flow_slices_cbor', metavar='PATH', nargs='?')
        parser.add_argument('--force-update', action='store_true')
        parser.add_argument('--parallelism', metavar='N', type=int, default=1)
        parser.add_argument('-v', '--verbose', action='count', default=0)
        global verbosity
        global files_processed
        global total_files

        self.args = parser.parse_args()

        self.args.option_permutations = [
            dict(),
            dict(removeInt=0x1000),
        ]

        verbosity = self.args.verbose
        files_processed = Value('i', 0)
        total_files = Value('i', 0)

        if self.args.output:
            os.makedirs(self.args.output, exist_ok=True)

        if os.path.isdir(self.args.binary):
            self.handle_folder()
        else:
            self.handle_binary(self.args.binary)

    def set_vars(self, processed, total):
        global files_processed
        global total_files
        files_processed = processed
        total_files = total

    def get_output_path(self, input_path, extension):
        return os.path.join(self.args.output, os.path.basename(input_path) + extension)

    def _handle_binary(self, binary_path: str):
        out_path = self.get_output_path(binary_path, '.temp')
        final_path = self.get_output_path(binary_path, '.cbor')

        if verbosity >= 1:
            print(f'File {files_processed.value+1} of {total_files.value}: {binary_path}')

        files_processed.value += 1

        if verbosity >= 2:
            print(f'Opening {binary_path}')
        if verbosity >= 2:
            print(f'Output: {out_path}')

        if os.path.exists(final_path):
            if self.args.force_update:
                os.remove(final_path)
            else:
                return

        with open(out_path, 'wb') as fd:
            write_file = dump_cbor(fd)
            write_file.send(None)

            _handle_binary(self.args, binary_path, write_file)

        os.replace(out_path, final_path)

    def handle_binary(self, binary_path: str):
        try:
            self._handle_binary(binary_path)
        except KeyboardInterrupt:
            sys.exit(0)

    def process_binaries_serially(self, file_paths):
        print("PROCESSING SERIALLY")
        files_processed.value = 0
        total_files.value = len(file_paths)

        for idx, path in enumerate(file_paths):
            self.handle_binary(path)

    def process_queue(self, paths: Queue, results: Queue, files_processed, total_files):
        original_stdout = sys.stdout
        original_stderr = sys.stderr
        self.set_vars(files_processed, total_files)
        binaryninja.disable_default_log()
        while True:
            path = paths.get()
            if path is None:
                break
            log_path = self.get_output_path(path, '.log')
            with open(log_path, 'w') as log_fd:
                sys.stdout = Logger([original_stdout], path)
                sys.stderr = Logger([original_stderr], path)
                binaryninja.log.log_to_file(binaryninja.log.LogLevel.WarningLog, log_path, False)
                # results.put(dict(processing=path))
                self.handle_binary(path)

    def process_binaries_in_parallel(self, file_paths):
        files_processed.value = 0
        total_files.value = len(file_paths)

        pathQueue = Queue()
        resultQueue = Queue()
        procs = []

        try:
            for _ in range(self.args.parallelism):
                proc = Process(target=self.process_queue, args=(pathQueue, resultQueue, files_processed, total_files))
                procs.append(proc)
                proc.start()

            for path in file_paths:
                pathQueue.put(path)

            for _ in range(self.args.parallelism):
                pathQueue.put(None)

            for proc in procs:
                proc.join()
        except KeyboardInterrupt:
            for proc in procs:
                proc.terminate()

    def handle_folder(self):
        file_paths = []
        for root, dirs, files in os.walk(self.args.binary):
            for file in files:
                file_path = os.path.join(root, file)
                file_paths.append(file_path)

        if self.args.parallelism <= 1:
            self.process_binaries_serially(file_paths)
        else:
            self.process_binaries_in_parallel(file_paths)


if __name__ == "__main__":
    Main()