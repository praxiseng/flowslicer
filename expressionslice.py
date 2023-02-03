from collections import defaultdict

import binaryninja

try:
    from .dfil import *
except ImportError:
    from dfil import *

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
        self.counters[function_name] = count + 1
        if count > self.analysis_limit:
            print(
                f'Iteration reached limit: {function_name} is {count}.  {len(self.nodes)} nodes, {len(self.expressions)} expressions')
            for name, count in self.counters.items():
                print(f'  {name:20} {count}')
            raise LimitExceededException()

    def remove_ints(self, min_value=0x1000):
        for expression in self.expressions:
            expression.remove_ints(min_value)

    def fold_node(self, node_id):
        assert (node_id in self.xmap)

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
                          edge: ExpressionEdge,
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
        return self.fold_remove_nodes(lambda expr: expr.base_node.operation == DataFlowILOperation.DFIL_DECLARE_CONST)

    def display_verbose(self, dfx):
        print(f'Input: {self.input_nodes}')
        # for n in self.nodes:
        #    dfx.print_verbose_node_line(n)

        for expression in self.expressions:
            use_nodes = [use.out_expr.base_node.node_id for use in expression.uses]
            use_txt = ','.join(str(u) for u in use_nodes)
            node = expression.base_node
            print(
                f'BB {node.block_id:2} Node {node.node_id:2} Use {use_txt:10} Expression {"".join(expression.get_text())}')
