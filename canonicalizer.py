from collections import defaultdict

try:
    from .dfil import *
    from .expressionslice import ExpressionSlice
    from .dfil_function import *
except ImportError:
    from dfil import *
    from expressionslice import ExpressionSlice
    from dfil_function import *

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

    def get_canonical_node_id(self, node: DataNode):
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

        sorted_block_ids = [bid for bid, cbid in sorted(self.canonical_block_ids.items(), key=lambda x: x[1])]
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

