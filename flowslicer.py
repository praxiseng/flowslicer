import enum
import itertools
from collections import defaultdict

import binaryninja
from binaryninja import commonil
from typing import Any, Union
from dataclasses import dataclass
from binaryninja.mediumlevelil import SSAVariable
from binaryninja import highlevelil
from binaryninja.enums import HighLevelILOperation as HLIL_OP


def roundrobin(*iterables):
    "roundrobin('ABC', 'D', 'EF') --> A D E B F C"
    # Recipe credited to George Sakkis
    num_active = len(iterables)
    nexts = itertools.cycle(iter(it).__next__ for it in iterables)
    while num_active:
        try:
            for next in nexts:
                yield next()
        except StopIteration:
            # Remove the iterator we just exhausted from the cycle.
            num_active -= 1
            nexts = itertools.cycle(itertools.islice(nexts, num_active))

def tok_bin(*pargs):
    def tok(data_items):
        return list(roundrobin(pargs, data_items))
    return tok


class TokenGen:
    def __init__(self, weaves, sep, end):
        self.weaves = weaves
        self.sep = sep
        self.end = end



tokenizers = {
    HLIL_OP.HLIL_ADD: tok_bin('', '+'),
}



class DataFlowILOperation(enum.IntEnum):
    DFIL_UNKNOWN = -1
    DFIL_NOP = 0
    DFIL_BLOCK = 1
    DFIL_ADD = 2
    DFIL_CALL = 3
    DFIL_DECLARE_CONST = 4
    DFIL_DECLARE_VAR = 5
    DFIL_PHI = 6
    DFIL_RET = 7
    DFIL_CMP_E = 8
    DFIL_CMP_NE = 9
    DFIL_CMP_SGT = 80
    DFIL_CMP_SLE = 81
    DFIL_ASSIGN = 10
    DFIL_INIT = 11
    DFIL_DEREF = 12
    DFIL_IF = 13
    DFIL_ZX = 14
    DFIL_TAILCALL = 15
    DFIL_LOGIC_OR = 16
    DFIL_LOGIC_AND = 17
    DFIL_SX = 18
    DFIL_LSL = 19
    DFIL_SUB = 20

    DFIL_ARRAY_INDEX = 21
    DFIL_STORE = 22

globals().update(HLIL_OP.__members__)
globals().update(DataFlowILOperation.__members__)

hlil_to_dfil_operations = {
    HLIL_OP.HLIL_ADD: DFIL_ADD,
    HLIL_OP.HLIL_CALL_SSA: DFIL_CALL,
    HLIL_OP.HLIL_VAR_PHI: DFIL_PHI,
    HLIL_OP.HLIL_RET: DFIL_RET,
    HLIL_OP.HLIL_CMP_E: DFIL_CMP_E,
    HLIL_OP.HLIL_CMP_NE: DFIL_CMP_NE,
    HLIL_OP.HLIL_ASSIGN: DFIL_ASSIGN,
    HLIL_OP.HLIL_VAR_INIT_SSA: DFIL_INIT,
    HLIL_OP.HLIL_DEREF_SSA: DFIL_DEREF,
    HLIL_OP.HLIL_IF: DFIL_IF,
    HLIL_OP.HLIL_ZX: DFIL_ZX,
    HLIL_OP.HLIL_TAILCALL : DFIL_TAILCALL,
    HLIL_OP.HLIL_OR : DFIL_LOGIC_OR,
    HLIL_OP.HLIL_AND : DFIL_LOGIC_AND,
    HLIL_OP.HLIL_SX : DFIL_SX,
    HLIL_OP.HLIL_LSL : DFIL_LSL,
    HLIL_OP.HLIL_SUB : DFIL_SUB,
    HLIL_OP.HLIL_ARRAY_INDEX_SSA : DFIL_ARRAY_INDEX,
    HLIL_OP.HLIL_ASSIGN_MEM_SSA : DFIL_STORE,
    HLIL_OP.HLIL_CMP_SGT : DFIL_CMP_SGT,
    HLIL_OP.HLIL_CMP_SLE : DFIL_CMP_SLE,
}


def get_dfil_op(expr):
    match expr:
        case commonil.Constant() | int():
            return DataFlowILOperation.DFIL_DECLARE_CONST
        case SSAVariable():
            return DataFlowILOperation.DFIL_DECLARE_VAR
        case highlevelil.HighLevelILInstruction() as hlil_instr if hlil_instr.operation in hlil_to_dfil_operations:
            return hlil_to_dfil_operations[hlil_instr.operation]
        case _:
            return DataFlowILOperation.DFIL_UNKNOWN

@dataclass()
class DataBasicBlock:
    il_block: binaryninja.basicblock.BasicBlock
    data_nodes: list
    block_id: int

    edges_in: list
    edges_out: list
    out_edge_node: 'DataNode'

    def get_txt(self):
        return f'Block {self.block_id} {self.il_block.start}-{self.il_block.end} has {len(self.data_nodes)} nodes'





def wrap_binary(pre, infix, post):
    def apply(dn):
        op1, op2 = dn.operands
        return pre + op1.get_txt() + infix + op2.get_txt() + post
    return apply

datanode_txt_map = {
    DataFlowILOperation.DFIL_ADD:wrap_binary('(', ' + ', ')'),
}


@dataclass(frozen=True)
class DataNode:
    base_instr: commonil.BaseILInstruction
    il_expr: Union[commonil.BaseILInstruction, SSAVariable, binaryninja.variable.Variable, int]
    operands: list
    node_id: int
    operation: DataFlowILOperation

    def get_txt(self):
        opname = type(self.il_expr).__name__
        optxt = '?'

        match self.operation:
            case DataFlowILOperation.DFIL_CALL:
                call_target = self.operands[0].get_txt()
                call_operands = ",".join(operand.get_txt() for operand in self.operands[1:])
                return f'{self.node_id}:{call_target}({call_operands})'

        match self.il_expr:
            case int():
                return f'{self.node_id}:{hex(self.il_expr)}'
            case binaryninja.variable.Variable() as var:
                return f'{self.node_id}:{var.name}'
            case SSAVariable() as ssa:
                return f'{self.node_id}:{ssa.var.name}#{ssa.version}'
            case highlevelil.HighLevelILVarSsa(var=ssa):
                return f'{self.node_id}:{ssa.var.name}#{ssa.version}'

            case _:
                optxt = ",".join([child_node.get_txt() for child_node in self.operands])
                operation_txt = f'{self.operation.name if self.operation else opname}({optxt})'
                if self.operation in datanode_txt_map:
                    operation_txt = datanode_txt_map[self.operation](self)
                return f'{self.node_id}:{operation_txt}'

    def get_operation_txt(self):
        if self.operation == DataFlowILOperation.DFIL_UNKNOWN:
            return type(self.il_expr).__name__
        return self.operation.name

    def is_control_related(self):
        match self.il_expr:
            case highlevelil.HighLevelILIf():
                return True
            #case commonil.ControlFlow():
            #    return True
        return False

    def get_token_text(self):
        match self.il_expr:
            case int():
                return [hex(self.il_expr)]
            case binaryninja.variable.Variable() as var:
                return [f'{var.name}']
            case SSAVariable() as ssa:
                return [f'{ssa.var.name}#{ssa.version}']
            case highlevelil.HighLevelILVarSsa(var=ssa):
                return [f'{ssa.var.name}#{ssa.version}']
            case highlevelil.HighLevelILInstruction():
                # The .tokens return objects that are not strings and don't compare like strings
                return [str(tok) for tok in self.il_expr.tokens]
            case _:
                return [f'Unknown expr {self.il_expr}']

    def get_token_list(self):
        match self.operation:
            case DataFlowILOperation.DFIL_CALL:
                call_target = self.operands[0]
                call_operands = self.operands[1:]
                txt_tokens = [call_target, '(']
                for idx, operand in enumerate(self.operands[1:]):
                    if idx:
                        txt_tokens.append(',')
                    txt_tokens.append(operand)
                txt_tokens.append(')')
                return txt_tokens

        txt_tokens = self.get_token_text()
        # Hack-in the subexpressions by trying to find sub-lists of tokens
        for operand in self.operands:
            operand_tokens = operand.get_token_text()
            len_t = len(operand_tokens)
            for i in range(0, len(txt_tokens)-len_t+1):
                if all(txt_tokens[i+j]==operand_tokens[j] for j in range(len_t)):
                    txt_tokens[i:i+len_t] = [operand]
                    break
            else:
                txt_tokens.append(operand)

        return txt_tokens

    def format_tokens(self):
        tokens = self.get_token_list()
        txt = []
        for token in tokens:
            match token:
                case DataNode():
                    #print('It\'s a DataNode!')
                    txt.append(token.get_txt())
                case _:
                    txt.append(token)
        return ' '.join(txt)

    def display(self):
        optxt = self.get_operation_txt()
        return f'{self.node_id:2} {self.base_instr.instr_index:2} {optxt:20}  {self.format_tokens()}'


class EdgeType(enum.IntEnum):
    Move = 1  # copy, used in arithmetic, etc
    Load = 2      # Used directly as an address for a load operation
    Store = 3     # Used directly as an address for a store operation
    Branch = 4    # Used directly as an address for indirect branching or control flow

    def onechar(self):
        return self.name[0]

    def short(self):
        if self == EdgeType.Move:
            return ''
        return self.onechar()

@dataclass(frozen=True)
class Edge:
    in_node: DataNode
    out_node: DataNode
    edge_type: EdgeType
    operand_index: int

    def txt(self):
        return f'{self.in_node.node_id}.{self.edge_type.onechar()}->{self.out_node.node_id}.{self.operand_index}'

    def short_in_txt(self):
        #return f'{self.in_node.node_id}.{self.operand_index}'
        # Usually these are in order when displaying a node's incoming edges
        return f'{self.in_node.node_id}'

    def short_out_txt(self):
        return f'{self.edge_type.short()}{self.out_node.node_id}.{self.operand_index}'

class ControlEdgeType(enum.IntEnum):
    Unconditional = 0
    FalseBranch = 1
    TrueBranch = 2
    IndirectBranch = 3
    FunctionReturn = 4
    Undefined = 99

    @staticmethod
    def from_basic_block_edge(bbe: binaryninja.basicblock.BasicBlockEdge):
        branch_map = {
            binaryninja.enums.BranchType.UnconditionalBranch: ControlEdgeType.Unconditional,
            binaryninja.enums.BranchType.FalseBranch: ControlEdgeType.FalseBranch,
            binaryninja.enums.BranchType.TrueBranch: ControlEdgeType.TrueBranch,
            binaryninja.enums.BranchType.IndirectBranch: ControlEdgeType.IndirectBranch,
            binaryninja.enums.BranchType.FunctionReturn: ControlEdgeType.FunctionReturn,
        }
        return branch_map.get(bbe.type, ControlEdgeType.Undefined)


@dataclass(frozen=True)
class ControlEdge:
    in_block: DataBasicBlock
    out_block: DataBasicBlock
    edge_type: ControlEdgeType
    data_node: DataNode



class ILParser:
    def __init__(self):
        self.next_node_id = 1
        self.data_blocks = []
        self.current_data_bb = None
        self.current_base_instr = None


        self.il_bb_to_dbb = {}

        self.varnodes = {}

        self.nodes_by_id = {}

        # node_id -> list[Edge]
        self.out_edges = defaultdict(list)
        self.in_edges = defaultdict(list)

    def _node(self, expr, operands : []):
        node_id = self.next_node_id
        self.next_node_id += 1
        dn = DataNode(self.current_base_instr, expr, operands, node_id, get_dfil_op(expr))
        self.current_data_bb.data_nodes.append(dn)
        self.nodes_by_id[node_id] = dn

        edge_type = EdgeType.Move
        for operand_index, operand in enumerate(operands):
            e = Edge(operand, dn, edge_type, operand_index)
            self.out_edges[operand.node_id].append(e)
            self.in_edges[node_id].append(e)

        return dn

    def _var(self, instr, var):
        if var in self.varnodes:
            return self.varnodes[var]

        print(f'var {var}')
        dn = self._node(var, [])
        self.varnodes[var] = dn

        return dn

    def _recurse(self, expr):
        match expr:
            case commonil.Constant():
                node = self._var(expr, expr.constant)
            case int():
                node = self._var(expr, expr)
            case binaryninja.variable.Variable() as var:
                node = self._var(expr, var)
            case highlevelil.HighLevelILVarSsa(var=ssa):
                node = self._var(expr, ssa)
            case highlevelil.HighLevelILVarPhi() as var_phi:
                operands = [self._recurse(operand) for operand in var_phi.src]
                node = self._node(expr, operands)
            case highlevelil.HighLevelILCallSsa() as call_ssa:
                dest, params, dest_mem, src_mem = expr.operands
                operands = [self._recurse(operand) for operand in [dest] + params]
                # TODO: unpack multi-value outputs
                node = self._node(expr, operands)
            case highlevelil.HighLevelILTailcall() as call_ssa:
                operands = [self._recurse(operand) for operand in [expr.dest] + expr.params]
                node = self._node(expr, operands)
            case highlevelil.HighLevelILMemPhi() | highlevelil.HighLevelILVarDeclare():
                node = None
                pass
            case SSAVariable() as ssa:
                node =  self._var(expr, ssa)
            case commonil.BaseILInstruction() as instr:
                operands = [self._recurse(operand) for operand in expr.operands]
                node =  self._node(expr, operands)
            case _:
                print(f'Type not handled: {type(expr)} for {expr}')
                node = self._node(expr, [])

        return node

    def parse(self, il):
        for index, il_bb in enumerate(il):
            dbb = DataBasicBlock(il_bb, [], index, [], [], None)
            self.data_blocks.append(dbb)
            self.il_bb_to_dbb[il_bb] = dbb

        for il_bb in il:
            dbb = self.il_bb_to_dbb[il_bb]
            self.current_data_bb = dbb
            for il_instr in il_bb:
                self.current_base_instr = il_instr
                dn = self._recurse(il_instr)

        for dbb in self.data_blocks:
            out_edge_node = None
            if dbb.data_nodes and dbb.data_nodes[-1].is_control_related():
                out_edge_node = dbb.data_nodes[-1]

            for out_edge in dbb.il_block.outgoing_edges:
                out_dbb = self.il_bb_to_dbb[out_edge.target]
                et = ControlEdgeType.from_basic_block_edge(out_edge)
                ce = ControlEdge(dbb, out_dbb, et, out_edge_node)
                dbb.edges_out.append(ce)
                out_dbb.edges_in.append(ce)


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



def analyze_function(bv: binaryninja.BinaryView,
                     fx: binaryninja.Function,
                     *pargs, **kwargs):
    print('Hello, world!')
    print(f'bv={bv}')
    print(f'fx={fx}')

    data_blocks = []
    parser = ILParser()
    parser.parse(fx.hlil.ssa_form)

    for block in parser.data_blocks:
        print(f'Block {block.block_id} with {len(block.data_nodes)} nodes: {block.il_block}, {block.il_block.start:3} {block.il_block.end}')
        for dn in block.data_nodes:
            #dn.display()
            optxt = dn.get_operation_txt()
            hlil_index = dn.base_instr.instr_index
            in_txt = ' '.join(edge.short_in_txt() for edge in parser.in_edges[dn.node_id])
            out_txt = ' '.join(edge.short_out_txt() for edge in parser.out_edges[dn.node_id])

            dfil_txt = parser.get_dfil_txt(dn)
            #print(f'{dn.node_id:2} {hlil_index:2} {optxt:20}  {in_txt:20} {out_txt:30} {dn.format_tokens():40}')
            print(f'{dn.node_id:2} {hlil_index:2} {dfil_txt:60} {dn.format_tokens():40}')

    for block in parser.data_blocks:

        print(block.get_txt())
        for oe in block.edges_out:
            print(f'   {oe.edge_type.name:16} {oe.out_block.block_id} {oe.data_node.node_id if oe.data_node else ""}')