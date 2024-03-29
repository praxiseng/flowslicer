import binaryninja
import enum

from binaryninja import commonil, highlevelil
from dataclasses import dataclass
from binaryninja.enums import HighLevelILOperation as HlilOp
from typing import Union, Callable, Optional


def intersperse(lst, item):
    result = [item] * (len(lst) * 2 - 1)
    result[0::2] = lst
    return result


def call_formatter(prefix='', infix=','):
    def formatter(operands) -> str:
        return f'{prefix}{operands[0]}({infix.join(operands[1:])})'

    return formatter


def infix_formatter(infix, use_paren=True):
    def formatter(operands) -> str:
        result = infix.join(operands)
        if use_paren:
            result = f'({result})'
        return result

    return formatter


def prefix_formatter(prefix, infix=',', postfix=''):
    def formatter(operands) -> str:
        return f'{prefix}{infix.join(operands)}{postfix}'

    return formatter


class TokenExpression:
    def __init__(self,
                 base_node: 'DataNode',
                 tokens: 'list[Union[str, int, DataNode, ExpressionEdge]]',
                 folded_nids=set()):
        self.base_node = base_node
        self.tokens = tokens
        self.folded_nids = folded_nids | {base_node.node_id}
        self.uses: 'list[ExpressionEdge]' = []

    def get_full_text(self):
        tokens = []
        for tok in self.tokens:
            match tok:
                case DataNode() as dn:
                    tokens.append('N' + str(dn.node_id))
                case ExpressionEdge() as ee:
                    tokens.append('X' + str(ee.in_expr.base_node.node_id))
                case other:
                    tokens.append(str(tok))
        return "".join(tokens)

    def remove_ints(self, min_value=0x1000):
        """ Remove all integer constants greater than or equal to min_value and replace them with "_" """
        for index, token in enumerate(self.tokens):
            match token:
                case int():
                    if token >= min_value:
                        self.tokens[index]='_'

    def get_short_text(self):
        tokens = []
        for tok in self.tokens:
            match tok:
                case DataNode() as dn:
                    tokens.append('N' + str(dn.node_id))
                case ExpressionEdge() as ee:
                    tokens.append('X' + str(ee.in_expr.base_node.node_id))
                case int():
                    tokens.append(f'0x{tok:x}')
                case other:
                    strtok = str(tok)
                    replacements = {
                        "DFIL_DECLARE_CONST" : "",
                        "DFIL_CMP_NE" : "!=",
                        "DFIL_CMP_E" : "==",
                        "DFIL_PHI" : u"ϕ"
                    }
                    if strtok in replacements:
                        strtok = replacements[strtok]
                    if strtok.startswith('DFIL_'):
                        strtok = strtok[5:]
                    tokens.append(strtok)
        return "".join(tokens)

    def get_anonymous_tokens(self):
        tokens = []
        for tok in self.tokens:
            match tok:
                case DataNode() as dn:
                    tokens.append('N' + str(dn.node_id))
                case ExpressionEdge() as ee:
                    tokens.append('EDGE')
                case int():
                    tokens.append(f'0x{tok:x}')
                case other:
                    strtok = str(tok)
                    tokens.append(str(tok))
        return "".join(tokens)


    def get_text(self):
        return self.get_short_text()

    def getIncoming(self) -> 'list[ExpressionEdge]':
        return [token for token in self.tokens if isinstance(token, ExpressionEdge)]


    def edgeIndex(self, edge: 'ExpressionEdge'):
        for index, token in enumerate(self.tokens):
            if token == edge:
                return index


@dataclass(init=True, frozen=True, eq=True)
class ExpressionEdge:
    in_expr: TokenExpression
    out_expr: TokenExpression

    def outIndex(self):
        return self.out_expr.edgeIndex(self)

    def __str__(self):
        return f'<{self.in_expr.base_node.node_id}->{self.out_expr.base_node.node_id}>'

    def __repr__(self):
        return self.__str__()

@dataclass(frozen=True)
class DataFlowOp:
    value: int
    formatter: Callable[[list[str]], str] = None


class DataFlowILOperation(enum.Enum):
    DFIL_UNKNOWN = DataFlowOp(-1)
    DFIL_NOP = DataFlowOp(0)
    DFIL_BLOCK = DataFlowOp(1)
    DFIL_CALL = DataFlowOp(3, call_formatter())
    DFIL_DECLARE_CONST = DataFlowOp(4)
    DFIL_DECLARE_VAR = DataFlowOp(5)
    DFIL_PHI = DataFlowOp(6, prefix_formatter('\u03d5(', ',', ')'))
    DFIL_RET = DataFlowOp(7, prefix_formatter('ret '))
    DFIL_CASE = DataFlowOp(8, prefix_formatter('case '))
    DFIL_WHILE = DataFlowOp(9, prefix_formatter('while '))

    DFIL_ASSIGN = DataFlowOp(10, infix_formatter('=', use_paren=False))
    DFIL_INIT = DataFlowOp(11)
    DFIL_DEREF = DataFlowOp(12, prefix_formatter('#'))
    DFIL_IF = DataFlowOp(13, prefix_formatter('if '))
    DFIL_ZX = DataFlowOp(14)
    DFIL_TAILCALL = DataFlowOp(15, call_formatter('ret '))
    DFIL_LOGIC_OR = DataFlowOp(16, infix_formatter('||'))
    DFIL_LOGIC_AND = DataFlowOp(17, infix_formatter('&&'))
    DFIL_SX = DataFlowOp(18)

    DFIL_LSL = DataFlowOp(20, infix_formatter('<<'))
    DFIL_LSR = DataFlowOp(21, infix_formatter('>>'))
    DFIL_XOR = DataFlowOp(22, infix_formatter('^'))

    DFIL_ADD = DataFlowOp(30, infix_formatter('+'))
    DFIL_SUB = DataFlowOp(31, infix_formatter('-'))
    DFIL_MUL = DataFlowOp(32, infix_formatter('*'))

    DFIL_ARRAY_INDEX = DataFlowOp(40, prefix_formatter('', '[', ']'))
    DFIL_STORE = DataFlowOp(41, prefix_formatter('', '='))
    DFIL_STRUCT_FIELD = DataFlowOp(42, prefix_formatter('', '.'))

    DFIL_CMP_E = DataFlowOp(80, infix_formatter('=='))
    DFIL_CMP_NE = DataFlowOp(81, infix_formatter('!='))
    DFIL_CMP_SGT = DataFlowOp(82, infix_formatter('s>'))
    DFIL_CMP_SLE = DataFlowOp(83, infix_formatter('s<'))

    DFIL_FSUB = DataFlowOp(50, infix_formatter('-'))
    DFIL_FADD = DataFlowOp(51, infix_formatter('+'))
    DFIL_FDIV = DataFlowOp(52, infix_formatter('/'))
    DFIL_FMUL = DataFlowOp(53, infix_formatter('*'))

    DFIL_FCONV = DataFlowOp(54, prefix_formatter('fconvert(', '', ')'))
    DFIL_INT_TO_FLOAT = DataFlowOp(55, prefix_formatter('float.d(', '', ')'))
    DFIL_FLOAT_TO_INT = DataFlowOp(55, prefix_formatter('int(', '', ')'))
    DFIL_FTRUNC = DataFlowOp(56, prefix_formatter('ftrunc(', '', ')'))

    DFIL_SPLIT = DataFlowOp(60, prefix_formatter('', ':'))
    DFIL_ADDRESS_OF = DataFlowOp(61, prefix_formatter('&'))


    DFIL_INTRINSIC = DataFlowOp(98, prefix_formatter('intrinsic(', '', ')'))
    DFIL_PACK_LIST = DataFlowOp(99, prefix_formatter('PACK[', ',', ']'))

    def format(self, operands: list[str]) -> str:
        if self.value.formatter:
            return self.value.formatter(operands)
        return f'{self.name}({",".join(operands)})'


globals().update(HlilOp.__members__)
globals().update(DataFlowILOperation.__members__)

hlil_to_dfil_operations = {
    HlilOp.HLIL_CALL_SSA: DataFlowILOperation.DFIL_CALL,
    HlilOp.HLIL_VAR_PHI: DataFlowILOperation.DFIL_PHI,
    HlilOp.HLIL_RET: DataFlowILOperation.DFIL_RET,

    HlilOp.HLIL_ASSIGN: DataFlowILOperation.DFIL_ASSIGN,
    HlilOp.HLIL_VAR_INIT_SSA: DataFlowILOperation.DFIL_ASSIGN,
    HlilOp.HLIL_VAR_INIT: DataFlowILOperation.DFIL_ASSIGN,
    HlilOp.HLIL_VAR: DataFlowILOperation.DFIL_ASSIGN,

    HlilOp.HLIL_DEREF_SSA: DataFlowILOperation.DFIL_DEREF,
    HlilOp.HLIL_IF: DataFlowILOperation.DFIL_IF,
    HlilOp.HLIL_ZX: DataFlowILOperation.DFIL_ZX,
    HlilOp.HLIL_TAILCALL: DataFlowILOperation.DFIL_TAILCALL,
    HlilOp.HLIL_OR: DataFlowILOperation.DFIL_LOGIC_OR,
    HlilOp.HLIL_AND: DataFlowILOperation.DFIL_LOGIC_AND,
    HlilOp.HLIL_SX: DataFlowILOperation.DFIL_SX,

    HlilOp.HLIL_LSL: DataFlowILOperation.DFIL_LSL,
    HlilOp.HLIL_LSR: DataFlowILOperation.DFIL_LSR,
    HlilOp.HLIL_XOR: DataFlowILOperation.DFIL_XOR,

    HlilOp.HLIL_ADD: DataFlowILOperation.DFIL_ADD,
    HlilOp.HLIL_SUB: DataFlowILOperation.DFIL_SUB,
    HlilOp.HLIL_MUL: DataFlowILOperation.DFIL_MUL,
    HlilOp.HLIL_MULU_DP: DataFlowILOperation.DFIL_MUL,

    HlilOp.HLIL_ARRAY_INDEX_SSA: DataFlowILOperation.DFIL_ARRAY_INDEX,
    HlilOp.HLIL_ASSIGN_MEM_SSA: DataFlowILOperation.DFIL_STORE,
    HlilOp.HLIL_STRUCT_FIELD: DataFlowILOperation.DFIL_STRUCT_FIELD,

    HlilOp.HLIL_CMP_E: DataFlowILOperation.DFIL_CMP_E,
    HlilOp.HLIL_CMP_NE: DataFlowILOperation.DFIL_CMP_NE,
    HlilOp.HLIL_CMP_SGT: DataFlowILOperation.DFIL_CMP_SGT,
    HlilOp.HLIL_CMP_SLE: DataFlowILOperation.DFIL_CMP_SLE,

    HlilOp.HLIL_FSUB: DataFlowILOperation.DFIL_FSUB,
    HlilOp.HLIL_FADD: DataFlowILOperation.DFIL_FSUB,
    HlilOp.HLIL_FDIV: DataFlowILOperation.DFIL_FDIV,
    HlilOp.HLIL_FMUL: DataFlowILOperation.DFIL_FMUL,
    HlilOp.HLIL_FLOAT_CONV: DataFlowILOperation.DFIL_FCONV,
    HlilOp.HLIL_INT_TO_FLOAT: DataFlowILOperation.DFIL_INT_TO_FLOAT,
    HlilOp.HLIL_FLOAT_TO_INT: DataFlowILOperation.DFIL_FLOAT_TO_INT,

    HlilOp.HLIL_FTRUNC: DataFlowILOperation.DFIL_FTRUNC,

    HlilOp.HLIL_CASE: DataFlowILOperation.DFIL_CASE,

    # TODO: need to distinguish multiple output flow direction
    HlilOp.HLIL_SPLIT: DataFlowILOperation.DFIL_SPLIT,
    HlilOp.HLIL_ADDRESS_OF: DataFlowILOperation.DFIL_ADDRESS_OF,

    HlilOp.HLIL_WHILE_SSA: DataFlowILOperation.DFIL_WHILE

}


def get_dfil_op(expr):
    match expr:
        case binaryninja.commonil.Constant() | int() | float():
            return DataFlowILOperation.DFIL_DECLARE_CONST
        case binaryninja.SSAVariable():
            return DataFlowILOperation.DFIL_DECLARE_VAR
        case highlevelil.HighLevelILInstruction() as instr if instr.operation in hlil_to_dfil_operations:
            return hlil_to_dfil_operations[instr.operation]
        case list():
            return DataFlowILOperation.DFIL_PACK_LIST
        case _:
            return DataFlowILOperation.DFIL_UNKNOWN


@dataclass(frozen=True)
class DataNode:
    base_instr: highlevelil.HighLevelILInstruction
    il_expr: Union[
        commonil.BaseILInstruction,
        binaryninja.mediumlevelil.SSAVariable,
        binaryninja.variable.Variable,
        int]
    operands: list
    node_id: int
    block_id: int
    operation: DataFlowILOperation

    def __post_init__(self):
        assert(all(self.operands))

    def get_operation_txt(self):
        if self.operation == DataFlowILOperation.DFIL_UNKNOWN:
            return type(self.il_expr).__name__
        return self.operation.name

    def is_control_related(self):
        match self.il_expr:
            case binaryninja.highlevelil.HighLevelILIf():
                return True
            # case commonil.ControlFlow():
            #    return True
        return False

    def get_const_var_txt(self):
        match self.il_expr:
            case float():
                return f'float({self.il_expr})'
            case int():
                return hex(self.il_expr)
            case binaryninja.variable.Variable() as var:
                return f'{var.name}'
            case binaryninja.mediumlevelil.SSAVariable() as ssa:
                version = f'#{ssa.version}' if ssa.version else ''
                return f'{ssa.var.name}{version}'
            case binaryninja.highlevelil.HighLevelILVarSsa(var=ssa):
                version = f'#{ssa.version}' if ssa.version else ''
                return f'{ssa.var.name}{version}'
        return None

    def get_il_index(self, default=None):
        return self.base_instr.instr_index
        #if hasattr(self.il_expr, 'instr_index'):
        #    return self.il_expr.instr_index
        #return default

    def get_il_address(self, default=None):
        return self.base_instr.address
        #if hasattr(self.il_expr, 'address'):
        #    return self.il_expr.address
        #return default


    def get_dfil_txt(self, displayed_nodes=frozenset()):
        if self.node_id in displayed_nodes:
            return f'{self.node_id}'
        optxts = [operand.get_dfil_txt(displayed_nodes) for operand in self.operands]
        match self.operation:
            case DataFlowILOperation.DFIL_DECLARE_CONST:
                return f'{self.get_const_var_txt()}'
            case DataFlowILOperation.DFIL_DECLARE_VAR:
                return f'{self.node_id}:{self.get_const_var_txt()}'
            case _:
                return f'{self.operation.format(optxts)}'

    def format_tokens(self):
        return self.get_dfil_txt()


    def _tokenize_operand(self, operand):
        match operand:
            #case DataNode() as dn:
            #    return f'M{dn.node_id}'
            case _:
                return str(operand)

    def _tokenize_operands(self):
        #return intersperse([self._tokenize_operand(operand) for operand in self.operands], ',')
        return intersperse(self.operands, ',')

    def get_expression(self) -> TokenExpression:
        if self.operation is DataFlowILOperation.DFIL_DECLARE_CONST:
            return TokenExpression(self, [self.operation.name, '(', self.il_expr, ')'])

        return TokenExpression(self, [self.operation.name, '('] + self._tokenize_operands() + [')'])



class EdgeType(enum.IntEnum):
    Move = 1    # copy, used in arithmetic, etc
    Load = 2    # Used directly as an address for a load operation
    Store = 3   # Used directly as an address for a store operation
    Branch = 4  # Used directly as an address for indirect branching or control flow
    Arg = 5     # Use as an argument to a function

    def onechar(self):
        return self.name[0]

    def short(self):
        if self == EdgeType.Move:
            return ''
        return self.onechar()


def get_edge_type_from_dfil_op(dfil_op, op_index):
    if dfil_op == DataFlowILOperation.DFIL_DEREF:
        return EdgeType.Load
    if dfil_op == DataFlowILOperation.DFIL_STORE and op_index == 0:
        return EdgeType.Store
    if dfil_op in [DataFlowILOperation.DFIL_CALL, DataFlowILOperation.DFIL_TAILCALL] and op_index == 0:
        return EdgeType.Branch
    return EdgeType.Move


@dataclass(frozen=True)
class DataFlowEdge:
    in_node: DataNode
    out_node: DataNode
    edge_type: EdgeType
    operand_index: int

    def txt(self):
        return f'{self.in_node.node_id}.{self.edge_type.onechar()}->{self.out_node.node_id}.{self.operand_index}'

    def short_in_txt(self):
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

    def short(self):
        return self.name[0]

@dataclass(frozen=True)
class ControlEdge:
    in_block: 'DataBasicBlock'
    out_block: 'DataBasicBlock'
    edge_type: ControlEdgeType
    data_node: DataNode

    def get_txt(self):
        data_node_id = f'N{self.data_node.node_id}' if self.data_node else ''
        return f'{self.edge_type.name:14} BB{self.in_block.block_id}->BB{self.out_block.block_id} {data_node_id}'


@dataclass()
class DataBasicBlock:
    il_block: binaryninja.basicblock.BasicBlock
    data_nodes: list[DataNode]
    block_id: int

    # Nodes that correspond with the outermost HLIL instruction
    base_data_nodes: list[DataNode]

    edges_in: list[ControlEdge]
    edges_out: list[ControlEdge]
    out_edge_node: Optional[DataNode]

    def get_txt(self):
        return f'Block {self.block_id} {self.il_block.start}-{self.il_block.end} has {len(self.data_nodes)} nodes'
