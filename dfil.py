import binaryninja
import enum

from binaryninja import commonil, highlevelil
from dataclasses import dataclass
from binaryninja.enums import HighLevelILOperation as HlilOp
from typing import Union, Callable, Optional


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

    # TODO: need to distinguish multiple output flow direction
    HlilOp.HLIL_SPLIT: DataFlowILOperation.DFIL_SPLIT,
    HlilOp.HLIL_ADDRESS_OF: DataFlowILOperation.DFIL_ADDRESS_OF,


}


def get_dfil_op(expr):
    match expr:
        case binaryninja.commonil.Constant() | int() | float():
            return DataFlowILOperation.DFIL_DECLARE_CONST
        case binaryninja.SSAVariable():
            return DataFlowILOperation.DFIL_DECLARE_VAR
        case highlevelil.HighLevelILInstruction() as instr if instr.operation in hlil_to_dfil_operations:
            return hlil_to_dfil_operations[instr.operation]
        case _:
            return DataFlowILOperation.DFIL_UNKNOWN


@dataclass(frozen=True)
class DataNode:
    base_instr: highlevelil.HighLevelILInstruction
    il_expr: Union[
        commonil.BaseILInstruction, binaryninja.mediumlevelil.SSAVariable, binaryninja.variable.Variable, int]
    operands: list
    node_id: int
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


class EdgeType(enum.IntEnum):
    Move = 1  # copy, used in arithmetic, etc
    Load = 2  # Used directly as an address for a load operation
    Store = 3  # Used directly as an address for a store operation
    Branch = 4  # Used directly as an address for indirect branching or control flow

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


@dataclass(frozen=True)
class ControlEdge:
    in_block: 'DataBasicBlock'
    out_block: 'DataBasicBlock'
    edge_type: ControlEdgeType
    data_node: DataNode


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
