import enum
import itertools
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
    DFIL_ASSIGN = 10
    DFIL_INIT = 11
    DFIL_DEREF = 12

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
    # the output list is used for multiple outputs, when this node is insufficient for representing the value
    outputs: list
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
                return f'Unknown expr {self.il_expr}'

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
        print(f'    {self.node_id:2} {self.base_instr.instr_index:2} {optxt:20}  {self.format_tokens()}')





class ILParser:
    def __init__(self):
        self.next_node_id = 1
        self.data_blocks = []
        self.current_data_bb = None
        self.current_base_instr = None

        self.varnodes = {}




    def _node(self, expr, operands : [], outputs : []):
        node_id = self.next_node_id
        self.next_node_id += 1
        dn = DataNode(self.current_base_instr, expr, operands, outputs, node_id, get_dfil_op(expr))
        self.current_data_bb.data_nodes.append(dn)
        return dn

    def _var(self, instr, var):
        if var in self.varnodes:
            return self.varnodes[var]

        print(f'var {var}')
        dn = self._node(var, [], [])
        self.varnodes[var] = dn

        return dn

    def _recurse(self, expr):
        match expr:
            case list():
                operands = [self._recurse(operand) for operand in expr]
                return self._node(expr, operands, [])
            case commonil.Constant():
                #return self._node(instr.constant, [])
                return self._var(expr, expr.constant)
            case int():
                return self._var(expr, expr)
            case binaryninja.variable.Variable() as var:
                return self._var(expr, var)
            case highlevelil.HighLevelILVarSsa(var=ssa):
                return self._var(expr, ssa)
            case highlevelil.HighLevelILVarPhi() as var_phi:
                operands = [self._recurse(operand) for operand in var_phi.src]
                return self._node(expr, operands, [])
            case highlevelil.HighLevelILCallSsa() as call_ssa:
                dest, params, dest_mem, src_mem = expr.operands
                operands = [self._recurse(operand) for operand in [dest] + params]
                # TODO: unpack multi-value outputs
                return self._node(expr, operands, [])
            case highlevelil.HighLevelILMemPhi() | highlevelil.HighLevelILVarDeclare():
                pass
            case SSAVariable() as ssa:
                return self._var(expr, ssa)
            case commonil.BaseILInstruction() as instr:
                operands = [self._recurse(operand) for operand in expr.operands]
                return self._node(expr, operands, [])
            #case highlevelil.HighLevelILCallSsa() as call_ssa:

            case _:
                print(f'Type not handled: {type(expr)} for {expr}')
                return self._node(expr, [])

    def parse_bb(self, il_bb):
        dbb = DataBasicBlock(il_bb, [])
        self.current_data_bb = dbb
        self.data_blocks.append(dbb)
        for il_instr in il_bb:
            self.current_base_instr = il_instr
            dn = self._recurse(il_instr)
            #dbb.data_nodes.append(dn)



def analyze_function(bv: binaryninja.BinaryView,
                     fx: binaryninja.Function,
                     *pargs, **kwargs):
    print('Hello, world!')
    print(f'bv={bv}')
    print(f'fx={fx}')

    data_blocks = []
    parser = ILParser()
    for il_bb in fx.hlil.ssa_form:
        parser.parse_bb(il_bb)

    for block in parser.data_blocks:
        print(f'Block with {len(block.data_nodes)} nodes: {block.il_block}')
        for dn in block.data_nodes:
            dn.display()