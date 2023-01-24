from importlib import reload
from flowslicer import flowslicer, dfil

import binaryninja



def flowslicer_closure(*pargs, **kwargs):
    def reload_call_flowslicer(bv, fx):
        reload(flowslicer)
        reload(dfil)
        flowslicer.analyze_function(bv, fx, *pargs, **kwargs)
    return reload_call_flowslicer

def flowslicer_hlil_closure(*pargs, **kwargs):
    def call_flowslicer_hlil_instruction(bv, hlil_instr):
        reload(flowslicer)
        reload(dfil)
        flowslicer.analyze_hlil_instruction(bv, hlil_instr, *pargs, **kwargs)
    return call_flowslicer_hlil_instruction


def setup():
    binaryninja.PluginCommand.register_for_function(
        'Flowslicer on current function',
        'Show function data flows',
        flowslicer_closure()
    )

    binaryninja.PluginCommand.register_for_high_level_il_instruction(
        'Flowslicer on HLIL instruction',
        'Show flows from instruction',
        flowslicer_hlil_closure()
    )

setup()