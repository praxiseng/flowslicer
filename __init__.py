from importlib import reload
from flowslicer import flowslicer

import binaryninja


def flowslicer_closure(*pargs, **kwargs):
    def reload_call_flowslicer(bv, fx):
        reload(flowslicer)
        flowslicer.analyze_function(bv, fx, *pargs, **kwargs)
    return reload_call_flowslicer


def setup():
    binaryninja.PluginCommand.register_for_function(
        'Flowslicer on current function',
        'Show function data flows',
        flowslicer_closure()
    )

setup()