import json
from importlib import reload
import os

import cbor2

try:
    from flowslicer import flowslicer, dfil, db
except ModuleNotFoundError:
    import flowslicer
    import dfil
    import db

import binaryninja

settings: binaryninja.Settings = None


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

def _adjust_working_dir(path: str) -> str:
    return os.path.expandvars(path)

def _get_path(settingName: str) -> str:
    path = settings.get_string(settingName)
    return _adjust_working_dir(path)



def slice(paths,
          options: list[str]=None,
          slices_dir = None,
          parallelism: int = 1) -> None:
    ''' Slice files (and files in folders) listed by paths.  When invoking from the Binary Ninja
        Python console, parallelism should be 1.
    '''

    reload(flowslicer)
    if not isinstance(paths, list):
        paths = [paths]
    slices_dir = slices_dir or _get_path("flowslicer.slicesDir")

    assert slices_dir

    flowslicer.Main(['slice',
                     '--parallelism', str(parallelism),
                     '--slices', slices_dir
                    ] +
                    paths +
                    (options or [])).run()
    print('Done slicing!')


def ingest(paths_to_slice: list[str] = None,
           options: list[str]=None,
           db_path = None,
           slices_dir = None) -> None:

    reload(flowslicer)

    if paths_to_slice:
        slice(paths_to_slice, slices_dir=slices_dir)

    db_path = db_path or _get_path("flowslicer.sliceDBPath")
    slices_dir = slices_dir or _get_path("flowslicer.slicesDir")
    assert(slices_dir)

    flowslicer.Main(['ingest', '--db', db_path, slices_dir] + (options or [])).run()

    print('Done ingesting!')


def search(bv_or_path: str | binaryninja.BinaryView,
           options: list[str]=None,
           db_path=None,
           slices_dir=None,
           detail_dir=None):

    reload(flowslicer)
    if isinstance(bv_or_path, binaryninja.BinaryView):
        binary_path = bv_or_path.file.filename
    else:
        assert (isinstance(bv_or_path, str))
        binary_path = bv_or_path

    slices_path = ''
    if binary_path.endswith('.slices'):
        slices_path = binary_path
    else:
        slices_dir = slices_dir or _get_path("flowslicer.slicesDir")
        slice([binary_path], slices_dir=slices_dir)
        slices_path = flowslicer.get_slice_output(slices_dir, binary_path)



    assert(os.path.exists(slices_path))

    db_path = db_path or _get_path("flowslicer.sliceDBPath")
    detail_dir = detail_dir or _get_path("flowslicer.detailDir")
    detail_dir = os.path.join(detail_dir, os.path.basename(binary_path))

    main = flowslicer.Main(['search',
                            '--db', db_path,
                            '--detail', detail_dir,
                            slices_path
                           ] +
                           (options or []))

    main.run()

    print('Done searching!')

def get_detail(bv_or_path: str | binaryninja.BinaryView,
               function_name: str,
               detail_dir=None
               ):

    reload(flowslicer)
    if isinstance(bv_or_path, binaryninja.BinaryView):
        binary_path = bv_or_path.file.filename
    else:
        assert (isinstance(bv_or_path, str))
        binary_path = bv_or_path

    detail_dir = detail_dir or _get_path("flowslicer.detailDir")
    detail_dir = os.path.join(detail_dir, os.path.basename(binary_path))

    detail_file = flowslicer.get_detail_file_name(detail_dir)

    results = []
    with open(detail_file, 'rb') as fd:
        try:
            while True:
                results.append(cbor2.load(fd))
        except cbor2.CBORDecodeEOF:
            pass

    print(f'There are {len(results)} results in {detail_file}')

    filtered_results = []
    for result in results:
        if any(fa['funcName'] == function_name for fa in result['thisFile']['funcAddresses']):
            filtered_results.append(result)
    for result in filtered_results:
        slice_hash = result['hash']

        thisFile = result['thisFile']
        otherFiles = result['otherFiles']

        fileNames = otherFiles['fileNames']
        fileCount = otherFiles['fileCount']

        funcAddresses = thisFile['funcAddresses']
        oneLineAddresses = ''
        funcName = ''
        if len(funcAddresses) == 1:
            funcName = funcAddresses[0]['funcName']
            oneLineAddresses = ' '.join(f'{addr:x}' for addr in sorted(funcAddresses[0]['addressSet']))

        fileNameTxt = ' '.join(fileNames)
        print(f'{db.btoh(slice_hash)} {fileCount:4} {fileNameTxt[:40]:40} {funcName:30} {oneLineAddresses}')

        for line in result['canonicalText'].split('\n'):
            print(f'    {line}')


        if len(funcAddresses) > 1:
            for fa in funcAddresses:
                funcName = funcAddresses[0]['funcName']
                addressesTxt = ' '.join(f'{addr:x}' for addr in sorted(fa['addressSet']))
                print(f'    {funcName:30} {addressesTxt}')

        #print(json.dumps(result, indent=4, default=str))



def setup():
    global settings
    settings = binaryninja.Settings()
    settings.register_group("flowslicer", "Flowslicer")

    settings.register_setting("flowslicer.slicesDir", json.dumps(dict(
        title="Slices Folder",
        description="Working directory to store .slices files",
        default=os.path.join("${USERPROFILE}", "flowslicer", "slices"),
        type="string"
    )))

    settings.register_setting("flowslicer.sliceDBPath", json.dumps(dict(
        title="SliceDB File Path",
        description="Path to SliceDB file",
        default=os.path.join("${USERPROFILE}", "flowslicer", "flowslicer.slicedb"),
        type="string"
    )))

    settings.register_setting("flowslicer.detailDir", json.dumps(dict(
        title="Data flow slice detail output folder",
        description="Path to place detailed slice output",
        default=os.path.join("${USERPROFILE}", "flowslicer", "detail"),
        type="string"
    )))


    binaryninja.PluginCommand.register_for_function(
        'Flowslicer - Show DFIL on current function',
        'Show function data flows',
        flowslicer_closure()
    )

    binaryninja.PluginCommand.register_for_high_level_il_instruction(
        'Flowslicer - Show DFIL on HLIL instruction',
        'Show flows from instruction',
        flowslicer_hlil_closure()
    )


setup()