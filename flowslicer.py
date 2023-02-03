#!/usr/bin/env python
import argparse
import json
import os.path
import queue
import sys
import typing

from collections import defaultdict
from threading import Thread
from typing import List, Generator

from multiprocessing import Pool, Value, Process, Queue
import cbor2

import binaryninja.highlevelil

try:
    from . import db
except ImportError:
    import db

try:
    from .dfil import *
    from .canonicalizer import *
    from .expressionslice import *
    from .settings import *
    from .dfil_function import *
except ImportError:
    from dfil import *
    from canonicalizer import *
    from expressionslice import *
    from settings import *
    from dfil_function import *



def slice_function(args,
                   bv: binaryninja.BinaryView,
                   fx: binaryninja.Function,
                   output: Generator[None, dict, None],
                   hlil: binaryninja.highlevelil.HighLevelILFunction = None):
    if not fx:
        print(f'Passed NULL function to process!')
        return
    if not fx.hlil:
        print(f'HLIL is null for function {fx}')
        return

    parser = ILParser()
    dfx: DataFlowILFunction = parser.parse(list((hlil or fx.hlil).ssa_form))

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
                print(f'Limit exceeded in {fx} during fold_const')

            try:
                xslice.fold_single_use()
            except LimitExceededException:
                print(f'Limit exceeded in {fx} during fold_single_use')

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

    slice_function(args, bv, fx, display)

def analyze_function(bv: binaryninja.BinaryView,
                     fx: binaryninja.Function):
    parser = ILParser()
    dfil_fx = parser.parse(list(fx.hlil.ssa_form))

    for block in parser.data_blocks:
        print(block.get_txt())
        for oe in block.edges_out:
            print(f'   {oe.edge_type.name:16} {oe.out_block.block_id} {oe.data_node.node_id if oe.data_node else ""}')

    test_node = parser.data_blocks[0].data_nodes[0]

    # dfil_fx.graph_flows_from(test_node)

    dfil_fx.graph()

def display_json():
    try:
        while True:
            data = yield
            print(json.dumps(data, indent=4, default=str))
    finally:
        print("Iteration stopped")


def slice_functions_by_name(args,
                            bv: binaryninja.BinaryView,
                            names: list[str],
                            output: Generator[None, dict, None]):
    for fxname in names:
        funcs = bv.get_functions_by_name(fxname)
        funcs = [func for func in funcs if not func.is_thunk]
        assert (len(funcs) == 1)
        fx = funcs[0]
        slice_function(args, bv, fx, output)


def slice_binary_bv(args,
                    bv: binaryninja.BinaryView,
                    output: Generator[None, dict, None]):
    if verbosity >= 2:
        print(f'bv has {len(bv.functions)} functions')
    if args.function:
        slice_functions_by_name(args, bv, args.function, output)
    else:
        #for fx_il in bv.hlil_functions(1024):
        #    fx = fx_il.source_function
        for fx in bv.functions:
            fx_il = fx.hlil
            if verbosity >= 2:
                print(f'Analyzing {fx}')
            slice_function(args, bv, fx, output, hlil=fx_il)


def _slice_binary(args,
                  binary_path: str,
                  output: Generator[None, dict, None]):

    with binaryninja.open_view(binary_path) as bv:
        bv.update_analysis_and_wait()
        slice_binary_bv(args, bv, output)


def dump_slices(out_fd):
    try:
        while True:
            data = yield
            out_fd.write(cbor2.dumps(data))
    finally:
        pass
        # print("Iteration stopped")


def get_slice_output(slices_dir, input_file_path, extension=SLICE_EXTENSION):
    assert slices_dir
    return os.path.join(slices_dir, os.path.basename(input_file_path) + extension)


def get_bndb_output(bndb_dir, input_file_path):
    assert bndb_dir
    return os.path.join(bndb_dir, os.path.basename(input_file_path) + '.bndb')


def get_detail_file_name(detail_path):
    return detail_path.rstrip('/\\') + '.detail'


def str_to_bool(v):
    if isinstance(v, bool):
        return
    if v.lower() in ['true', 'yes', 't', 'y', '1']:
        return True
    if v.lower() in ['false', 'no', 'f', 'n', '0']:
        return False
    raise argparse.ArgumentTypeError("Expected boolean value")


class Main:
    def __init__(self, args=None):
        self.detail_file = None

        import argparse

        parser = argparse.ArgumentParser(description='Data flow slicing and match set analysis tool.')

        general = parser.add_argument_group('General options')
        parser.add_argument('command',
                            choices={"slice", "ingest", "search"},
                            metavar='COMMAND',
                            help='Available commands are slice, ingest, and search')
        parser.add_argument('files',
                            nargs='+',
                            metavar='FILES',
                            help="List of binaries to slice, slice files to ingest into a slicedb, or slice file(s) "
                                 "to use for a search against a slicedb. Folders will be recursively enumerated.")

        general.add_argument('-d', '--db', default='slices.slicedb', metavar='SLICEDB', nargs='?',
                            help='Database file for ingest and search.  Defaults to "slices.slicedb".')

        slicing = parser.add_argument_group('Slice command options')
        slicing.add_argument('-s', '--slices',
                             default='data_flow_slices', metavar='SLICES_DIR', nargs='?',
                             help='Folder to place slice files and logs.  Defaults to "data_flow_slices".')
        slicing.add_argument('-b', '--bndb',
                             default='bndb_files', metavar='SLICES_DIR', nargs='?',
                             help='Folder to place .bndb files to cache analysis.  Defaults to "bndb_files".')

        slicing.add_argument('-x', '--function',
                             metavar='NAME', action='append',
                             help='Limit slicing to a function or functions.')
        slicing.add_argument('-f', '--force-update', action='store_true', help=
                             f'Replace output {SLICE_EXTENSION} files if they already exist.'
                             )
        slicing.add_argument('-u', '--update-bndb', action='store_true', help=
                             f'Replace output .bndb files if they already exist.'
                             )

        slicing.add_argument('-p', '--parallelism', metavar='N', type=int, default=1, help=
                             'Run N instances in parallel.  While Binary Ninja has some multithreading of ' +
                             'its own, flowslicer.py has some serial portions that can be sped up by parallelism. ' +
                             'While this will vary per system, we suggest around 50%% and 75%% of the number of ' +
                             'physical cores.')
        slicing.add_argument('--bn-int',
                             metavar='SETTING',
                             action='append',
                             default=[],
                             help='Set a BinaryNinja integer setting in the global scope.'
                             )
        slicing.add_argument('--bn-str',
                             metavar='SETTING',
                             action='append',
                             default=[],
                             help='Set a BinaryNinja string setting in the global scope.'
                             )
        slicing.add_argument('--bn-bool',
                             metavar='SETTING',
                             action='append',
                             default=[],
                             help='Set a BinaryNinja boolean setting in the global scope.'
                             )
        slicing.add_argument('--bn-reset',
                             action='store_true',
                             help='Reset all Binary Ninja settings in the global scope.')

        search = parser.add_argument_group('Search command options')
        search.add_argument('--detail',
                            metavar='OUT_DIR',
                            default='match_set_detail',
                            help='Specify a folder to output detailed information on each slice in each match set.'
                            )

        parser.add_argument('-v', '--verbose', action='count', default=0)
        global verbosity
        global files_processed
        global total_files

        self.args = parser.parse_args(args)
        self.args.search = self.args.command == "search"

        self.handle_binja_settings()

        verbosity = self.args.verbose
        files_processed = Value('i', 0)
        total_files = Value('i', 0)

        # This configures slicing.  Separate slices will be generated with each option set listed here.  They
        # are then deduplicated.
        #   removeInt - Replace all integer constants greater than or equal to the value.
        self.args.option_permutations = [
            dict(),
            dict(removeInt=0x1000),
        ]

        self.dbmain = None

    def handle_binja_settings(self):
        bn_settings = binaryninja.settings.Settings()

        if self.args.bn_reset:
            bn_settings.reset_all()

        for setting in self.args.bn_int:
            key, value = setting.split('=')
            bn_settings.set_integer(key, int(value))

        for setting in self.args.bn_str:
            key, value = setting.split('=', 1)
            bn_settings.set_string(key, value)

        for setting in self.args.bn_bool:
            key, value = setting.split('=', 1)
            bn_settings.set_bool(key, str_to_bool(value))

    def gen_paths(self, paths):
        for path in self.args.files:
            if not os.path.isdir(path):
                yield path
                continue

            for root, dirs, files in os.walk(path):
                for file in files:
                    file_path = os.path.join(root, file)
                    yield file_path

    def run(self):
        if self.args.command == "slice":
            if self.args.slices:
                os.makedirs(self.args.slices, exist_ok=True)
            bin_paths = self.gen_paths(self.args.files)
            self.slice_binaries(bin_paths)

        if self.args.command in ["ingest", "search"]:
            self.dbmain = db.DBMain(self.args)

    def set_vars(self, processed, total):
        global files_processed
        global total_files
        files_processed = processed
        total_files = total

    def get_slice_output_path(self, input_path, extension):
        return get_slice_output(self.args.slices, input_path, extension)

    def get_bndb_output_path(self, input_path):
        return get_bndb_output(self.args.bndb, input_path)

    def make_bndb(self, path):
        if path.endswith('.bndb'):
            return path

        os.makedirs(self.args.bndb, exist_ok=True)
        output_bndb = self.get_bndb_output_path(path)
        if os.path.exists(output_bndb):
            if self.args.update_bndb:
                os.remove(output_bndb)
            else:
                return output_bndb
        with binaryninja.open_view(path) as bv:
            bv.update_analysis_and_wait()
            bv.create_database(output_bndb)
        return output_bndb

    def _slice_binary(self, bv_or_path: str | binaryninja.BinaryView):
        if isinstance(bv_or_path, binaryninja.BinaryView):
            binary_path = bv_or_path.file.filename
        else:
            assert(isinstance(bv_or_path, str))
            binary_path = bv_or_path

        temp_path = self.get_slice_output_path(binary_path, '.temp')
        final_path = self.get_slice_output_path(binary_path, SLICE_EXTENSION)

        if verbosity >= 2:
            print(f'File {files_processed.value + 1} of {total_files.value}: {binary_path}')

        files_processed.value += 1

        if verbosity >= 2:
            print(f'Opening {binary_path}')
        if verbosity >= 2:
            print(f'Output: {final_path}')

        if os.path.exists(final_path):
            if self.args.force_update:
                os.remove(final_path)
            else:
                return

        with open(temp_path, 'wb') as fd:
            write_file = dump_slices(fd)
            write_file.send(None)

            if isinstance(bv_or_path, binaryninja.BinaryView):
                slice_binary_bv(self.args, bv=bv_or_path, output=write_file)
            else:
                binary_path = self.make_bndb(binary_path)
                _slice_binary(self.args, binary_path, write_file)

        os.replace(temp_path, final_path)

    def slice_binary(self, bv_or_path: str | binaryninja.BinaryView):
        try:
            self._slice_binary(bv_or_path)
        except KeyboardInterrupt:
            sys.exit(0)

    def slice_binaries_serially(self, file_paths):
        print("Slicing binaries one-at-a-time.  This can take a long time.")
        file_paths = list(file_paths)
        files_processed.value = 0
        total_files.value = len(file_paths)

        for idx, path in enumerate(file_paths):
            #print(f'Slicing {idx+1} of {len(file_paths)}: {path}')
            self.slice_binary(path)

    def slice_subprocess_queue_handler(self, paths: Queue, results: Queue, files_processed, total_files):
        original_stdout = sys.stdout
        original_stderr = sys.stderr
        self.set_vars(files_processed, total_files)
        binaryninja.disable_default_log()
        while True:
            if self.exiting:
                break

            try:
                path = paths.get()
                if path is None:
                    break
                log_path = self.get_slice_output_path(path, '.log')
                with open(log_path, 'w') as log_fd:
                    binaryninja.log.log_to_file(binaryninja.log.LogLevel.WarningLog, log_path, False)
                    results.put(f'Slicing {files_processed.value + 1} of {total_files.value}: {path}')
                    # results.put(dict(processing=path))
                    self.slice_binary(path)
            except Exception as e:
                print('Exception in thread')
                results.put(str(e))

    def poll_result_queue(self):
        while True:
            try:
                result = self.resultQueue.get(timeout=1)
                print(result)
            except queue.Empty:
                if self.exiting:
                    break
                pass

    def slice_binaries_in_parallel(self, file_paths):
        files_processed.value = 0
        total_files.value = 0 # len(file_paths)

        pathQueue = Queue()
        self.exiting = False
        self.resultQueue = Queue()
        procs = []

        if binaryninja.core_ui_enabled():
            # Can't create subprocesses from Binja UI, so use threads instead.  This may bottleneck somewhat on
            # the python Global Interpreter Lock (GIL), but it is faster than serial processing.
            mpType = Thread
        else:
            # Less GIL bottlenecking with subprocesses
            mpType = Process

        try:
            outThread = Thread(target=self.poll_result_queue)
            outThread.daemon = True
            outThread.start()

            for _ in range(self.args.parallelism):
                proc = mpType(target=self.slice_subprocess_queue_handler,
                              args=(pathQueue, self.resultQueue, files_processed, total_files))
                procs.append(proc)
                proc.start()

            for path in file_paths:
                while True:
                    try:
                        pathQueue.put(path, timeout=0.5)
                        break
                    except queue.Full:
                        print('Timeout!')
                        pass

            for _ in range(self.args.parallelism):
                total_files.value += 1
                pathQueue.put(None)

            for proc in procs:
                while True:
                    proc.join(timeout=1)
                    if mpType == Thread:
                        if not proc.is_alive():
                            break
                    else:
                        if proc.exitcode is not None:
                            break
        except KeyboardInterrupt:
            self.exiting = True
            if mpType == Process:
                for proc in procs:
                    print('Killing')
                    proc.kill()
                print('Done killing')
            os._exit(1)
            sys.exit()
        self.exiting = True

    def slice_binaries(self, bin_paths):
        if self.args.parallelism <= 1:
            self.slice_binaries_serially(bin_paths)
        else:
            self.slice_binaries_in_parallel(bin_paths)


if __name__ == "__main__":
    Main().run()
