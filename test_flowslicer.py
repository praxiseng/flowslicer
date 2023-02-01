import itertools
import os
import sys

import binaryninja
import cbor2
import pytest

try:
    from . import flowslicer
except:
    import flowslicer

def get_bin_shortlist():
    sample_bins_dir = os.path.join('test', 'sample_binaries')
    if os.path.exists(sample_bins_dir):
        bins = []
        for root, dirs, files in os.walk(sample_bins_dir):
            for file in files:
                file_path = os.path.join(root, file)
                bins.append(file_path)
                assert (len(bins) < 100)
        if bins:
            return bins

    # At least grab the python executable
    bins = [sys.executable]
    if sys.platform.startswith('win32'):
        bins.append(r'C:\Windows\explorer.exe')
    if sys.platform.startswith('linux'):
        bins.extend([
            '/usr/bin/ls',
            '/usr/bin/pwd',
            '/usr/bin/yes'
        ])
    return bins


@pytest.fixture
def bin_shortlist():
    return get_bin_shortlist()

@pytest.fixture
def slices_dir(tmp_path):
    return os.path.join(tmp_path, 'slices')

baseline_dir = os.path.join('test', 'baseline')

def slices_files_output(input_bins, slices_dir):
    return [flowslicer.get_slice_output(slices_dir, bin_path)
            for bin_path in input_bins]

def baseline_path(slices_path):
    return os.path.join(baseline_dir, os.path.basename(slices_path))

def load_cbor_file(path):
    lines = []
    with open(path, 'rb') as fd:
        try:
            while True:
                lines.append(cbor2.load(fd))
        except cbor2.CBORDecodeEOF:
            pass
    return lines

def load_sort_file(path):
    lines = load_cbor_file(path)
    sorted_file = sorted(lines, key=lambda x:(sorted(x['addressSet']), x['canonicalText']))
    return (line for line in sorted_file)

def diff_slices(a, b):
    line_num = 0
    file_a = load_sort_file(a)
    file_b = load_sort_file(b)

    #assert len(file_a) == len(file_b)
    for line_a, line_b in itertools.zip_longest(file_a, file_b):
        for key, value_a in line_a.items():
            value_b = line_b[key]
            assert value_a == value_b

def slice(bins, slices_out_dir):
    main = flowslicer.Main(
        ['slice', '--slices', slices_out_dir] +
        ['--parallelism', '8'] +
        ['--force-update'] +
        #['--update-bndb'] +
        #['--function', 'main'] +
        ['--bn-int', 'analysis.limits.workerThreadCount=20'] +
        ['--bn-str', 'analysis.mode=full'] +
        ['--bn-bool', 'analysis.experimental.gratuitousFunctionUpdate=False'] +
        bins
    )
    main.run()

    slices_paths = slices_files_output(bins, slices_out_dir)

    for slices_path in slices_paths:
        assert(os.path.exists(slices_path))

    for slices_path in slices_paths:
        baseline_file = baseline_path(slices_path)
        if os.path.exists(baseline_file):
            diff_slices(slices_path, baseline_file)



def test_slice(bin_shortlist, slices_dir):
    slice(bin_shortlist, slices_dir)


if __name__ == "__main__":
    # Run baselining
    slice(get_bin_shortlist(), baseline_dir)
