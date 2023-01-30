import os
import sys
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

def bin_to_slice_path(bin_path, slices_dir):
    return os.path.join(slices_dir, f'{os.path.basename(bin_path)}.slices')

def slices_files_output(input_bins, slices_dir):
    return [bin_to_slice_path(bin_path, slices_dir) for bin_path in input_bins]

def baseline_path(slices_path):
    return os.path.join(baseline_dir, os.path.basename(slices_path))

def slice(bins, slices_out_dir):
    main = flowslicer.Main(
        ['slice', '--slices', slices_out_dir] +
        ['--parallelism', '8'] +
        ['--bn-int', 'analysis.limits.workerThreadCount=20'] +
        bins
    )
    main.run()

    slices_paths = slices_files_output(bins, slices_out_dir)

    for slices_path in slices_paths:
        assert(os.path.exists(slices_path))

def test_slice(bin_shortlist, slices_dir):
    slice(bin_shortlist, slices_dir)


if __name__ == "__main__":
    slice(get_bin_shortlist(), baseline_dir)
