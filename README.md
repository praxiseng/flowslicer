
# Flowslicer

Flowslicer extracts data flow slices from binaries.  It can then ingest them into a database, then compare them with
match set analysis.

See the related project [REveal](https://github.com/praxiseng/reveal)

## Prerequisites

Flowslicer requires Python 3.10 or newer.

A Binary Ninja license with "GUI-less processing" is needed to run the command line.  There are plans to make it
available purely through the plug-in interface to allow Binary Ninja users with personal licences to use this tool,
but that is not yet implemented.

To install the Binary Ninja API, run `scripts/install_api.py` with your specific python interpreter.

Also install the `cbor2` library using pip:

```commandline
python -m pip install cbor2
```


## Command line reference

```commandline
usage: flowslicer.py [-h] [--db [PATH]] [--slices [PATH]] [--function NAME [NAME ...]] [--force-update] [--parallelism N] [-v] {search,ingest,slice} [files ...]

Data flow slicing and match set analysis tool.

positional arguments:
  {search,ingest,slice}
                        Command to run
  files                 List of binaries to slice, slice files to ingest, or slice files to search. Folders will be recursively enumerated.

options:
  -h, --help            show this help message and exit
  --db [PATH]           Database file for ingest and search. Defaults to "slices.slicedb".
  -v, --verbose

Slice command options:
  --slices [PATH]       Folder to place slice files and logs. Defaults to "data_flow_slices".
  --function NAME [NAME ...]
  --force-update
  --parallelism N
```

## The `slice` command

The slice command will take a list of binaries (or folders to enumerate binaries) and produce a `.slices` file for each 
binary.  These `.slices` files will be placed in a folder specified by `--slices` (defaults to `data_flow_slices`).

These `.slices` files can then be processed and combined into a single database file using `ingest` command or used
for search files using the `search` command.

Example usage:

```commandline
./flowslicer.py slice sample_binaries/linux_bin/ --slices slices/linux_bin --parallelism 10
``` 

This command will process all executables in the sample_binaries\linux_bin folder, create a folder called
`slices/linux_bin/`, and generate one `.slices` file for each binary in that folder.  900 linux binaries takes up to 90 
minutes.

### Notes on Parallelism

The data flow slicing process can take a long time, depending on the binaries. For example, on an Intel 12700H 
processor (14C 20T) `--parallelism 10` will spin up 10 instances to load the cores to nearly 100%, and can process 
900 linux files from `/usr/bin` in about 90 minutes.

Note that while each parallel instance will create a separate Binary Ninja (headless) instance, and each Binary Ninja
instance is multi-threaded.  However, some of the processing will be single-threaded in the python script, so a 
parallelism limit somewhere in the 50-75% of the physical cores is recommended to saturate processing resources.

The memory usage per instance can vary greatly per binary.  With basic Linux and Windows system binaries and Binary
Ninja 3.3, each binary probably uses an average of 1GB, with few binaries exceeding 2GB of memory.  So a 
parallelism-to-memory ratio of 1:2GB should be sufficient for processing the majority of binaries.

However, some binaries can consume much more memory.  For example, according to the Binary Ninja blog Linux Chrome 
with full symbols can consume ~34GB of memory.  Since Flowslicer will output to a `.temp` file and rename to `.slices`
only after processing completes, binaries that failed with out-of-memory can be retried with more memory (e.g. lower
parallelism to increase memory ratio).

## The `ingest` command

The `ingest` command takes a list of files with the `.slices` extension and combines them into a single `.slicedb` file. 

>  ./flowslicer.py ingest --db linux_bin.slicedb slices/linux_bin

Example output:
```commandline
...
   691 slices,    428 unique slices\linux_bin\zdump.slices
  6929 slices,   2842 unique slices\linux_bin\zenity.slices
  9246 slices,   4309 unique slices\linux_bin\zipinfo.slices
  2095 slices,   1062 unique slices\linux_bin\[.slices
Merged counts 3309730
Wrote 2003995 items from 900 files to .\linux_bin.slicedb
```

## The `search` command

The `search` command searches for data flow slices
```commandline
./flowslicer.py search --db linux_bin.slicedb slices/linux_bin/git.slices
```

This produces match set output:

```commandline
...
    49   3 emacs-gtk git-shell git
    49   2 git snap
    55   5 cmake cpack ctest git-shell git
    87   4 git-shell git qemu-system-i386 qemu-system-x86_64
   130   4 git-shell git vim.basic vim.tiny
   133   2 gdb git
   181   3 gdb git-shell git
   229   3 bat git-shell git
 18357   2 git-shell git
 38031   1 git
```

The first column is the number of slices in the match set.  The second is the number of files.  The last is the list
of files by name.  Using this output, we can for example conclude there are `18357` slices that match `git-shell` and 
`git`, but no other files in the input database.

## Next Features

* List slice information for each match set.
* Support not-of-interest databases that subtract out match sets that include known/common libraries.
* Run commands (slice, ingest, search) from within Binary Ninja.

