
# Flowslicer

Flowslicer extracts data flow slices from binaries.  It can then ingest them into a database, then compare them with
match set analysis.

## Prerequisites

A Binary Ninja license with "GUI-less processing" is needed to run the command line.  There are plans to make it
available purely through the plug-in interface to allow non-commercial users to use this tool, but that is not yet
implemented.


## Flowslicer.py


```
usage: flowslicer.py [-h] [--function NAME [NAME ...]] [--db [PATH]] [--force-update] [--parallelism N] binary

positional arguments:
  binary

options:
  -h, --help            show this help message and exit
  --function NAME [NAME ...]
  --output [PATH]
  --force-update
  --parallelism N
```

Flowslicer will produce a .cbor file for each parsed binary.  These .cbor files will be placed in the folder specified
by --db.

These .cbor files can then be processed and combined into a single database file using the db.py script.


## db.py


```
usage: db.py [-h] [--db [PATH]] [-s] [files ...]

positional arguments:
  files

options:
  -h, --help    show this help message and exit
  --db [PATH]
  -s, --search
```


The default mode for db.py processes .cbor files and combines them into an output file specified by --db.  Then db.py
can be used in search mode with -s.

In search mode, specify an input database with --db, and a query binary as a positional argument.  This will slice
data flows, query the database, and perform match set analysis on the results.


## Example Usage

Note: These examples are done with PowerShell.  Hence, \ is not an escape character.

> py .\flowslicer.py sample_binaries\linux_bin\ --db linux_bin_cbor --parallelism 14

This command will process all executables in the sample_binaries\linux_bin folder,  create a folder called
`linux_bin_cbor`, and generate one .cbor for each binary in that folder.  This process can take a long time.
--parallelism 10 will spin up 10 instances to load the cores.  An Intel 12700H processor (14C 20T) will be able to
process 900 linux /usr/bin files in about 90 minutes.

Note that each instance will run a separate Binary Ninja instance, which are each multi-threaded.  However, some
of the processing will be single-threaded in the python script.

> py .\db.py --db linux_bin.db linux_bin_cbor

This command will process the win_sys_db database of CBOR files and summarize into a win_sys.db file.  It will hash
slice text, then store compact summaries that include the following details:

 * Slice hash
 * Count of files, functions, and instances of the slice
 * The list of files with that slice

 > py .\db.py --db linux_bin.db --search linux_bin_cbor\ls.cbor

 This command queries the processed database linux_bin.db with the linux_bin_cbor\ls.cbor CBOR file.  To create a cbor
 file on a new executable, run the flowslicer.py script on the new file.  db.py in search mode will summarize
 the match set output.