
Flowslicer
==========

Flowslicer extracts data flow slices from binaries.  It can then ingest them into a database, then compare them with match set analysis.



Command Line
============


Prerequisites
-------------

A Binary ninja with "GUI-less processing" is needed to run the command line.  There are plans to make it avaliable purely through the plug-in interface to allow non-commercial users to use this tool, but that is not yet implemented.


Flowslicer.py
-------------

usage: flowslicer.py [-h] [--function NAME [NAME ...]] [--db [PATH]] [--force-update] [--parallelism N] binary

positional arguments:
  binary

options:
  -h, --help            show this help message and exit
  --function NAME [NAME ...]
  --db [PATH]
  --force-update
  --parallelism N


Flowslicer will produce a .cbor file for each parsed binary.  These .cbor files will be placed in the folder specified by --db.  

These .cbor files can then be processed and combined into a single database file using the db.py script.


db.py
-----


usage: db.py [-h] [--db [PATH]] [-s] [files ...]

positional arguments:
  files

options:
  -h, --help    show this help message and exit
  --db [PATH]
  -s, --search


The default mode for db.py processes .cbor files and combines them into an output file specified by --db.  Then db.py can be used in search mode with -s. 

I know it's confusing to have --db in flowslicer.py refer to a folder of .cbor files whereas db.py's --db argument refers to a combined database file (with whatever extension you want).  This will be fixed eventually.

In search mode, specify an input database with --db, and a query binary as a positional argument.  This will slice data flows, query the database, and perform match set analysis on the results.


Example Usage
-------------

These examples are done with PowerShell.  Hence \ is not an escape character.


> py .\flowslicer.py sample_binaries\win_sys\ --db win_sys.db --parallelism 20

This command will create a folder called `win_sys.db`