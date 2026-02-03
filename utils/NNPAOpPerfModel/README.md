<!--- SPDX-License-Identifier: Apache-2.0 -->

# Directory to build measurements

Files in this directory enables a user to generate a performance model of most NNPA instructions. Below are the steps to follow.

* measurements on a z16/z17 machine: move all the `.mlir` and `.sh` files to a LoZ machine with an `onnx-mlir` repo (which includes the `onnx-mlir/utils` subdir).
* Execute the `commands.sh` after updating in it the `arch` flag.
* File with execute a list of `driver-*.sh` scripts that gather data for each op. They can also be individually called if only interested in a subset.
* Results are in the `res` subdir where all the `*.csv` file are located, with one per "experiment". Detail logs are found in `log` subdir.
* Move the `res` subdir to a machine with a working browser, where you fire the jupyter lab notebook .
* Open the `scanOpvUnitNewMeas.ipynb` jupyter lab notebook, and edit in the top box the `dir` variable with an absolute path to your local `res` subdirectory. 
* In the same file, also update the `zarch` variable with the correct flag (`z16`, `z17`...).
* Run all the cells... or a subset if you only ran some specific operations.

Note that for stick and unstick (as well as any other operations), you may want to import the relevant `res` csv file to a spreadsheet for further analysis.