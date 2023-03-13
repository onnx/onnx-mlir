#!/usr/bin/env python3

# Copyright 2023 The IBM Research Authors.
# This script invokes cppcheck to scan cpp files.

import logging
import os
import subprocess
from pathlib import Path

WORKSPACE_DIR = os.getenv('WORKSPACE')
if WORKSPACE_DIR is None:
    WORKSPACE_DIR = '/workdir'
ONNX_MLIR_DIR = WORKSPACE_DIR + '/onnx-mlir/'
BUILD_DIR = ONNX_MLIR_DIR + 'build/'
UTILS_DIR = ONNX_MLIR_DIR + 'utils/'
CPPCHECK_SCAN_DIR = UTILS_DIR + 'cppcheck_scan/'
EXCLUDES_FILE = CPPCHECK_SCAN_DIR + 'cppcheck_exclude_dirs.txt'
PROJECT_FILE = BUILD_DIR + 'compile_commands.json'
RESULTS_FILE = BUILD_DIR + 'cppcheck_results.xml'
SUPPRESSIONS_FILE = CPPCHECK_SCAN_DIR + 'cppcheck_suppressions.txt'

def main():
    logging.basicConfig(
        level = logging.INFO,
        format = '[%(asctime)s][%(lineno)03d] %(levelname)s: %(message)s',
        datefmt = '%Y-%m-%d %H:%M:%S')
    logging.info('WORKSPACE_DIR: %s', WORKSPACE_DIR)

    # Obtain excludes the excludes file exists
    EXCLUDES=""
    excludes_file = Path(EXCLUDES_FILE)
    if excludes_file.is_file():
        with open(EXCLUDES_FILE,'r') as fin:
           lines = fin.readlines()
           for line in lines:
              line = line.strip()
              if not line.startswith("#") and len(line) != 0:
                  EXCLUDES = EXCLUDES + '-i' + line + ' '

    # Obtain suppressions if the suppressions file exists
    SUPPRESSIONS=""
    suppressions_file = Path(SUPPRESSIONS_FILE)
    if suppressions_file.is_file():
        SUPPRESSIONS = '--suppressions-list=' + SUPPRESSIONS_FILE + ' '

    # Invoke cppcheck
    cppscan_string = 'cppcheck ' \
            + SUPPRESSIONS \
            + EXCLUDES \
            + ' --project=' \
            + PROJECT_FILE \
            + ' --xml' \
            + ' 2>' + RESULTS_FILE
    logging.info('%s', cppscan_string)
    subprocess.run(cppscan_string, shell=True)

if __name__ == "__main__":
    main()
