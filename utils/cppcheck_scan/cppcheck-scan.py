#!/usr/bin/env python3

# SPDX-License-Identifier: Apache-2.0
# Copyright 2023 The IBM Research Authors.
# This script invokes cppcheck to scan cpp files.

import logging
import os
import subprocess
import sys
from pathlib import Path

WORKSPACE_DIR = "/workdir"
ONNX_MLIR_DIR = WORKSPACE_DIR + "/onnx-mlir/"
BUILD_DIR = ONNX_MLIR_DIR + "build/"
UTILS_DIR = ONNX_MLIR_DIR + "utils/"
CPPCHECK_SCAN_DIR = UTILS_DIR + "cppcheck_scan/"
EXCLUDES_FILE = CPPCHECK_SCAN_DIR + "cppcheck_exclude_dirs.txt"
PROJECT_FILE = BUILD_DIR + "compile_commands.json"
LOG_FILE = BUILD_DIR + "cppcheck_log.xml"
RESULTS_FILE = BUILD_DIR + "cppcheck_results.xml"
SUPPRESSIONS_TXT = "cppcheck_suppressions.txt"
SUPPRESSIONS_FILE = CPPCHECK_SCAN_DIR + SUPPRESSIONS_TXT

NPROC = os.getenv("NPROC", "4")


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s][%(lineno)03d] %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Obtain excludes if the excludes file exists
    EXCLUDES = ""
    excludes_file = Path(EXCLUDES_FILE)
    if excludes_file.is_file():
        with open(EXCLUDES_FILE, "r") as fin:
            lines = fin.readlines()
            for line in lines:
                line = line.strip()
                if not line.startswith("#") and len(line) != 0:
                    EXCLUDES = EXCLUDES + "-i" + line + " "

    # Obtain suppressions if the suppressions file exists
    SUPPRESSIONS = ""
    suppressions_file = Path(SUPPRESSIONS_FILE)
    if suppressions_file.is_file():
        SUPPRESSIONS = "--suppressions-list=" + SUPPRESSIONS_FILE + " "

    # Invoke cppcheck
    cppscan_string = (
        "cppcheck "
        + SUPPRESSIONS
        + EXCLUDES
        + " -j"
        + NPROC
        + " --project="
        + PROJECT_FILE
        + " --xml"
        + " 1>"
        + LOG_FILE
        + " 2>"
        + RESULTS_FILE
    )
    logging.info("%s", cppscan_string)
    completed_process = subprocess.run(cppscan_string, shell=True)
    if completed_process.returncode != 0:
        logging.error(
            "%s",
            "Error invoking cppcheck, return code = "
            + str(completed_process.returncode),
        )
        sys.exit(completed_process.returncode)

    # Check results
    results_file = Path(RESULTS_FILE)
    if not results_file.is_file():
        error_string = "The cppcheck results file " + RESULTS_FILE + " does not exist"
        logging.error("%s", error_string)
        sys.exit(error_string)
    with open(RESULTS_FILE, "r") as rfin:
        word_count = 0
        lines = rfin.readlines()
        for line in lines:
            words = line.split()
            for word in words:
                # If 'location' is in the results it indicates an error location
                if "location" in word:
                    word_count += 1
        if not word_count == 0:
            error_string = (
                str(word_count)
                + " new cppcheck errors were found. "
                + "Correct the errors or if false positives add to "
                + SUPPRESSIONS_TXT
                + "."
            )
            logging.error("%s", error_string)
            # Print results for debug
            with open(RESULTS_FILE, "r") as rfin:
                print(rfin.read())
            sys.exit(error_string)
    logging.info("%s", "No new cppcheck errors were found")


if __name__ == "__main__":
    main()
