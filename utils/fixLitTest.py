#!/usr/bin/python3

# SPDX-License-Identifier: Apache-2.0

##################### fixLitTest.py ########################################
#
# Copyright 2022 The IBM Research Authors.
#
################################################################################
#
# This file convert fixes an existing lit test
#
################################################################################

import sys
import os
import getopt
import fileinput
import re
import subprocess

################################################################################
# Usage.


def dprint(msg):
    sys.stderr.write(msg + "\n")


def print_usage(error_msg="", options=False, usage=False, file_format=False):
    if error_msg:
        dprint("ERROR: " + error_msg)
        dprint("")
    if options:
        dprint("")
        dprint(
            'Fixes and tests lit-test file. Repairs are done by "utils/mlir2FileCHeck"'
        )
        dprint("utility.")
        dprint("")
        dprint("fixLitTest [-dhprt] [-f <func-name> <lit-test-filename>")
        dprint("  -t/--test   : Run FileCheck on each function individually.")
        dprint('                When combined with "--repair", test repaired lit test.')
        dprint("                Default flag is none is provided.")
        dprint("  -r/--repair : Repair lit test for each function individually.")
        dprint("  -f,--func <func-name>: Perform test/repair only on given function.")
        dprint("  -p/--print  : Print original lit-test files for the individual.")
        dprint(
            "                functions that were not repaired. Useful only when used"
        )
        dprint('                in combination with "-r -f <func-name>".')
        dprint("  -d/--debug  : Print debug info. Default on with -f,")
        dprint("  -h/--help   : Print help.")
        dprint("")
    if file_format:
        dprint("File format for input list-test files:")
        dprint(' * A single "// RUN:" comment')
        dprint(' * A "// -----" comment')
        dprint(' * Subsequent functions separated by a "// -----" comment')
        dprint("")
    if usage:
        dprint("Workflow for debugging test.mlir:")
        dprint(' * Test original file: "fixLitTest [-t] test.mlir".')
        dprint(
            "   - Test (-t) each function separately, easier to read errors in a long test file."
        )
        dprint(
            ' * Check a failing function X ("-f X"): "fixLitTest [-td] -f X test.mlir".'
        )
        dprint('   - You may inspect the "flt_*.mlir" temp files in the current dir.')
        dprint(
            '   - The "-d" flag list each employed commands, so you may run the command natively.'
        )
        dprint(' * Spurious errors repair in func X:  "fixLitTest -trf X test.mlir".')
        dprint(
            '   - Create a new FileCheck version for function X ("-rf X"), and test it right away ("-t").'
        )
        dprint(
            ' * If good, save fix for X:  "fixLitTest -prf X test.mlir > test.mlir".'
        )
        dprint('   - Create a new FileCheck version for function X ("-rf X"),')
        dprint('     and print the other tests unmodified ("p").')
        dprint(
            ' * When all failures are similar, the "-r" alone will fix all models that failed, '
        )
        dprint("   and print the models that succeeded. Please use sparingly.")
        dprint("")
    sys.exit()


################################################################################
# Globals.

run_command = ""
fix_fct_name = ""
debug = 0
debug_command_str = ""
test_error_functions = []

# File names.
flt_orig_model_file_name = "flt_orig_model.mlir"
flt_compiled_file_name = "flt_compiled.mlir"
flt_new_model_file_name = "flt_new_model.mlir"

# Segments are all of the text between "// -----"
# Segment database for text, function name, mlir2FileCheck command.
segment_text = []
segment_fct_name = []
segment_mlir2FileCheck_command = []


################################################################################
# Run commands.


def run_onnx_mlir_opt(code_file_name, omo_command, output_file_name):
    global debug, debug_command_str

    # Gen command from string.
    command = omo_command.split()
    command.append(code_file_name)
    if debug:
        debug_command_str += "//    " + " ".join(command) + "\n"
    proc = subprocess.run(command, capture_output=True, text=True)
    # Write command output
    with open(output_file_name, "w") as f:
        f.write(proc.stdout)


def run_mlir2FileCheck(
    model_file_name, compiled_file_name, m2fc_command, output_file_name
):
    global debug, debug_command_str

    if not m2fc_command:
        m2fc_command = "mlir2FileCheck.py"
    # Add the directory of this file to mlir2FileCheck.py
    directory = os.path.dirname(sys.argv[0])
    m2fc_command = directory + "/" + m2fc_command
    # There are some issues with spaces inside the json map/arrays
    m2fc_command = re.sub(r"\s*,\s*", r",", m2fc_command)
    m2fc_command = re.sub(r"\s*:\s*", r":", m2fc_command)
    # Gen command from string.
    command = ["python3"]
    command.extend(m2fc_command.split())
    command.extend(["-i", compiled_file_name])
    command.extend(["-m", model_file_name])
    if debug:
        debug_command_str += "//    " + " ".join(command) + "\n"
    res = subprocess.run(command, capture_output=True, text=True).stdout
    # Write command output
    with open(output_file_name, "w") as f:
        f.write(res)


# return True on success
def run_FileCheck(test_name, compiled_file_name, model_file_name, silent):
    global debug, debug_command_str, test_error_functions
    command = ["FileCheck", "--input-file=" + compiled_file_name, model_file_name]
    if debug:
        debug_command_str += "//    " + " ".join(command) + "\n"
    res = subprocess.run(command, capture_output=True, text=True).stderr
    if len(res) == 0:
        # Success.
        if not silent:
            dprint('// >> Successful test of "' + test_name + '".')
        return True
    # Failure
    if not silent:
        test_error_functions.append(test_name)
        dprint('// >> Start failure report of test "' + test_name + '".')
        dprint(res)
        dprint('// >> Stop failure report of test "' + test_name + '".')
        dprint(
            '// >> run again with option "-tdf '
            + test_name
            + '" to focus on this test.'
        )
    return False


def print_file(file_name):
    with open(file_name, "r") as f:
        print(f.read())


################################################################################
# Process segments.


def gen_orig_model(i, output_file_name):
    # Create the input file.
    with open(output_file_name, "w") as f:
        for l in segment_text[i]:
            f.write(l + "\n")


def emit_unmodified_segment(i):
    if i > 0:
        print("// -----")
    for l in segment_text[i]:
        print(l)


def emit_modified_segment(i, has_test):
    global run_command, debug
    global flt_orig_model_file_name, flt_compiled_file_name
    global flt_new_model_file_name
    global segment_text, segment_fct_name, segment_mlir2FileCheck_command

    # Print separator.
    if i > 0:
        print("// -----\n")
    # Print leading comments up to the function header line.
    for l in segment_text[i]:
        if re.match(r"\s*func", l) is not None:
            # Has function, stop.
            break
        if re.match(r"\s*//", l) is not None and re.match(r"\s*// CHECK", l) is None:
            # Has comment, print.
            print(l)
    print()
    gen_orig_model(i, flt_orig_model_file_name)
    run_onnx_mlir_opt(flt_orig_model_file_name, run_command, flt_compiled_file_name)

    # either we fix in all cases, or we failed the original test.
    run_mlir2FileCheck(
        flt_orig_model_file_name,
        flt_compiled_file_name,
        segment_mlir2FileCheck_command[i],
        flt_new_model_file_name,
    )
    print_file(flt_new_model_file_name)

    if has_test:
        run_FileCheck(
            segment_fct_name[i],
            flt_compiled_file_name,
            flt_new_model_file_name,
            silent=False,
        )


# return true on success
def test_orig_model(i, silent):
    global run_command, debug
    global flt_orig_model_file_name, flt_compiled_file_name

    gen_orig_model(i, flt_orig_model_file_name)
    run_onnx_mlir_opt(flt_orig_model_file_name, run_command, flt_compiled_file_name)
    return run_FileCheck(
        segment_fct_name[i], flt_compiled_file_name, flt_orig_model_file_name, silent
    )


################################################################################
# Main.


def main(argv):
    global run_command, fix_fct_name, debug
    global segment_text, segment_fct_name, segment_mlir2FileCheck_command
    input_command = "fixLitTest.py"
    has_fct = False
    has_repair = False
    has_test = False
    has_print = False
    has_repair_all = False
    try:
        opts, args = getopt.gnu_getopt(
            argv, "rtdf:hp", ["repair", "test", "debug", "func=", "help", "print"]
        )
    except getopt.GetoptError:
        print_usage("unknown options", options=True)
    for opt, arg in opts:
        if opt in ("-r", "--repair"):
            has_repair = True
        elif opt in ("-t", "--test"):
            has_test = 1
        elif opt in ("-p", "--print"):
            has_print = 1
        elif opt in ("-d", "--debug"):
            debug = 1
        elif opt in ("-f", "--func"):
            fix_fct_name = arg
            has_fct = True
            debug = 1  # debug on default with -f option
        elif opt in ("-h", "--help"):
            print_usage(options=True, usage=True, file_format=True)

    if not has_repair and not has_test:
        has_test = 1

    if len(args) != 1:
        # All commands after the file name seems to be added here!!!
        dprint("Need an single input file as last option: ", args, ".")
        return
    lit_test_filename = args[0]
    if not os.path.exists(lit_test_filename):
        # If don't find the path, try in the test/mlir sub directory.
        directory = os.path.dirname(sys.argv[0])
        # This file is in onnx-mlir/utils... tests are in onnx-mlir/test/mlir.
        directory += "/../test/mlir/"
        lit_test_filename = directory + lit_test_filename
    if debug:
        dprint('// Process lit test file "' + lit_test_filename + '".')
    if not os.access(lit_test_filename, os.R_OK):
        print_usage('could not open file "' + lit_test_filename + '"')

    # Process the lit test file.
    # Segments are all of the text between "// -----".
    # Ensure at most one function per segment.
    # Current segment data.
    curr_segment_text = []
    curr_segment_fct_name = ""
    curr_segment_mlir2FileCheck_command = ""
    # Counters.
    run_command_num = 0
    fct_between_delimiters = 0
    found_fct_to_fix = False
    # Scan file and write info into segment database.
    for line in open(lit_test_filename, "r"):
        # Handle segments.
        if re.match(r"// -----", line) is not None:
            # Has a new segment.
            segment_text.append(curr_segment_text)
            curr_segment_text = []
            segment_fct_name.append(curr_segment_fct_name)
            curr_segment_fct_name = ""
            segment_mlir2FileCheck_command.append(curr_segment_mlir2FileCheck_command)
            curr_segment_mlir2FileCheck_command = ""
            fct_between_delimiters = 0
        else:
            # Not a new segment, append line to current segment.
            l = line.rstrip()
            curr_segment_text.append(l)
        # Look for RUN command.
        m = re.match(r"// RUN:\s*(.*)\|", line)
        if m is not None:
            if run_command_num > 0:
                print_usage('Got too many "// RUN:" command.', file_format=True)
            run_command_num = 1
            run_command = m.group(1)
            # Strip the "%s" and ("--split-input-file" or "-split-input-file")
            run_command = run_command.replace("%s", "")
            run_command = run_command.replace("--split-input-file", "")
            run_command = run_command.replace("-split-input-file", "")
            continue
        # Handle function
        m = re.match(r"\s*func.*@(\w+)\(", line)
        if m is not None:
            curr_segment_fct_name = m.group(1)
            if fct_between_delimiters > 0:
                print_usage(
                    'Got too many function bodies between "// -----" command starting with '
                    + curr_segment_fct_name,
                    file_format=True,
                )
            fct_between_delimiters = 1
            if has_fct and curr_segment_fct_name == fix_fct_name:
                found_fct_to_fix = True
                if debug:
                    dprint("// Found function to fix: " + curr_segment_fct_name)
            continue
        # Handle mlir2FileCheck command
        m = re.match(r"\s*//\s*(mlir2FileCheck.py.*)$", line)
        if m is not None:
            curr_segment_mlir2FileCheck_command = m.group(1)
    # Record the last segment.
    segment_text.append(curr_segment_text)
    segment_fct_name.append(curr_segment_fct_name)
    segment_mlir2FileCheck_command.append(curr_segment_mlir2FileCheck_command)

    # Make sure we got what we were waiting for.
    if len(segment_text) < 2:
        print_usage(
            'Expected at least 2 segments (text between "// -----"): one for RUN command and one for a function.',
            options=False,
        )

    if has_fct and not found_fct_to_fix:
        dprint("Did not find function to fix: '" + fix_fct_name + "'.")
        sys.exit()
    if run_command_num == 0:
        print_usage('Expected "// RUN:" command.', file_format=True)

    # Process segments.
    dprint('// File runs "' + run_command + '" ')
    if has_repair:
        emit_unmodified_segment(0)
    for i in range(1, len(segment_text)):
        if has_fct:
            if segment_fct_name[i] == fix_fct_name:
                # We have the selected function.
                if has_repair:
                    dprint("// > repair " + segment_fct_name[i])
                    emit_modified_segment(i, has_test)
                elif has_test:
                    test_orig_model(i, silent=False)
            elif has_print:
                # We don't have the selected function but we want to print the others.
                if debug:
                    dprint("// > print " + segment_fct_name[i])
                emit_unmodified_segment(i)
        elif has_repair:
            # Has repair for all (failing) tests.
            if test_orig_model(i, silent=True):
                dprint(" // > successful test; print " + segment_fct_name[i])
                emit_unmodified_segment(i)
            else:
                dprint(" // > failed test; repair " + segment_fct_name[i])
                emit_modified_segment(i, has_test)
        elif has_test:
            test_orig_model(i, silent=False)

    if debug:
        dprint("// Commands used:")
        dprint(debug_command_str)

    test_error_num = len(test_error_functions)
    if has_test:
        if test_error_num == 0:
            dprint("\n>> Tested successfully without errors.")
        else:
            dprint("\n>> Tested with " + str(test_error_num) + " errors:")
            for f in test_error_functions:
                dprint(">>   " + f)
    dprint(">> Completed processing of " + lit_test_filename + "\n")


if __name__ == "__main__":
    main(sys.argv[1:])
