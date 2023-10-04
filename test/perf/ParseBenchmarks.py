# SPDX-License-Identifier: Apache-2.0

########################################################################################################
# There are four possible run args to add to this script (select one):
#
# --run <op>: Compute performance benchmarks for specified op, and write to output file
#
# --runall: Compute performance benchmarks for all Ops in OpsWithPerformanceBenchmarks array
#
# --readrun <filename> <op> <metric>: Compute performance benchmarks for specified op, and compare
# with benchmarks already written to specified file (File must contain the same op)
#
# --compare <op> <op> <metric>: Compare performance benchmarks written to file for both specified
# files (each should contain the same op)
#
#
# Further, there are two options:
#
# --verbose: Print all non-null benchmarks to stdout. (When using --compare or --readrun, output will
# be generated for each set of benchmarks (from arg 1 and arg 2)).
#
# --max-relative-slowdown <pct>: When using --readrun or --compare, indicate on each line whether
# the relative change exceeds the specified percent increase, and exit(1) if more than one benchmark
# does so. (Exit(0) if one or fewer do so).
#
#
########################################################################################################


import os
import subprocess
import sys
import time

########################################################################################################
# Global variables

# These indicate argument selection
Run = False
Runall = False
Readrun = False
Compare = False
PrintOutputs = False
LimitSlowdown = False

# Activated in the event that more than one compared metric exceeds specified max slowdown
SlowdownLimitExceeded = False

# Epoch timestamp at time script is run for filename purposes
RuntimeEpoch = int(time.time())

# This shall contain additional options added to arguments
ArgOptions = []

# Add compatible ops here; used when script is run with '--runall'
OpsWithPerformanceBenchmarks = ["Gemm", "Conv"]

# List of argument flags to allow
ValidArgs = ["verbose", "max-relative-slowdown", "run", "runall", "readrun", "compare"]

########################################################################################################


# Checking whether ONNX_MLIR_HOME environment variable is set
def check_home_env_var():
    if not os.environ.get("ONNX_MLIR_HOME", None):
        raise RuntimeError(
            "Environment variable ONNX_MLIR_HOME is not set. Please set it to the path to `/workdir/onnx-mlir/`\n"
            "To do this manually on Linux:\n\n"
            "# pushd /workdir/onnx-mlir/\n"
            "# ONNX_MLIR_HOME=$(pwd)\n"
            "# export ONNX_MLIR_HOME\n"
            "# popd\n"
        )


# Error message function
def print_usage(error_message):
    print("\nError: " + error_message)
    print(
        "Correct usage below:\n"
        "ParseBenchmarks.py [{run args}] [{options}]\n"
        "Run args:\n"
        "--run <op>\n"
        "--runall\n"
        "--readrun <filename> <op> <metric>\n"
        "--compare <filename> <filename> <metric>\n\n"
        "Options:\n"
        "--verbose\n--max-relative-slowdown <percent value>\n"
    )
    sys.exit()


# Function extracts supplied arguments safely
def ReadSysArgs():
    # Subroutine for handling each argument's supplied options
    def ValidateOptions(i, NumberOfExpectedOptions):
        arg = sys.argv[i]
        if len(sys.argv) <= i + NumberOfExpectedOptions:
            print_usage("Missing " + arg + " option(s)")

        for j in range(1, NumberOfExpectedOptions + 1):
            if (sys.argv[i + j])[:2] == "--":
                print_usage("Missing " + arg + " option(s)")

            ArgOptions.append(sys.argv[i + j])

            # Ignore option in future iterations
            sys.argv[i + j] = "/ignorethis"

    # Number of supplied run arguments (must not exceed 1)
    SuppliedRunArgs = 0

    # Looping through all supplied arguments
    for i in range(len(sys.argv)):
        # Skipping argument denoting name of script
        if i == 0:
            continue

        arg = sys.argv[i]
        # Skipping ignore option
        if arg == "/ignorethis":
            continue

        if arg[:2] != "--":
            print_usage("Invalid argument: " + arg)

        argname = arg[2:].lower()
        if argname not in ValidArgs:
            print_usage("Invalid argument: " + arg)

        # --verbose argument
        if argname == "verbose":
            if SuppliedRunArgs < 1:
                print_usage("Supply run args before optional args")

            global PrintOutputs
            PrintOutputs = True

        # --max-relative-slowdown argument
        elif argname == "max-relative-slowdown":
            if SuppliedRunArgs < 1:
                print_usage("Supply run args before optional args")

            global LimitSlowdown
            LimitSlowdown = True

            # Set for MRS
            NumberOfExpectedOptions = 1

            # Check that the options are supplied
            ValidateOptions(i, NumberOfExpectedOptions)

        # --run argument
        elif argname == "run":
            global Run
            Run = True
            SuppliedRunArgs += 1

            # Set for op
            NumberOfExpectedOptions = 1

            # Check that the options are supplied
            ValidateOptions(i, NumberOfExpectedOptions)

        # --runall argument
        elif argname == "runall":
            global Runall
            Runall = True
            SuppliedRunArgs += 1

        # --readrun argument
        elif argname == "readrun":
            global Readrun
            Readrun = True
            SuppliedRunArgs += 1

            # Set for filename, op, and metric
            NumberOfExpectedOptions = 3

            # Check that the options are supplied
            ValidateOptions(i, NumberOfExpectedOptions)

        # --compare argument
        elif argname == "compare":
            global Compare
            Compare = True
            SuppliedRunArgs += 1

            # Set for filename, filename, and metric
            NumberOfExpectedOptions = 3

            # Check that the options are supplied
            ValidateOptions(i, NumberOfExpectedOptions)

    if not any([Run, Runall, Readrun, Compare]):
        print_usage("No valid argument selected")
    if SuppliedRunArgs > 1:
        print_usage("Too many arguments selected")


# Validates that both written files are the same op, and truncates unneeded first lines from files
def CompareFileAndFile(filename1, filename2):
    Contents1 = open(filename1, "r").read().splitlines()
    Contents2 = open(filename2, "r").read().splitlines()

    ResultString1 = ""
    ResultString2 = ""

    for i in range(10):
        if i == 1:
            PossibleOp = Contents1[0].split("Perf")[1]
            if PossibleOp not in Contents2[0]:
                raise RuntimeError("Written files might not contain the same op")
        Contents1.pop(0)
        Contents2.pop(0)

    for c in Contents1:
        ResultString1 += c + "\n"

    for c in Contents2:
        ResultString2 += c + "\n"

    return (ResultString1, ResultString2)


# Reads supplied CSV file, truncates unneeded lines, and returns benchmark
# output in the same format and shape as when run fresh
def ExtractFileOutput(filename):
    OutputLines = open(filename, "r").read().splitlines()

    # Truncating hardware info in first lines of file
    for line in OutputLines:
        if line[5:] == "name,":
            break
        else:
            OutputLines.pop(0)

    # Merging lines back together for use by other functions
    RawBenchmarkOutput = ""
    for line in OutputLines:
        RawBenchmarkOutput += line + "\n"

    return RawBenchmarkOutput


# Runs Benchmark binary and writes CSV results to file
# Function returns CSV-format result
def RunPerformanceBenchmark(op):
    PerfOp = "Perf" + op

    # Filename of output file containing benchmark output
    OutName = PerfOp + "_Benchmark_" + str(RuntimeEpoch)

    BenchmarkCommand = os.path.join(
        os.environ["ONNX_MLIR_HOME"], "build/Debug/bin", PerfOp
    )
    BenchmarkOptions = [
        "--benchmark_format=csv",
        "--benchmark_out=" + OutName,
        "--benchmark_out_format=csv",
    ]

    print("Running " + PerfOp + " ...")

    result = (
        subprocess.run(
            [BenchmarkCommand] + BenchmarkOptions, capture_output=True, text=True
        )
    ).stdout

    print("Results written to " + OutName)

    return result


def CompareOutput(output1, output2, metric, MaxRelativeSlowdown):
    # Number of benchmarks exceeding max relative slowdown
    ExceededLimit = 0

    OutputDicts1 = ReadCSVOutput(output1)
    OutputDicts2 = ReadCSVOutput(output2)

    ComparisonOutput = ""

    if metric in ["cpu_time", "real_time"]:
        ComparisonOutput += "# Negative values indicate a reduction from arg 1 to arg 2, meaning arg 2 is faster.\n\n"

    for OutputDict1 in OutputDicts1:
        dict1name = OutputDict1["name"]
        for OutputDict2 in OutputDicts2:
            dict2name = OutputDict2["name"]
            if dict1name == dict2name:
                if metric in OutputDict1.keys() and metric in OutputDict2.keys():
                    Value1 = (
                        (float)(OutputDict1[metric]) if (OutputDict1[metric]) else 0
                    )
                    Value2 = (
                        (float)(OutputDict2[metric]) if (OutputDict2[metric]) else 0
                    )
                    Difference = round(Value2 - Value1, 2)
                    DifferenceStr = str(Difference)
                    Pct = round((Difference / Value1) * 100, 2)
                    PctStr = str(Pct)

                    if metric in ["cpu_time", "real_time"]:
                        TimeUnit = OutputDict1["time_unit"]
                        DifferenceStr += " " + TimeUnit

                    ComparisonOutput += dict1name + " " + metric + " delta: "
                    ComparisonOutput += DifferenceStr + " (" + PctStr + "%) "
                    ComparisonOutput += (
                        "("
                        + str(round(Value1, 2))
                        + " -> "
                        + str(round(Value2, 2))
                        + ")"
                    )

                    if LimitSlowdown and float(MaxRelativeSlowdown) < Pct:
                        ComparisonOutput += (
                            " [EXCEEDS MAX RELATIVE SLOWDOWN ("
                            + str(MaxRelativeSlowdown)
                            + "%)]"
                        )
                        ExceededLimit += 1

                    ComparisonOutput += "\n"

    if ExceededLimit > 1:
        global SlowdownLimitExceeded
        SlowdownLimitExceeded = True

    return ComparisonOutput


# Convert raw CSV output to dictionary format
def ReadCSVOutput(result):
    OutputDicts = []

    OutputLines = result.splitlines()

    # Get list of identifiers for result (name, iterations, cpu time, etc.)
    CSVIdentifiers = OutputLines.pop(0).split(",")

    for i in OutputLines:
        newdict = {}
        LineEntries = i.split(",")
        for j in CSVIdentifiers:
            newdict[j] = LineEntries.pop(0)
        OutputDicts.append(newdict)

    return OutputDicts


# Called only if --verbose flag is supplied
def WriteFormattedOutput(RawBenchmarkOutput, SpecificArg):
    # List of dictionaries for each benchmark supplied by raw CSV output
    OutputDicts = ReadCSVOutput(RawBenchmarkOutput)

    for OutputDict in OutputDicts:
        # If using --verbose with --compare or --readrun, each output segment will
        # indicate whether it came from arg 1 or arg 2.
        if SpecificArg is not None:
            print("# arg = " + SpecificArg)

        # Keeps track of greatest number of characters in a key for clean printing
        LongestKey = 0
        for key in OutputDict.keys():
            if (len(key) > LongestKey) and (OutputDict[key]):
                LongestKey = len(key)

        for key in OutputDict.keys():
            if OutputDict[key]:
                SpaceBuffer = LongestKey - len(key)
                print(key + " " * (SpaceBuffer + 2) + OutputDict[key])

        print("-" * 40)


# Main function
def main():
    # Check that ONNX_MLIR_HOME is set to path of /workdir/onnx-mlir
    check_home_env_var()

    # Read in script arguments safely
    ReadSysArgs()

    # Compute performance benchmarks for specified op, and write to output file
    if Run:
        op = ArgOptions[0]

        # Get CSV benchmark results and write to output file
        RawBenchmarkOutput = RunPerformanceBenchmark(op)

        # Set by --verbose argument
        if PrintOutputs:
            WriteFormattedOutput(RawBenchmarkOutput, None)

    # Compute performance benchmarks for all Ops in OpsWithPerformanceBenchmarks array
    elif Runall:
        for op in OpsWithPerformanceBenchmarks:
            # Get CSV benchmark results and write to output file
            RawBenchmarkOutput = RunPerformanceBenchmark(op)

            # Set by --verbose argument
            if PrintOutputs:
                WriteFormattedOutput(RawBenchmarkOutput, None)

    # Compute performance benchmarks for specified op, and compare with benchmarks
    # already written to specifiedfile (File must contain same op)
    elif Readrun:
        filename = ArgOptions[0]
        op = ArgOptions[1]
        metric = ArgOptions[2]
        maxrelativeslowdown = None

        if LimitSlowdown:
            maxrelativeslowdown = ArgOptions[3]

        # Handling file
        RawBenchmarkOutput1 = ExtractFileOutput(filename)

        # Handling op
        # Get CSV benchmark results and write to output file
        RawBenchmarkOutput2 = RunPerformanceBenchmark(op)

        # Set by --verbose argument
        if PrintOutputs:
            WriteFormattedOutput(RawBenchmarkOutput1, filename)
            WriteFormattedOutput(RawBenchmarkOutput2, op)

        ComparisonOutput = CompareOutput(
            RawBenchmarkOutput1, RawBenchmarkOutput2, metric, maxrelativeslowdown
        )

        print(ComparisonOutput)

    # Compare performance benchmarks written to file for both specified files (each
    # should contain same op)
    elif Compare:
        filename1 = ArgOptions[0]
        filename2 = ArgOptions[1]
        metric = ArgOptions[2]
        maxrelativeslowdown = None

        if LimitSlowdown:
            maxrelativeslowdown = ArgOptions[3]

        RawBenchmarkOutput1 = ExtractFileOutput(filename1)

        RawBenchmarkOutput2 = ExtractFileOutput(filename2)

        # Set by --verbose argument
        if PrintOutputs:
            WriteFormattedOutput(RawBenchmarkOutput1, filename1)
            WriteFormattedOutput(RawBenchmarkOutput2, filename2)

        ComparisonOutput = CompareOutput(
            RawBenchmarkOutput1, RawBenchmarkOutput2, metric, maxrelativeslowdown
        )

        print(ComparisonOutput)

    sys.exit(1) if (SlowdownLimitExceeded) else sys.exit(0)


main()
