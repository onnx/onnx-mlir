# SPDX-License-Identifier: Apache-2.0

######################################################################################################
# There are four possible flags to add to this script:
#
# --run <op>: Compute performance benchmarks for specified op, and write to output file
#
# --runall: Compute performance benchmarks for all Ops in OpsWithPerformanceBenchmarks array
#
# --readrun <File> <op>: Compute performance benchmarks for specified op, and compare with benchmarks
# already written to specified file (File must contain the same op)
#
# --compare <Op1> <Op2>: Compare performance benchmarks written to file for both specified files (each
# should contain the same op)
######################################################################################################



import os
import subprocess
import sys
import time



# Checking whether ONNX_MLIR_HOME environment variable is set
if (not os.environ.get('ONNX_MLIR_HOME', None)):
    raise RuntimeError(
        "Environment variable ONNX_MLIR_HOME is not set, please set it to the path to "
        "the HOME directory for onnx-mlir. The HOME directory for onnx-mlir refers to "
        "the parent folder containing the bin, lib, etc sub-folders in which ONNX-MLIR "
        "executables and libraries can be found, typically `onnx-mlir/build/Debug`"
    )


# Add compatible ops here; used when script is run with '--runall'
OpsWithPerformanceBenchmarks = [
    'Gemm',
    'Conv'
]


# Epoch timestamp at time script is run for filename purposes
RuntimeEpoch = int(time.time())

def print_usage(error_message):
    print("\nError: " + error_message)
    print("Correct usage below:")
    print("ParseBenchmarks.py [--run <op>|--runall|--readrun <file> <op>|--compare <op> <op>] [--verbose]\n")
    sys.exit()

# Global vars indicating argument selection
PrintOutputs = False
Run          = False
Runall       = False
Readrun      = False
Compare      = False

ArgOptions = []

# List of argument flags to allow
ValidArgs = [
    'verbose',
    'run',
    'runall',
    'readrun',
    'compare'
]

# Durable and reliable function to extract supplied arguments safely
def ReadSysArgs():

    # Number of supplied run arguments (must not exceed 1)
    SuppliedRunArgs = 0

    # Looping through all supplied arguments
    for i in range(len(sys.argv)):

        # Skipping argument denoting name of script
        if (i == 0):
            continue

        arg = sys.argv[i]
        # Skipping ignore option
        if (arg == "/ignorethis"):
            continue


        if (arg[:2] == "--"):
            argname = arg[2:].lower()
            if argname not in ValidArgs:
                print_usage("Invalid argument: " + arg)
            else:
                if (argname == "verbose"):
                    global PrintOutputs
                    PrintOutputs = True
                elif (argname == "run"):
                    global Run
                    Run = True
                    SuppliedRunArgs+=1
                    # Expect supplied op
                    if (len(sys.argv) > i+1):
                        ArgOptions.append(sys.argv[i+1])

                        # Ignore option argument in future iterations
                        sys.argv[i+1] = "/ignorethis"
                    else:
                        print_usage("Missing --run option (op expected)")
                elif (argname == "runall"):
                    global Runall
                    Runall = True
                    SuppliedRunArgs+=1
                elif (argname == "readrun"):
                    global Readrun
                    Readrun = True
                    SuppliedRunArgs+=1
                    
                    # Expect supplied file and supplied op
                    if (len(sys.argv) > i+2):
                        ArgOptions.append(sys.argv[i+1])
                        ArgOptions.append(sys.argv[i+2])

                        # Ignore option arguments in future iterations
                        sys.argv[i+1] = "/ignorethis"
                        sys.argv[i+2] = "/ignorethis"
                    else:
                        print_usage("Missing --readrun option(s) (file and op expected)")
                    
                elif (argname == "compare"):
                    global Compare
                    Compare = True
                    SuppliedRunArgs+=1

                    # Expect two supplied ops
                    if (len(sys.argv) > i+2):
                        ArgOptions.append(sys.argv[i+1])
                        ArgOptions.append(sys.argv[i+2])

                        # Ignore option arguments in future iterations
                        sys.argv[i+1] = "/ignorethis"
                        sys.argv[i+2] = "/ignorethis"
                    else:
                        print_usage("Missing --compare option(s) (op and op expected)")

        else:
            print_usage("Invalid argument: " + arg)
    
    if not any([Run, Runall, Readrun, Compare]):
        print_usage("No valid argument selected")
    if (SuppliedRunArgs > 1):
        print_usage("Too many arguments selected")



def ParseBenchmarkCSV(result):

    OutputLines = result.splitlines()
    OutputLines.pop(0)

    VersionResults = []

    for line in OutputLines:
        tokens = line.split(",")

        Metrics = []

        Metrics.append(tokens[0])
        Metrics.append(tokens[1])
        Metrics.append(tokens[2])
        Metrics.append(tokens[3])
        # Metrics.append(tokens[4])
        Metrics.append(tokens[10])
        Metrics.append(tokens[11])

        VersionResults.append(Metrics)

        # LineContentResults = ""
        # LineContentResults += "benchmark=" + Benchmark + " "
        # LineContentResults += "time=" + RealTime + TimeUnits + " "
        # LineContentResults += "cpu=" + CpuTime + TimeUnits + " "
        # LineContentResults += "iterations=" + Iterations + " "
        # LineContentResults += "FLOP=" + Flop + " "
        # LineContentResults += "FLOPS=" + FlopS

        # VersionResults.append[LineContentResults]
        # print(LineContentResults)


    return VersionResults



# Validates that both written files are the same op, and truncates unneeded first lines from files
def CompareFileAndFile(filename1, filename2):
    Contents1 = open(filename1, "r").read().splitlines()
    Contents2 = open(filename2, "r").read().splitlines()

    ResultString1 = ""
    ResultString2 = ""

    for i in range(10):
        if (i == 1):
            PossibleOp = Contents1[0].split("Perf")[1]
            if (PossibleOp not in Contents2[0]):
                raise RuntimeError(
                    "Written files might not contain the same op"
                )
        Contents1.pop(0)
        Contents2.pop(0)
    
    for c in Contents1:
        ResultString1 += c + "\n"
    
    for c in Contents2:
        ResultString2 += c + "\n"
    
    return (ResultString1, ResultString2)


# Validates that written file and specified op are the same op, and truncates unneeded first lines from file
def CompareOpAndFile(filename, op):
    Contents = open(filename, "r").read().splitlines()

    ResultString = ""

    for i in range(10):
        if (i == 1):
            if (op not in Contents[0]):
                raise RuntimeError(
                    "Written file might not contain the same op as op specified"
                )
        Contents.pop(0)
    
    for c in Contents:
        ResultString += c + "\n"
    
    return ResultString


# Runs Benchmark binary and writes CSV results to file
# Function returns CSV-format result
def RunPerformanceBenchmark(op):

    PerfOp = "Perf" + op

    # Filename of output file containing benchmark output
    OutName = PerfOp + "_Benchmark_" + str(RuntimeEpoch)

    BenchmarkCommand = os.path.join(os.environ['ONNX_MLIR_HOME'], "build/Debug/bin", PerfOp)
    BenchmarkOptions = ["--benchmark_format=csv", "--benchmark_out=" + OutName, "--benchmark_out_format=csv"]

    print("Running " + PerfOp + " ...")

    result = (subprocess.run([BenchmarkCommand] + BenchmarkOptions, capture_output=True, text=True)).stdout

    print("Results written to " + OutName)

    return result



def CompareBenchmarkResults(BenchmarkLists1, BenchmarkLists2):

    AllComparedEntries = []

    for i in range(len(BenchmarkLists1)):

        BenchmarkEntry = []
        for j in range(len(BenchmarkLists1[i])):
            if (j == 0):
                BenchmarkEntry.append(BenchmarkLists1[i][0])
            else:
                Value1 = (float)(BenchmarkLists1[i][j]) if (BenchmarkLists1[i][j]) else 0
                Value2 = (float)(BenchmarkLists2[i][j]) if (BenchmarkLists2[i][j]) else 0
                Difference = round(Value2 - Value1, 2)
                BenchmarkEntry.append(Difference)
        AllComparedEntries.append(BenchmarkEntry)
            
        # print(BenchmarkLists1[i])
        # print(BenchmarkLists2[i])
        # print("\n")

    for m in range(len(AllComparedEntries)):
        print(AllComparedEntries[m])


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

def WriteFormattedOutput(RawBenchmarkOutput):
    # List of dictionaries for each benchmark supplied by raw CSV output
    OutputDicts = ReadCSVOutput(RawBenchmarkOutput)

    for OutputDict in OutputDicts:
        # Keeps track of greatest number of characters in a key for clean printing
        LongestKey = 0
        for key in OutputDict.keys():
            if ((len(key) > LongestKey) and (OutputDict[key])):
                LongestKey = len(key)

        for key in OutputDict.keys():
            if (OutputDict[key]):
                SpaceBuffer = LongestKey - len(key)
                print(key + ' ' * (SpaceBuffer + 2) + OutputDict[key])


        print('-'*40)


# Read in script arguments safely
ReadSysArgs()

# Compute performance benchmarks for specified op, and write to output file
if (Run):
    op = ArgOptions[0]

    # Get CSV benchmark results and write to output file
    RawBenchmarkOutput = RunPerformanceBenchmark(op)

    # Set by --verbose argument
    if (PrintOutputs):
        WriteFormattedOutput(RawBenchmarkOutput)



# Compute performance benchmarks for all Ops in OpsWithPerformanceBenchmarks array
elif (Runall):
    for op in OpsWithPerformanceBenchmarks:

        # Get CSV benchmark results and write to output file
        RawBenchmarkOutput = RunPerformanceBenchmark(op)

        # Set by --verbose argument
        if (PrintOutputs):
            WriteFormattedOutput(RawBenchmarkOutput)



# Compute performance benchmarks for specified op, and compare with benchmarks
# already written to specifiedfile (File must contain same op)
# elif(Readrun):
#     InFile = ArgOptions[0]
#     op     = ArgOptions[1]

#     FileBenchmarks = CompareOpAndFile(InFile, op)

#     BenchmarkLists1 = ParseBenchmarkCSV(FileBenchmarks)
#     BenchmarkLists2 = ParseBenchmarkCSV(RunPerformanceBenchmark(op))

#     for b in BenchmarkLists1:
#         print(b)
#     print("\n")
#     for b in BenchmarkLists2:
#         print(b)
    
#     print("\n")
#     CompareBenchmarkResults(BenchmarkLists1, BenchmarkLists2)



# Compare performance benchmarks written to file for both specified files (each
# should contain same op)
# elif(Compare):
#     InFile1 = ArgOptions[0]
#     InFile2 = ArgOptions[1]

#     FileBenchmarks1, FileBenchmarks2 = CompareFileAndFile(InFile1, InFile2)

#     BenchmarkLists1 = ParseBenchmarkCSV(FileBenchmarks1)
#     BenchmarkLists2 = ParseBenchmarkCSV(FileBenchmarks2)

#     for b in BenchmarkLists1:
#         print(b)
#     print("\n")
#     for b in BenchmarkLists2:
#         print(b)
    
#     print("\n")
#     CompareBenchmarkResults(BenchmarkLists1, BenchmarkLists2)


