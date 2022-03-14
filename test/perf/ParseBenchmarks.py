# SPDX-License-Identifier: Apache-2.0

######################################################################################################
# There are four possible flags to add to this script:
#
# --run <Op>: Compute performance benchmarks for specified Op, and write to output file
#
# --runall: Compute performance benchmarks for all Ops in OpsWithPerformanceBenchmarks array
#
# --readrun <File Op>: Compute performance benchmarks for specified Op, and compare with benchmarks
# already written to specifiedfile (File must contain same Op)
#
# --compare <Op1 Op2>: Compare performance benchmarks written to file for both specified files (each
# should contain same Op)
######################################################################################################



import os
import subprocess
import sys
import time


if (len(sys.argv) < 2):
    raise RuntimeError(
        "Specify an option: --run|--runall|--readrun|--compare"
    )


if (not os.environ.get('ONNX_MLIR_HOME', None)):
    raise RuntimeError(
        "Environment variable ONNX_MLIR_HOME is not set, please set it to the path to "
        "the HOME directory for onnx-mlir. The HOME directory for onnx-mlir refers to "
        "the parent folder containing the bin, lib, etc sub-folders in which ONNX-MLIR "
        "executables and libraries can be found, typically `onnx-mlir/build/Debug`"
    )


# Add compatible ops here
OpsWithPerformanceBenchmarks = [
    'Gemm',
    'Conv'
]


RuntimeEpoch = int(time.time())



def parse_benchmark_csv(result):

    ResultLines = result.splitlines()
    ResultLines.pop(0)

    VersionResults = []

    for line in ResultLines:
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


# Validates that written file and specified Op are the same Op, and truncates unneeded first lines in file
def ExtractResultFromFile(filename, op):
    Contents = open(filename, "r").read().splitlines()

    ResultString = ""

    for i in range(10):
        if (i == 1):
            if (op not in Contents[0]):
                raise RuntimeError(
                    "Written file might not contain the same Op as Op specified!"
                )
        Contents.pop(0)
    
    for c in Contents:
        ResultString += c + "\n"
    
    return ResultString


# Runs Benchmark binary and returns CSV-format result
def RunPerformanceBenchmark(Op):

    PerfOp = "Perf" + Op

    OutName = PerfOp + "_Benchmark_" + str(RuntimeEpoch)

    BenchmarkCommand = os.path.join(os.environ['ONNX_MLIR_HOME'], "build/Debug/bin", PerfOp)

    BenchmarkOptions = ["--benchmark_format=csv", "--benchmark_out=" + OutName, "--benchmark_out_format=csv"]

    print("Writing results to " + OutName)
    print("Running " + PerfOp + " ...")

    result = (subprocess.run([BenchmarkCommand] + BenchmarkOptions, capture_output=True, text=True)).stdout

    return result



def CompareBenchmarkResults(BenchmarkLists1, BenchmarkLists2):

    AllComparedEntries = []

    for i in range(len(BenchmarkLists1)):

        BenchmarkEntry = []
        for j in range(len(BenchmarkLists1[i])):
            if (j == 0):
                BenchmarkEntry.append(BenchmarkLists1[i][0])
            else:
                Value1 = (float)(BenchmarkLists1[i][j])
                Value2 = (float)(BenchmarkLists2[i][j])
                Difference = round(Value2 - Value1, 2)
                BenchmarkEntry.append(Difference)
        AllComparedEntries.append(BenchmarkEntry)
            
        # print(BenchmarkLists1[i])
        # print(BenchmarkLists2[i])
        # print("\n")

    for m in range(len(AllComparedEntries)):
        print(AllComparedEntries[m])



# Compute performance benchmarks for specified Op, and write to output file
if (sys.argv[1] == "--run"):
    if (len(sys.argv) != 3):
        raise RuntimeError(
            "Specify an Op"
        )
    Op = sys.argv[2]
    BenchmarkOutput = RunPerformanceBenchmark(Op)

    BenchmarkLists = parse_benchmark_csv(BenchmarkOutput)

    for Output in BenchmarkLists:
        print(Output)



# Compute performance benchmarks for all Ops in OpsWithPerformanceBenchmarks array
elif (sys.argv[1] == "--runall"):
    for Op in OpsWithPerformanceBenchmarks:
        BenchmarkOutput = RunPerformanceBenchmark(Op)

        BenchmarkLists = parse_benchmark_csv(BenchmarkOutput)

        for Output in BenchmarkLists:
            print(Output)
        print("\n")



# Compute performance benchmarks for specified Op, and compare with benchmarks
# already written to specifiedfile (File must contain same Op)
elif(sys.argv[1] == "--readrun"):
    if (len(sys.argv) != 4):
        raise RuntimeError(
            "Specify File and an Op"
        )

    InFile = sys.argv[2]
    Op = sys.argv[3]

    FileBenchmarks = ExtractResultFromFile(InFile, Op)

    BenchmarkLists1 = parse_benchmark_csv(FileBenchmarks)
    BenchmarkLists2 = parse_benchmark_csv(RunPerformanceBenchmark(Op))

    for b in BenchmarkLists1:
        print(b)
    print("\n")
    for b in BenchmarkLists2:
        print(b)
    
    print("\n")
    CompareBenchmarkResults(BenchmarkLists1, BenchmarkLists2)



# Compare performance benchmarks written to file for both specified files (each
# should contain same Op) (Unfinished)
elif(sys.argv[1] == "--compare"):
    print("Compare")
    # Check if specified inputs are compatible

print("\n")
