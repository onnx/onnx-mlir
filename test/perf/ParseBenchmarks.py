import subprocess
import sys

# Add compatible ops here
OpsWithPerformanceBenchmarks = [
    'Gemm',
    'Conv'
]

ExecutablePath = "/workdir/onnx-mlir/build/Debug/bin/"

CumulativeResults = []

for Op in OpsWithPerformanceBenchmarks:
    OpPerf = "Perf" + Op
    print("Capturing " + OpPerf)


    BenchmarkCommand = ExecutablePath + OpPerf
    result = (subprocess.run([BenchmarkCommand], capture_output=True, text=True)).stdout

    # print(result)
    ResultLines = result.splitlines()

    for line in ResultLines:
        tokens = line.split()

        if (len(tokens) != 8):
            continue

        LineContentResults = []

        LineContentResults.append("benchmark=" + tokens[0])
        LineContentResults.append("time=" + tokens[1] + "ms")
        LineContentResults.append("cpu=" + tokens[3] + "ms")
        LineContentResults.append("iterations=" + tokens[5])
        LineContentResults.append(tokens[6])
        LineContentResults.append(tokens[7])

        CumulativeResults.append(LineContentResults)

# Working output -- should be updated according to specific needs
for i in CumulativeResults:
    print(i)