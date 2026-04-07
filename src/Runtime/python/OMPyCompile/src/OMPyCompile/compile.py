from . import compiler_bin
import subprocess
def compile(model, flags):
    compiler=compiler_bin + '/onnx-mlir'
    command = compiler_bin + " " + flags + " " + model
    print(command)
    result = subprocess.run(
            [compiler, model],
            capture_output=True,
            text=True,
            check=False)
    print(result.stdout)
    print(result.stderr)
    return command
