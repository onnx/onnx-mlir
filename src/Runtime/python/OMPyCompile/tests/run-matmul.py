import numpy as np
import OMPyInfer
import OMPyCompile
from pathlib import Path

import argparse


def main():
    parser = argparse.ArgumentParser(description="application arguments")
    parser.add_argument("--batchSize", type=int, default=1, help="batch size")
    parser.add_argument("--seqLength", type=int, default=256, help="sequence length")
    # batchSize positional integer (optional, default = 1)
    parser.add_argument(
        "--hiddenState", type=int, default=768, help="hidden state size"
    )
    parser.add_argument("--flags", type=str, default="-O3", help="compiler flags")
    args = OMPyInfer.parse_args(parser)

    # Use "matmul.mlir" as model
    # Path to make sure the script can be invoked in other directory.
    script_dir = Path(__file__).resolve().parent
    args.model = str(script_dir / "matmul.mlir")

    # inputs
    DIM = args.hiddenState
    input1 = np.random.default_rng().random(
        (args.batchSize, args.seqLength, DIM), dtype=np.float32
    )
    input2 = np.random.default_rng().random((DIM, DIM), dtype=np.float32)

    if args.model.endswith(".so"):
        compiled_model = args.model
    else:
        try:
            compile_session = OMPyCompile.OMCompile(args.model, args.flags)
        except RuntimeError as e:
            print(f"Compilation failed: {e}")
            exit(1)

        compiled_model = compile_session.get_output_file_name()

    run_session = OMPyInfer.InferenceSession(compiled_model)
    outputs = OMPyInfer.run_model_with_input_output_arrays(
        run_session,
        [input1, input2],
        None,
        warmup=args.warmup,
        repeat=args.n_iteration,
        atol=args.atol,
        rtol=args.rtol,
    )


if __name__ == "__main__":
    main()
