import numpy as np
import om_pyrt
import os

import argparse

def main():
    parser = argparse.ArgumentParser(description="Run example for fixed size KV cache arguments")
    # seqLength positional integer (required)
    parser.add_argument("--cache-size", type=int, default=16, help="max cache size")
    parser.add_argument("--step", type=int, default=3, help="how many steps")
    parser.add_argument("--batch", type=int, default=1, help="batch size")
    parser.add_argument("--cache-type", type=str, default="fixed", help="kv cache type, fixed or growing")

    args = om_pyrt.utils.parse_args(parser)
    print(args)

    # inputs
    head = 12
    STATES = 64
    if args.cache_size > args.step:
        print("May cause cache overflow unless circular cache is used")
    if args.cache_type = "fixed"
        K = np.full((args.batch, head, args.cache_size, STATES), 0.0, dtype=np.float32)
        V = np.full((args.batch, head, args.cache_size, STATES), 0.0, dtype=np.float32)
    else:
        K = np.full((args.batch, head, i, STATES), 0.0, dtype=np.float32)
        V = np.full((args.batch, head, i, STATES), 0.0, dtype=np.float32)
    for i in range(args.step):
        print(f"step {i}")
        pos_id = np.full(args.batch, i, dtype=np.int64)
        token = np.full((args.batch, 1, head*STATES), i+1.0, dtype=np.float32)
        if args.cache_type == "fixed":
            inputArrays = [token, K, V, pos_id]
        else:
            inputArrays = [token, K, V]
        refArrays = None
        sess = om_pyrt.InferenceSession(args.model)
        output = om_pyrt.run_model_with_input_output_arrays(sess, inputArrays, refArrays, args.warmup, args.n_iteration)
        K = output[1]
        V = output[2]
        print(K[0, 0, i, 0])

if __name__ == "__main__":
    main()
