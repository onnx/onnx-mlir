"""Read .npy files saved by run-torch.py and return their contents as numpy arrays."""

import numpy as np
import sys
import time

import argparse


def check_positive(argname, value):
    value = int(value)
    if value <= 0:
        parser.error("Value passed to {} must be positive".format(argname))
    return value


def check_non_negative(argname, value):
    value = int(value)
    if value < 0:
        parser.error("Value passed to {} must be non-negative".format(argname))
    return value


def parse_args(parser=None):
    """Parse command line arguments."""
    if parser is None:
        parser = argparse.ArgumentParser(
            description="Use onnx-mlir compiled model to run inference"
        )
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        default="./z17-nnpa.so",
        help="Path to the onnx-mlir compiled model",
    )
    parser.add_argument(
        "--atol",
        type=float,
        default=1e-1,
        help="Absolute tolerance for numerical comparisons (default: 1e-1)",
    )
    parser.add_argument(
        "--rtol",
        type=float,
        default=5e-2,
        help="Relative tolerance for numerical comparisons (default: 5e-2)",
    )
    parser.add_argument(
        "-w",
        "--warmup",
        type=lambda s: check_non_negative("--warmup", s),
        default=0,
        help="The number of warmup inference runs.",
    )
    parser.add_argument(
        "-n",
        "--n-iteration",
        type=lambda s: check_positive("--n-iteration", s),
        default=1,
        help="The number of inference runs excluding warmup.",
    )

    return parser.parse_args()


def read_npy_files(filenames):
    """Read a list of .npy files and return a dict mapping filename to numpy array.

    Args:
        filenames: list of .npy file paths to read.

    Returns:
        List of np.array read from .npy file.
    """
    arrays = []
    for path in filenames:
        arr = np.load(path)
        # Convert the data for endian
        arrays.append(arr.astype(arr.dtype.newbyteorder("=")))
    return arrays


"""
Compare two lists of numpy arrays element by element.
This is the implementation from RunONNXModel.py.
The tolerant differene is computed with the following formula
diff = float(atol) + float(rtol) * abs(ref_val)
"""


def compare_result(actual_outs, ref_outs, atol=1e-8, rtol=1e-5, debug=0):
    """Compare two lists of numpy arrays element by element.

    Parameters
    ----------
    actual : list of np.ndarray
        Actual arrays to check.
    reference : list of np.ndarray
        Expected arrays.
    atol : float
        Absolute tolerance for floating-point comparison.
    rtol : float
        Relative tolerance for floating-point comparison.

    Returns
    -------
    bool
        True if all arrays match within the given tolerances.
    """

    if len(ref_outs) != len(actual_outs):
        print(
            f"FAIL: list length mismatch: reference={len(reference)}, actual={len(actual)}"
        )
        return False

    all_pass = True
    for q_index, (q_ref_outs, q_actual_outs) in enumerate(zip(ref_outs, actual_outs)):
        if q_ref_outs.shape != q_actual_outs.shape:
            print(
                f"FAIL: output[{q_index}] shape mismatch: reference={q_ref_outs.shape}, actual={q_actual_outs.shape}"
            )
            all_pass = False
            continue

        if q_ref_outs.dtype != q_actual_outs.dtype:
            print(
                f"FAIL: output[{q_index}] dtype mismatch: reference={q_ref_outs.dtype}, actual={q_actual_outs.dtype}"
            )
            all_pass = False
            continue

        total_elements = 0
        mismatched_elements = 0
        max_atol = 0
        for index, actual_val in np.ndenumerate(q_actual_outs):
            total_elements += 1
            ref_val = q_ref_outs[index]
            this_atol = abs(ref_val - actual_val)
            max_atol = max(this_atol, max_atol)
            if np.issubdtype(q_actual_outs.dtype, np.dtype(bool).type):
                if ref_val == actual_val:
                    continue
            elif np.issubdtype(q_actual_outs.dtype, np.integer):
                if ref_val == actual_val:
                    continue
            else:
                # Use equation atol + rtol * abs(desired), that is used in assert_allclose.
                all_pass = False
                diff = float(atol) + float(rtol) * abs(ref_val)
                if abs(actual_val - ref_val) <= diff:
                    continue
            mismatched_elements += 1
            if debug >= 1:
                print(
                    "  at {}".format(index),
                    "mismatch {} (actual)".format(actual_val),
                    "vs {} (reference)".format(ref_val),
                )
        if mismatched_elements > 0:
            print(
                "Output {} got mismatched elements {}/{}, ({:.2f}%). Max absolute difference {:.2f}\n".format(
                    q_index,
                    mismatched_elements,
                    total_elements,
                    mismatched_elements / total_elements * 100.0,
                    max_atol,
                )
            )
    if all_pass:
        print("\n✓ Correct.\n")
    else:
        print("\n✗  Failed\n")
    return all_pass


def ordinal(n):
    suffix = ["th", "st", "nd", "rd", "th"][min(n % 10, 4)]
    if 11 <= (n % 100) <= 13:
        suffix = "th"
    return str(n) + suffix


def data_without_top_bottom_quartile(data, percent):
    data = np.array(sorted(data))
    trim = int(percent * data.size / 100.0)
    if trim == 0 or data.size - 2 * trim < 1:
        # Want at least one element, return as is.
        return data
    return data[trim:-trim]


def process_perf_results(perf_results):
    # Print statistics info, e.g., min/max/stddev inference time.
    print(
        "  Statistics 1 (excluding warmup),"
        " min, {:.6e}, max, {:.6e}, mean, {:.6e}, stdev, {:.6e}".format(
            np.min(perf_results),
            np.max(perf_results),
            np.mean(perf_results),
            np.std(perf_results, dtype=np.float64),
        )
    )
    t_perf_results = data_without_top_bottom_quartile(perf_results, 25)
    print(
        "  Statistics 2 (no warmup/quart.),"
        " min, {:.6e}, max, {:.6e}, mean, {:.6e}, stdev, {:.6e}".format(
            np.min(t_perf_results),
            np.max(t_perf_results),
            np.mean(t_perf_results),
            np.std(t_perf_results, dtype=np.float64),
        )
    )


def run_model_with_input_output_arrays(
    session,
    input_arrays,
    ref_outputs=None,
    warmup=0,
    repeat=1,
    debug=0,
    atol=0.1,
    rtol=0.05,
):
    try:
        for i in range(warmup):
            outputs = session.run(input_arrays, debug)
    except Exception as e:
        print(f"Inference {i} got exception: {e}")
        sys.exit(-1)

    # Copied from ONNXRunModel.py
    perf_results = []
    for i in range(repeat):
        start = time.perf_counter()
        outputs = session.run(input_arrays)
        end = time.perf_counter()
        elapsed = end - start
        perf_results += [elapsed]
        print("  {} iteration, {}, seconds".format(ordinal(i + 1), elapsed))
        session.print_instrumentation()
    if repeat > 1:
        process_perf_results(perf_results)

    if ref_outputs:
        all_match = compare_result(outputs, ref_outputs, atol, rtol)

        # Not sure whether to continue becuase the numerical difference
        # does not necessarily mean failure for end-to-end
        if not all_match:
            sys.exit(-1)

    return outputs


def run_model_with_input_output_files(
    session,
    input_files,
    ref_output_files=None,
    warmup=0,
    repeat=1,
    debug=0,
    atol=0.1,
    rtol=0.05,
):
    input_arrays = read_npy_files(input_files)
    if ref_output_files:
        ref_outputs = read_npy_files(ref_output_files)
    else:
        ref_outputs = None

    return run_model_with_input_output_arrays(
        session, input_arrays, ref_outputs, warmup, repeat, debug, atol, rtol
    )
