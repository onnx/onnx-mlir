"""Read .npy files saved by run-torch.py and return their contents as numpy arrays."""

import numpy as np
import sys


def read_npy_files(filenames):
    """Read a list of .npy files and return a dict mapping filename to numpy array.

    Args:
        filenames: list of .npy file paths to read.

    Returns:
        dict[str, np.ndarray]: mapping from filename to loaded array.
    """
    arrays = {}
    for path in filenames:
        arr = np.load(path)
        # Convert the data for endian
        arrays[path] = arr.astype(arr.dtype.newbyteorder("="))
        # print(f"{path}: shape={arr.shape}, dtype={arr.dtype}")
    return arrays


"""Utility to compare two lists of numpy arrays (reference vs actual)."""


def compare_result_bob(actual, reference, atol=1e-8, rtol=1e-5):
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
    if len(reference) != len(actual):
        print(
            f"FAIL: list length mismatch: reference={len(reference)}, actual={len(actual)}"
        )
        return False

    all_pass = True
    for i, (ref, act) in enumerate(zip(reference, actual)):
        if ref.shape != act.shape:
            print(
                f"FAIL: output[{i}] shape mismatch: reference={ref.shape}, actual={act.shape}"
            )
            all_pass = False
            continue

        if ref.dtype != act.dtype:
            print(
                f"FAIL: output[{i}] dtype mismatch: reference={ref.dtype}, actual={act.dtype}"
            )
            all_pass = False
            continue

        if np.issubdtype(ref.dtype, np.integer):
            if not np.array_equal(ref, act):
                diff_count = int(np.count_nonzero(ref != act))
                print(
                    f"FAIL: output[{i}] (integer) has {diff_count} mismatched elements"
                )
                all_pass = False
            else:
                print(f"PASS: output[{i}] (integer) shape={ref.shape} exact match")
        elif np.issubdtype(ref.dtype, np.floating):
            if np.allclose(ref, act, rtol=rtol, atol=atol):
                print(
                    f"PASS: output[{i}] (float) shape={ref.shape} within rtol={rtol}, atol={atol}"
                )
            else:
                abs_diff = np.abs(ref - act)
                max_abs = float(np.max(abs_diff))
                # Avoid division by zero for relative diff
                denom = np.maximum(np.abs(ref), 1e-30)
                max_rel = float(np.max(abs_diff / denom))
                diff_count = int(
                    np.count_nonzero(~np.isclose(ref, act, rtol=rtol, atol=atol))
                )
                print(
                    f"FAIL: output[{i}] (float) shape={ref.shape} "
                    f"max_abs_diff={max_abs:.6e}, max_rel_diff={max_rel:.6e}, "
                    f"mismatched={diff_count}/{ref.size}"
                    f"comparison parameter atol = {atol} rtol = {rtol}"
                )
                all_pass = False
        else:
            # Fall back to exact comparison for other dtypes (bool, etc.)
            if np.array_equal(ref, act):
                print(
                    f"PASS: output[{i}] dtype={ref.dtype} shape={ref.shape} exact match"
                )
            else:
                diff_count = int(np.count_nonzero(ref != act))
                print(
                    f"FAIL: output[{i}] dtype={ref.dtype} has {diff_count} mismatched elements"
                )
                all_pass = False

    return all_pass


def compare_result(actual_outs, ref_outs, atol=1e-8, rtol=1e-5, debug=0):
    total_elements = 0
    mismatched_elements = 0
    max_atol = 0
    sum_rtol = 0
    # The outputs are of shape: #queries, seq_length, hidden_states, with
    # seq_length various with query.
    for q_index in range(len(actual_outs)):
        q_actual_outs = actual_outs[q_index]
        q_ref_outs = ref_outs[q_index]
        for index, actual_val in np.ndenumerate(q_actual_outs):
            total_elements += 1
            ref_val = q_ref_outs[index]
            this_atol = abs(ref_val - actual_val)
            max_atol = max(this_atol, max_atol)
            sum_rtol += abs((this_atol) / ref_val)
            if np.issubdtype(q_actual_outs.dtype, np.dtype(bool).type):
                if ref_val == actual_val:
                    continue
            else:
                # Use equation atol + rtol * abs(desired), that is used in assert_allclose.
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

    print(
        f"Max absolute difference: {max_atol}\nMean of relative difference {sum_rtol/total_elements}"
    )
    if mismatched_elements == 0:
        print("  correct.\n")
        return True
    else:
        print(
            "  got mismatched elements {}/{}, in percentage {}.\n".format(
                mismatched_elements,
                total_elements,
                mismatched_elements / total_elements,
            )
        )
        return False


def run_model_with_file(
    session, input_files, ref_output_files=None, atol=0.1, rtol=0.05
):
    input_dir = read_npy_files(input_files)
    input_arrays = [input_dir[f] for f in input_files]

    outputs = session.run(input_arrays)

    if ref_output_files:
        arrays = read_npy_files(ref_output_files)
        ref_outputs = [arrays[f] for f in ref_output_files]
        all_match = compare_result(outputs, ref_outputs, atol, rtol)
        if all_match:
            print("\n✓ All outputs match the reference within specified tolerances")
        else:
            print("\n✗ Some outputs do not match the reference")

    return outputs
