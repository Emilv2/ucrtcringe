import subprocess
import numpy as np
import os
import struct
import sys

from asin import asin as py_asin

c_file = "asin_test.c"
exe_name = "asin_test.exe"
script_dir = os.path.dirname(os.path.abspath(__file__))
c_path = os.path.join(script_dir, c_file)
exe_path = os.path.join(script_dir, exe_name)
bin_path_native = os.path.join(script_dir, "asin_test_native.bin")
bin_path_timing = os.path.join(script_dir, "asin_test_timing.bin")

RECORD_SIZE = 16  # 2 doubles per record (input, output)
BATCH_SIZE = 1024 * 1024  # Number of records per batch


def compile_asin_test():
    print("Compiling asin_test.exe from asin_test.c...")
    result = subprocess.run(
        [
            "x86_64-w64-mingw32-gcc",
            "-O2",
            "-o", exe_name,
            c_file,
            "-lm"
        ],
        cwd=script_dir,
        capture_output=True
    )
    if result.returncode != 0:
        print("Compilation failed:")
        print(result.stderr.decode())
        raise RuntimeError("Compilation failed")
    if not os.path.isfile(exe_path):
        raise FileNotFoundError(f"Compilation did not produce {exe_path}")


def run_and_get_bin(bin_path, env, native=0, extra_args=None):
    if os.path.exists(bin_path):
        os.remove(bin_path)
    env = env.copy()
    env["WINEDEBUG"] = "+loaddll"
    cmd = ["wine", exe_name, os.path.basename(bin_path), str(native)]
    if extra_args:
        cmd.extend(extra_args)
    result = subprocess.run(
        cmd,
        cwd=script_dir,
        env=env,
        capture_output=True
    )
    stderr_output = result.stderr.decode()
    print(stderr_output)
    if result.returncode != 0:
        print("Wine execution failed:")
        print(stderr_output)
        raise RuntimeError("Wine failed")
    if not os.path.isfile(bin_path):
        raise FileNotFoundError(f"Output bin file {bin_path} was not generated.")


def get_native_results():
    wine_env = os.environ.copy()
    wine_env.pop("WINEDLLOVERRIDES", None)
    if not os.path.isfile(bin_path_native):
        print("Running native Wine to capture asin results...")
        run_and_get_bin(bin_path_native, wine_env, native=0)


def get_timing_results():
    wine_env = os.environ.copy()
    wine_env.pop("WINEDLLOVERRIDES", None)
    if not os.path.isfile(bin_path_timing):
        print("Running timing analysis...")
        run_and_get_bin(bin_path_timing, wine_env, native=0, extra_args=["--timing"])


def floats_exact_equal_batch(arr1, arr2):
    """Exact bit-level comparison of float64 arrays, handling NaN and Inf."""
    nan_both = np.isnan(arr1) & np.isnan(arr2)
    inf_both = np.isinf(arr1) & np.isinf(arr2) & (np.sign(arr1) == np.sign(arr2))
    exact_match = (arr1 == arr2)
    return nan_both | inf_both | exact_match


def compare_python_vs_native(native_path):
    """Compare Python asin implementation against native Windows results.

    Loads binary data captured from the native Windows UCRT, runs the Python
    implementation on the same inputs, and reports bit-level differences.
    """
    data = np.fromfile(native_path, dtype=np.float64).reshape(-1, 2)
    inputs = data[:, 0]
    native_outputs = data[:, 1]

    total = len(inputs)
    print(f"Loaded {total} test points from {native_path}")

    # Run Python implementation on all inputs
    py_outputs = np.array([py_asin(x) for x in inputs])

    # Exact bit-level comparison
    native_bits = native_outputs.view(np.uint64)
    py_bits = py_outputs.view(np.uint64)

    nan_both = np.isnan(native_outputs) & np.isnan(py_outputs)
    exact_match = (native_bits == py_bits) | nan_both

    diffs = np.where(~exact_match)[0]

    print(f"Exact matches: {np.sum(exact_match)} / {total}")
    print(f"Differences: {len(diffs)}")

    if len(diffs) > 0:
        print(f"\nFirst {min(20, len(diffs))} differences:")
        for i in diffs[:20]:
            ulp_diff = abs(int(native_bits[i]) - int(py_bits[i]))
            print(f"  x={inputs[i]:.17e}  native={native_outputs[i]:.17e}  "
                  f"python={py_outputs[i]:.17e}  ULP_diff={ulp_diff}")

        # Analyze which ranges have the most differences
        tiny_mask = np.abs(inputs[diffs]) < 2**-26
        low_mask = (np.abs(inputs[diffs]) >= 2**-26) & (np.abs(inputs[diffs]) < 0.5)
        mid_mask = (np.abs(inputs[diffs]) >= 0.5) & (np.abs(inputs[diffs]) < 0.975)
        high_mask = np.abs(inputs[diffs]) >= 0.975

        print(f"\nDifferences by range:")
        print(f"  |x| < 2^-26 (identity):     {np.sum(tiny_mask)}")
        print(f"  2^-26 <= |x| < 0.5 (poly):  {np.sum(low_mask)}")
        print(f"  0.5 <= |x| < 0.975 (sqrt):  {np.sum(mid_mask)}")
        print(f"  0.975 <= |x| < 1.0 (direct): {np.sum(high_mask)}")

    return len(diffs)


def analyze_timing(timing_path):
    """Analyze timing data to detect implementation range boundaries.

    Large discontinuities in execution time indicate different code paths,
    which correspond to range boundaries in the implementation.
    """
    data = np.fromfile(timing_path, dtype=np.float64).reshape(-1, 2)
    inputs = data[:, 0]
    times = data[:, 1]

    # Compute timing changes between adjacent inputs
    dt = np.diff(times)
    median_dt = np.median(np.abs(dt))

    # Find large discontinuities (>5x the median change)
    threshold = median_dt * 5
    boundaries = np.where(np.abs(dt) > threshold)[0]

    print(f"\n=== Timing Analysis ===")
    print(f"Tested {len(inputs)} input values from {inputs[0]:.6f} to {inputs[-1]:.6f}")
    print(f"Median timing delta: {median_dt:.6e}s")
    print(f"Detected {len(boundaries)} potential range boundaries:")
    for idx in boundaries:
        print(f"  x ≈ {inputs[idx]:.17e}  (timing jump: {dt[idx]:+.6e}s)")

    return inputs, times


if __name__ == "__main__":
    if not os.path.isfile(c_path):
        raise FileNotFoundError(f"Could not find {c_path}")

    # 0. Always recompile
    compile_asin_test()

    # 1. Generate native results if needed
    get_native_results()

    # 2. Compare Python implementation against native
    print("\n=== Comparison: Python vs Native ===")
    compare_python_vs_native(bin_path_native)

    # 3. Run timing analysis if requested
    if "--timing" in sys.argv:
        get_timing_results()
        analyze_timing(bin_path_timing)
