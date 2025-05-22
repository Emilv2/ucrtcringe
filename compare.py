import subprocess
import numpy as np
import os
import struct
import sys
import math

c_file = "math_test.c"
exe_name = "math_test.exe"
dll_name = "ucrtbase.dll"
bin_native = "math_test_native.bin"
bin_custom = "math_test_custom.bin"
script_dir = os.path.dirname(os.path.abspath(__file__))
c_path = os.path.join(script_dir, c_file)
exe_path = os.path.join(script_dir, exe_name)
dll_path = os.path.join(script_dir, dll_name)
bin_path_native = os.path.join(script_dir, bin_native)
bin_path_custom = os.path.join(script_dir, bin_custom)

RECORD_SIZE = 16  # 4 floats per record: input, sqrt, asin, acos
BATCH_SIZE = 1024 * 1024  # Number of records per batch

def assert_custom_dll_loaded(stderr_output, exe_path):
    exe_dir = os.path.dirname(os.path.abspath(exe_path))
    expected_path = os.path.join(exe_dir, 'ucrtbase.dll').replace("\\", "/").lower()
    loaded = False
    for line in stderr_output.lower().splitlines():
        if "ucrtbase.dll" in line and expected_path in line:
            loaded = True
            break
    if not loaded:
        print("ERROR: ucrtbase.dll was NOT loaded from the exe directory. Check DLL placement and overrides!")
        sys.exit(1)

def compile_math_test():
    print("Compiling math_test.exe from math_test.c...")
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

def run_and_get_bin(bin_path, env, check_dll=False, native=0):
    if os.path.exists(bin_path):
        os.remove(bin_path)
    # Enable DLL load logging for verification
    env = env.copy()
    env["WINEDEBUG"] = "+loaddll"
    result = subprocess.run(
        ["wine", exe_name, os.path.basename(bin_path), str(native)],
        cwd=script_dir,
        env=env,
        capture_output=True
    )
    stderr_output = result.stderr.decode()
    print(stderr_output)  # Always print for visibility
    if check_dll:
        assert_custom_dll_loaded(stderr_output, exe_path)
    if result.returncode != 0:
        print("Wine execution failed:")
        print(stderr_output)
        raise RuntimeError("Wine failed")
    if not os.path.isfile(bin_path):
        raise FileNotFoundError(f"Output bin file {bin_path} was not generated.")

def get_native_results():
    wine_env_native = os.environ.copy()
    wine_env_native.pop("WINEDLLOVERRIDES", None)
    if not os.path.isfile(bin_path_native):
        print("Running native Wine (builtin ucrtbase)...")
        run_and_get_bin(bin_path_native, wine_env_native, native=0)

def get_custom_results():
    wine_env_custom = os.environ.copy()
    wine_env_custom["WINEDLLOVERRIDES"] = "ucrtbase=n,b"
    if not os.path.isfile(bin_path_custom):
        print("Running Wine with custom ucrtbase.dll...")
        run_and_get_bin(bin_path_custom, wine_env_custom, check_dll=False, native=1)

def floats_exact_equal_batch(arr1, arr2):
    a64 = arr1.astype(np.float64)
    b64 = arr2.astype(np.float64)
    nan_both = np.isnan(a64) & np.isnan(b64)
    inf_both = np.isinf(a64) & np.isinf(b64) & (np.sign(a64) == np.sign(b64))
    exact_match = (a64 == b64)
    return nan_both | inf_both | exact_match

def diff_outputs(native_path, custom_path, diff_out_path, max_to_keep=None, write_output=True):
    total_count = 0
    diff_count = 0

    filesize = os.path.getsize(native_path)
    total_records = filesize // RECORD_SIZE

    with open(native_path, "rb") as f_native, open(custom_path, "rb") as f_custom, open(diff_out_path, "wb") as f_diff:
        while True:
            native_bytes = f_native.read(BATCH_SIZE * RECORD_SIZE)
            custom_bytes = f_custom.read(BATCH_SIZE * RECORD_SIZE)
            if not native_bytes or not custom_bytes:
                break
            actual_records = min(len(native_bytes), len(custom_bytes)) // RECORD_SIZE

            native_arr = np.frombuffer(native_bytes, dtype=np.float32).reshape(-1, 4)
            custom_arr = np.frombuffer(custom_bytes, dtype=np.float32).reshape(-1, 4)

            # Consistency check: input bit patterns must match
            if not np.all(native_arr[:, 0].view(np.uint32) == custom_arr[:, 0].view(np.uint32)):
                mismatch_idx = np.where(native_arr[:, 0].view(np.uint32) != custom_arr[:, 0].view(np.uint32))[0][0]
                raise ValueError(f"Input mismatch at batch record {total_count + mismatch_idx}")

            # Compare all three outputs: sqrt, asin, acos
            eq_sqrt = floats_exact_equal_batch(native_arr[:, 1], custom_arr[:, 1])
            eq_asin = floats_exact_equal_batch(native_arr[:, 2], custom_arr[:, 2])
            #eq_asin = floats_exact_equal_batch(asinf(native_arr[:, 0]), custom_arr[:, 2])
            #eq_acos = floats_exact_equal_batch(native_arr[:, 3], custom_arr[:, 3])
            diff_mask = ~(eq_asin)
            diff_idx = np.where(diff_mask)[0]

            for idx in diff_idx:
                # Write input, native_sqrt, custom_sqrt, native_asin, custom_asin, native_acos, custom_acos
                if write_output:
                    f_diff.write(struct.pack(
                        "f"      # input
                        "f" "f"  # sqrt_native, sqrt_custom
                        "f" "f"  # asin_native, asin_custom
                        "f" "f", # acos_native, acos_custom
                        native_arr[idx, 0],
                        native_arr[idx, 1], custom_arr[idx, 1],
                        native_arr[idx, 2], custom_arr[idx, 2],
                        native_arr[idx, 3], custom_arr[idx, 3],
                    ))
                diff_count += 1
                if max_to_keep is not None and diff_count >= max_to_keep:
                    print(f"Stopped at max_to_keep={max_to_keep} diffs.")
                    print(f"Processed {total_count + idx + 1} records so far.")
                    return

            total_count += actual_records
            print(f"Processed {total_count/1e6:.1f}M records, {diff_count} diffs so far...")

    print(f"Found {diff_count} differing entries out of {total_count} total.")

def print_diff_samples(diff_path, max_samples=10):
    with open(diff_path, "rb") as f:
        max_input = -1
        min_input = 10
        count = 0
        while True:
            data = f.read(28)  # 7 floats: input, sqrt_native, sqrt_custom, asin_native, asin_custom, acos_native, acos_custom
            if not data:
                break
            inp, sqrt_nat, sqrt_cust, asin_nat, asin_cust, acos_nat, acos_cust = struct.unpack("fffffff", data)
            # print(f"input={inp}")
            # print(f"  sqrt: native={sqrt_nat} | custom={sqrt_cust}")
            # print(f"  asin: native={asin_nat} | custom={asin_cust}")
            # print(f"  acos: native={acos_nat} | custom={acos_cust}")
            # print("-" * 60)
            # count += 1
            # if count >= max_samples:
            #     break
        # if count == 0:
        #     print("No differences found.")
            if abs(inp) > abs(max_input):
                max_input = inp
            if abs(inp) < abs(min_input):
                min_input = inp
        print(f"{max_input=}")
        print(f"{min_input=}")

def asinf_R(z):
    # coefficients for R(x^2)
   #p1 = 1.66666672e-01
    p1 = 1.6666666e-01
    p2 = -5.11644611e-02
    p3 = -1.21124933e-02
    p4 = -3.58742251e-03
    q1 = -7.56982703e-01

    p = z * (p1 + z * (p2 + z * (p3 + z * p4)))
    q = 1.0 + z * q1
    return p / q

def asinf(x):
    pio2 = 1.570796326794896558e+00
    pio4_hi = 0.785398125648
    pio2_lo = 7.54978941586e-08

    x = np.float32(x)  # ensure float32 for bit pattern compatibility
    absx = np.abs(x)
    ix = np.frombuffer(x.tobytes(), dtype=np.uint32)[()] & 0x7fffffff

    # Handle |x| >= 1
    if ix >= 0x3f800000:
        if ix == 0x3f800000:
            return float(x * pio2 + 7.5231638453e-37)
        if np.isnan(x):
            return float(x)
        raise ValueError("math domain error for asinf({})".format(x))

    # |x| < 0.5
    if ix < 0x3f000000:
        if ix < 0x39800000 and ix >= 0x00800000:
            return float(x)
        return float(x + x * asinf_R(x * x))

    # 1 > |x| >= 0.5
    z = (1 - absx) * 0.5
    s = np.sqrt(z).astype(np.float32)
    # f+c = sqrt(z)
    # Simulate truncation of lower 16 bits (as in C: f = s with 16 LSB zeroed)
    s_bits = np.frombuffer(s.tobytes(), dtype=np.uint32)[()]
    f_bits = s_bits & 0xffff0000
    f = np.frombuffer(np.uint32(f_bits).tobytes(), dtype=np.float32)[()]
    c = (z - f * f) / (s + f)
    res = pio4_hi - (2 * s * asinf_R(z) - (pio2_lo - 2 * c) - (pio4_hi - 2 * f))
    if x < 0:
        return np.float32(-res)
    return np.float32(res)

def asinf_arr(x):
    return np.stack(np.vectorize(asinf)(test), axis=2)




if __name__ == "__main__":
    if not os.path.isfile(c_path):
        raise FileNotFoundError(f"Could not find {c_path}")
    # if not os.path.isfile(dll_path):
    #     raise FileNotFoundError(f"Could not find {dll_path}")

    # 0. Always recompile math_test.exe
    compile_math_test()

    # # 1. Generate native and custom bin files if needed
    get_native_results()
    get_custom_results()

    # # 2. Compare piecewise in large batches, only keep diffs
    #diff_bin_path = os.path.join(script_dir, "math_test_diff.bin")
    #print("Comparing outputs and saving only differences...")
    #diff_outputs(bin_path_native, bin_path_custom, diff_bin_path, write_output=True)

    # # 3. Print a sample of diffs
    #print("\nSample of differing results:")
    #print_diff_samples(diff_bin_path, max_samples=10)
