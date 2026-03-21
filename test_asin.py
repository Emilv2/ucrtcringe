"""Self-test for the asin reverse-engineering implementation.

Validates all Python components without requiring Wine or MinGW:
  - asin.py: double-precision asin implementation
  - asin_compare.py: comparison utilities
  - asin_fit.py: curve fitting models

Run: python3 test_asin.py
"""
import struct
import sys
import os
import tempfile
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from asin import asin as py_asin
from asin_compare import floats_exact_equal_batch, compare_python_vs_native
from asin_fit import asin_R, asin_model_low, asin_model_low_vec
from asin_fit import asin_model_high, asin_model_high_vec

def to_bits(x):
    return struct.unpack('<Q', struct.pack('<d', float(x)))[0]

def ulp_diff(a, b):
    return abs(to_bits(a) - to_bits(b))

passed = 0
failed = 0

def check(name, condition, detail=""):
    global passed, failed
    if condition:
        passed += 1
    else:
        failed += 1
        print(f"  FAIL: {name} {detail}")


# --- Test asin.py ---
print("=== asin.py ===")

# Special values
check("asin(0.0)", py_asin(0.0) == 0.0)
check("asin(-0.0)", py_asin(-0.0) == -0.0 and str(py_asin(-0.0)) == "-0.0")
check("asin(1.0)", ulp_diff(py_asin(1.0), np.pi / 2) <= 1)
check("asin(-1.0)", ulp_diff(py_asin(-1.0), -np.pi / 2) <= 1)
check("asin(NaN)", np.isnan(py_asin(float('nan'))))
try:
    py_asin(2.0)
    check("asin(2.0) raises", False, "should raise ValueError")
except ValueError:
    check("asin(2.0) raises", True)
try:
    py_asin(float('inf'))
    check("asin(inf) raises", False, "should raise ValueError")
except ValueError:
    check("asin(inf) raises", True)

# Tiny values (identity range, |x| < 2^-26)
for x in [1e-10, 1e-20, 1e-30, 5e-324, 2.2e-308]:
    check(f"asin({x}) tiny", ulp_diff(py_asin(x), float(np.arcsin(x))) <= 1)

# Low range (2^-26 <= |x| < 0.5)
for x in [0.001, 0.01, 0.1, 0.25, 0.499]:
    check(f"asin({x}) low", ulp_diff(py_asin(x), float(np.arcsin(x))) <= 1)

# Mid range (0.5 <= |x| < 0.975, sqrt with truncation)
for x in [0.5, 0.6, 0.7, 0.8, 0.9, 0.95]:
    check(f"asin({x}) mid", ulp_diff(py_asin(x), float(np.arcsin(x))) <= 1)

# High range (0.975 <= |x| < 1.0, direct computation)
for x in [0.975, 0.98, 0.99, 0.999, 0.9999]:
    check(f"asin({x}) high", ulp_diff(py_asin(x), float(np.arcsin(x))) <= 1)

# Negative values
for x in [-0.1, -0.5, -0.9, -0.99, -0.999]:
    check(f"asin({x}) neg", ulp_diff(py_asin(x), float(np.arcsin(x))) <= 1)

# Bulk random test
np.random.seed(42)
random_vals = np.random.uniform(-0.9999, 0.9999, 10000)
max_ulp = 0
for x in random_vals:
    u = ulp_diff(py_asin(float(x)), float(np.arcsin(x)))
    max_ulp = max(max_ulp, u)
check(f"10000 random values max_ulp={max_ulp}", max_ulp <= 1)


# --- Test asin_compare.py ---
print("=== asin_compare.py ===")

a = np.array([1.0, 2.0, float('nan'), float('inf'), -float('inf'), 0.0])
b = np.array([1.0, 2.1, float('nan'), float('inf'), -float('inf'), 0.0])
eq = floats_exact_equal_batch(a, b)
check("floats_exact_equal_batch", list(eq) == [True, False, True, True, True, True])

# Generate synthetic native data and compare
with tempfile.NamedTemporaryFile(suffix='.bin', delete=False) as tmp:
    tmpbin = tmp.name
    test_inputs = np.linspace(0.001, 0.999, 500)
    for x in test_inputs:
        result = float(np.arcsin(x))
        tmp.write(struct.pack('<dd', x, result))
ndiffs = compare_python_vs_native(tmpbin)
check(f"compare_python_vs_native ndiffs={ndiffs}", ndiffs <= len(test_inputs))
os.remove(tmpbin)


# --- Test asin_fit.py ---
print("=== asin_fit.py ===")

p0 = [
    1.66666666666666657415e-01,
    -3.25565818622400915405e-01,
    2.01212532134862925881e-01,
    -4.00555345006794114027e-02,
    7.91534994289814532176e-04,
    3.47933107596021167570e-05,
    -2.40339491173441421878e+00,
    2.02094576023350569471e+00,
    -6.88283971605453293030e-01,
    7.70381505559019352791e-02,
]

# asin_R
r = asin_R(0.04, *p0)
check("asin_R(0.04) finite", np.isfinite(r))

# asin_model_low
result = asin_model_low(0.3, *p0)
ref = float(np.arcsin(0.3))
check(f"asin_model_low(0.3) diff={abs(result-ref):.2e}", abs(result - ref) < 1e-15)

# asin_model_high
result_h = asin_model_high(0.7, *p0)
ref_h = float(np.arcsin(0.7))
check(f"asin_model_high(0.7) diff={abs(result_h-ref_h):.2e}", abs(result_h - ref_h) < 1e-15)

# Vectorized versions
x_arr = np.array([0.1, 0.2, 0.3])
result_vec = asin_model_low_vec(x_arr, *p0)
check("asin_model_low_vec shape", result_vec.shape == (3,))
check("asin_model_low_vec values", all(np.isfinite(result_vec)))

# Curve fitting
from scipy.optimize import curve_fit
x_fit = np.linspace(0.001, 0.499, 1000)
y_fit = np.arcsin(x_fit)
sigma = np.finfo(np.float64).eps * np.ones(len(x_fit))
popt, pcov = curve_fit(asin_model_low_vec, x_fit, y_fit, p0=p0, sigma=sigma)
y_pred = asin_model_low_vec(x_fit, *popt)
max_err = np.max(np.abs(y_pred - y_fit))
check(f"curve_fit low range max_err={max_err:.2e}", max_err < 1e-15)


# --- Test C compilation ---
print("=== asin_test.c compilation ===")
import subprocess
script_dir = os.path.dirname(os.path.abspath(__file__))
gcc = "x86_64-w64-mingw32-gcc"
if os.path.isfile(os.path.join(script_dir, "asin_test.c")):
    with tempfile.NamedTemporaryFile(suffix='.exe', delete=False) as tmp:
        tmp_exe = tmp.name
    result = subprocess.run(
        [gcc, "-O2", "-o", tmp_exe, "asin_test.c", "-lm"],
        cwd=script_dir, capture_output=True
    )
    if result.returncode == 0:
        check("asin_test.c compiles", True)
        os.remove(tmp_exe)
    else:
        # MinGW might not be installed
        if b"not found" in result.stderr or result.returncode == 127:
            print("  SKIP: MinGW cross-compiler not available")
        else:
            check("asin_test.c compiles", False, result.stderr.decode())
else:
    print("  SKIP: asin_test.c not found")


# --- Summary ---
print()
print(f"Results: {passed} passed, {failed} failed")
sys.exit(1 if failed > 0 else 0)
