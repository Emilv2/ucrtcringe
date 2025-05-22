import numpy as np
from scipy.optimize import curve_fit

import os

# Define the function to fit

def asinf_R(z, p1, p2, p3, p4, p5, q1):
    # coefficients for R(x^2)
   #p1 = 1.66666672e-01
    #p1 = 1.6666666e-01
    # p2 = -5.11644611e-02
    # p3 = -1.21124933e-02
    # p4 = -3.58742251e-03
    # q1 = -7.56982703e-01

    p = z * (p1 + z * (p2 + z * (p3 + z * (p4 + z * p5))))
    q = 1.0 + z * q1
    return p / q

def asinf(x, p1, p2, p3, p4, p5, q1):
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
        return float(x + x * asinf_R(x * x, p1, p2, p3, p4, p5, q1))

    # 1 > |x| >= 0.5
    z = (1 - absx) * 0.5
    s = np.sqrt(z).astype(np.float32)
    # f+c = sqrt(z)
    # Simulate truncation of lower 16 bits (as in C: f = s with 16 LSB zeroed)
    s_bits = np.frombuffer(s.tobytes(), dtype=np.uint32)[()]
    f_bits = s_bits & 0xffff0000
    f = np.frombuffer(np.uint32(f_bits).tobytes(), dtype=np.float32)[()]
    c = (z - f * f) / (s + f)
    res = pio4_hi - (2 * s * asinf_R(z, p1, p2, p3, p4, p5, q1) - (pio2_lo - 2 * c) - (pio4_hi - 2 * f))
    if x < 0:
        return np.float32(-res)
    return np.float32(res)

asinf_arr = np.vectorize(asinf)


def load_full_diff_bin(diff_path):
    """
    Loads diff records of structure:
    float32 input
    float32 sqrt_native, sqrt_custom
    float32 asin_native, asin_custom
    float32 acos_native, acos_custom
    Returns arrays: input, sqrt_native, sqrt_custom, asin_native, asin_custom, acos_native, acos_custom
    """
    if not os.path.isfile(diff_path):
        raise FileNotFoundError(f"File not found: {diff_path}")
    arr = np.fromfile(diff_path, dtype=np.float32)
    if arr.size % 2 != 0:
        raise ValueError("Corrupt diff file: not a whole number of records.")
    arr = arr.reshape(-1, 7)
    input_vals    = arr[:, 0]
    sqrt_native   = arr[:, 1]
    sqrt_custom   = arr[:, 2]
    asin_native   = arr[:, 3]
    asin_custom   = arr[:, 4]
    acos_native   = arr[:, 5]
    acos_custom   = arr[:, 6]
    return input_vals, sqrt_native, sqrt_custom, asin_native, asin_custom, acos_native, acos_custom

# Example usage:
if __name__ == "__main__":
    # diff_path = "math_test_diff.bin"
    # (input_vals, sqrt_native, sqrt_custom,
    #  asin_native, asin_custom,
    #  acos_native, acos_custom) = load_full_diff_bin(diff_path)
    # print(f"Loaded {len(input_vals)} records.")
    # print("First 3 records:")
    # for i in range(min(3, len(input_vals))):
    #     print(f"input={input_vals[i]:.8g}, sqrt_native={sqrt_native[i]:.8g}, sqrt_custom={sqrt_custom[i]:.8g}, "
    #           f"asin_native={asin_native[i]:.8g}, asin_custom={asin_custom[i]:.8g}, "
    #           f"acos_native={acos_native[i]:.8g}, acos_custom={acos_custom[i]:.8g}")

    native_path = "math_test_custom.bin"
    custom_path = "math_test_native.bin"

    native_arr = np.fromfile(native_path, dtype=np.float32)
    if native_arr.size % 2 != 0:
        raise ValueError("Corrupt diff file: not a whole number of records.")

    native_arr = native_arr.reshape(-1, 2)
    input_vals    = native_arr[:, 0]
    asin_native   = native_arr[:, 1]

    custom_arr = np.fromfile(custom_path, dtype=np.float32)
    if custom_arr.size % 2 != 0:
        raise ValueError("Corrupt diff file: not a whole number of records.")

    custom_arr = custom_arr.reshape(-1, 2)
    input_vals    = custom_arr[:, 0]
    asin_custom   = custom_arr[:, 1]


    # Example: xdata and ydata are your data arrays
    # xdata = np.array([...])
    # ydata = np.array([...])

    # Initial parameter guess (adjust as needed)
    asin_native = np.float64(asin_native)
    asin_custom = np.float64(asin_custom)
    p0 = [1.66666672e-01,
            -5.11644611e-02,
          -1.21124933e-02,
          -3.58742251e-03,
          0,
#          1.0,
          -7.56982703e-01,
#    1.570796326794896558e+00,
#    0.785398125648,
#    7.54978941586e-08
          ]
    #[-5.11644610e-02 -1.21124933e-02 -3.59184653e-03  5.65420179e-09 -7.56982702e-01  1.57079633e+00  7.85398134e-01  7.54978942e-08]

    sigma = np.finfo(np.float64).eps * np.ones(len(asin_custom))

    # Fit the function to data
    popt, pcov = curve_fit(asinf_arr, input_vals, asin_custom, p0=p0, sigma=sigma)

    print("Fitted parameters:", popt)
    print("pcov:", pcov)
