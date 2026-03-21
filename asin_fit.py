import numpy as np
from scipy.optimize import curve_fit

import os


def asin_R(z, pS0, pS1, pS2, pS3, pS4, pS5, qS1, qS2, qS3, qS4):
    """Parameterized rational approximation R(z) for curve fitting.

    P(z)/Q(z) where P is degree 5 and Q is degree 4.
    If the fit shows pS5 ≈ 0, try reducing P to degree 4.
    If the fit is poor, try increasing P to degree 6 or Q to degree 5.
    """
    p = z * (pS0 + z * (pS1 + z * (pS2 + z * (pS3 + z * (pS4 + z * pS5)))))
    q = 1.0 + z * (qS1 + z * (qS2 + z * (qS3 + z * qS4)))
    return p / q


def asin_model_low(x, pS0, pS1, pS2, pS3, pS4, pS5, qS1, qS2, qS3, qS4):
    """Model for |x| < 0.5 range: asin(x) = x + x*R(x^2)"""
    x = np.float64(x)
    return x + x * asin_R(x * x, pS0, pS1, pS2, pS3, pS4, pS5, qS1, qS2, qS3, qS4)

asin_model_low_vec = np.vectorize(asin_model_low)


def asin_model_high(x, pS0, pS1, pS2, pS3, pS4, pS5, qS1, qS2, qS3, qS4):
    """Model for 0.5 <= |x| < 1.0: range reduction via sqrt.

    asin(x) = pi/2 - 2*(s + s*R(z)) where z = (1-x)/2, s = sqrt(z)
    This simplified version is for fitting only (no truncation).
    """
    pio2_hi = 1.57079632679489655800e+00
    pio2_lo = 6.12323399573676603587e-17
    x = np.float64(x)
    z = (1.0 - x) * 0.5
    s = np.sqrt(z)
    return pio2_hi - (2.0 * (s + s * asin_R(z, pS0, pS1, pS2, pS3, pS4, pS5, qS1, qS2, qS3, qS4)) - pio2_lo)

asin_model_high_vec = np.vectorize(asin_model_high)


if __name__ == "__main__":
    native_path = "asin_test_native.bin"

    if not os.path.isfile(native_path):
        print(f"Error: {native_path} not found.")
        print("Run asin_compare.py first to generate native test data.")
        raise SystemExit(1)

    data = np.fromfile(native_path, dtype=np.float64)
    data = data.reshape(-1, 2)
    inputs = data[:, 0]
    outputs = data[:, 1]

    # Initial guesses (FreeBSD/Sun coefficients)
    p0 = [
        1.66666666666666657415e-01,   # pS0
        -3.25565818622400915405e-01,  # pS1
        2.01212532134862925881e-01,   # pS2
        -4.00555345006794114027e-02,  # pS3
        7.91534994289814532176e-04,   # pS4
        3.47933107596021167570e-05,   # pS5
        -2.40339491173441421878e+00,  # qS1
        2.02094576023350569471e+00,   # qS2
        -6.88283971605453293030e-01,  # qS3
        7.70381505559019352791e-02,   # qS4
    ]

    # --- Fit for |x| < 0.5 range ---
    mask_low = (np.abs(inputs) >= 2**-26) & (np.abs(inputs) < 0.5)
    x_low = inputs[mask_low]
    y_low = outputs[mask_low]

    if len(x_low) > 0:
        sigma_low = np.finfo(np.float64).eps * np.ones(len(x_low))
        popt_low, pcov_low = curve_fit(
            asin_model_low_vec, x_low, y_low, p0=p0, sigma=sigma_low
        )

        print("=== Fitted coefficients for |x| < 0.5 range ===")
        names = ['pS0', 'pS1', 'pS2', 'pS3', 'pS4', 'pS5',
                 'qS1', 'qS2', 'qS3', 'qS4']
        for name, val in zip(names, popt_low):
            print(f"  {name} = {val:.20e}")
        print("\nCovariance diagonal (variance of each parameter):")
        for name, var in zip(names, np.diag(pcov_low)):
            print(f"  {name}: {var:.6e}")

        # Check residuals
        y_pred = asin_model_low_vec(x_low, *popt_low)
        max_err = np.max(np.abs(y_pred - y_low))
        print(f"\nMax absolute error: {max_err:.6e}")
        print(f"Max relative error: {np.max(np.abs((y_pred - y_low) / y_low)):.6e}")
    else:
        print("No data in |x| < 0.5 range.")

    # --- Fit for 0.5 <= |x| < 1.0 range ---
    mask_high = (inputs >= 0.5) & (inputs < 1.0)
    x_high = inputs[mask_high]
    y_high = outputs[mask_high]

    if len(x_high) > 0:
        sigma_high = np.finfo(np.float64).eps * np.ones(len(x_high))
        popt_high, pcov_high = curve_fit(
            asin_model_high_vec, x_high, y_high, p0=p0, sigma=sigma_high
        )

        print("\n=== Fitted coefficients for 0.5 <= |x| < 1.0 range ===")
        for name, val in zip(names, popt_high):
            print(f"  {name} = {val:.20e}")
        print("\nCovariance diagonal:")
        for name, var in zip(names, np.diag(pcov_high)):
            print(f"  {name}: {var:.6e}")

        y_pred_high = asin_model_high_vec(x_high, *popt_high)
        max_err_high = np.max(np.abs(y_pred_high - y_high))
        print(f"\nMax absolute error: {max_err_high:.6e}")
        print(f"Max relative error: {np.max(np.abs((y_pred_high - y_high) / y_high)):.6e}")
    else:
        print("No data in 0.5 <= |x| < 1.0 range.")
