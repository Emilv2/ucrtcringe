import numpy as np
import struct

def asin_R(z):
    """Rational approximation R(z) for asin.

    Computes P(z)/Q(z) where P is degree 5 and Q is degree 4.

    These are initial coefficients based on the well-known Sun/FreeBSD libm
    implementation. They serve as starting guesses and should be refined via
    asin_fit.py using captured native Windows UCRT asin output data.

    The polynomial structure (degree of P and Q) may also need adjustment
    based on the actual Windows implementation. Try fitting with fewer or
    more terms if the initial fit is poor.
    """
    # Numerator coefficients (degree 5)
    pS0 =  1.66666666666666657415e-01  # 0x3FC5555555555555
    pS1 = -3.25565818622400915405e-01  # 0xBFD4D61203EB6F7D
    pS2 =  2.01212532134862925881e-01  # 0x3FC9C1550E884455
    pS3 = -4.00555345006794114027e-02  # 0xBFA48228B5688F3B
    pS4 =  7.91534994289814532176e-04  # 0x3F49EFE07501B288
    pS5 =  3.47933107596021167570e-05  # 0x3F023DE10DFDF709

    # Denominator coefficients (degree 4, leading 1)
    qS1 = -2.40339491173441421878e+00  # 0xC0033A271C8A2D4B
    qS2 =  2.02094576023350569471e+00  # 0x40002AE59C598AC8
    qS3 = -6.88283971605453293030e-01  # 0xBFE6066C1B8D0159
    qS4 =  7.70381505559019352791e-02  # 0x3FB3B8C5B12E9282

    p = z * (pS0 + z * (pS1 + z * (pS2 + z * (pS3 + z * (pS4 + z * pS5)))))
    q = 1.0 + z * (qS1 + z * (qS2 + z * (qS3 + z * qS4)))
    return p / q

def asin(x):
    """Double-precision arcsine implementation.

    Reverse-engineering approach for the Windows UCRT asin function:

    1. The implementation follows a standard range-reduction approach:
       - |x| >= 1: special cases (exact ±1 returns ±pi/2, NaN, domain error)
       - |x| < 0.5: direct polynomial approximation asin(x) = x + x*R(x^2)
       - 0.5 <= |x| < 1: range reduction via sqrt using the identity
         asin(x) = pi/2 - 2*asin(sqrt((1-x)/2))

    2. Within the upper range (0.5 to 1.0), there is typically a sub-split
       around |x| ~ 0.975 which changes how the sqrt term is handled
       (with or without lower-word truncation for precision).

    3. Constants and polynomial coefficients need to be refined via
       asin_fit.py using captured native Windows UCRT asin output data.
       The initial values here are starting guesses.

    4. Key differences from asinf (float32 version):
       - Uses float64 and uint64 for bit manipulation
       - More polynomial terms (P5/Q4 vs P4/Q1)
       - Truncates lower 32 bits of sqrt (vs 16 bits for float)
       - Different threshold constants
    """
    pio2_hi =  1.57079632679489655800e+00  # 0x3FF921FB54442D18
    pio2_lo =  6.12323399573676603587e-17  # 0x3C91A62633145C07
    pio4_hi =  7.85398163397448278999e-01  # 0x3FE921FB54442D18

    x = np.float64(x)
    absx = np.abs(x)
    ix = struct.unpack('<Q', x.tobytes())[0] & 0x7fffffffffffffff
    hx = ix >> 32  # upper 32 bits of |x|

    # Handle |x| >= 1
    if hx >= 0x3ff00000:
        # Check for exact ±1
        if ix == 0x3ff0000000000000:
            return float(x * pio2_hi + x * pio2_lo)
        if np.isnan(x):
            return float(x)
        raise ValueError("math domain error for asin({})".format(x))

    # |x| < 0.5
    if hx < 0x3fe00000:
        # For very small |x|, asin(x) ≈ x to within rounding
        # Threshold: |x| < 2^-26 ensures x^3/6 < 0.5 ULP(x)
        if hx < 0x3e500000:
            if hx >= 0x00100000:  # normalized
                return float(x)
        return float(x + x * asin_R(x * x))

    # 0.5 <= |x| < 1.0
    z = (1.0 - absx) * 0.5
    s = np.sqrt(z)

    if hx >= 0x3fef3333:  # |x| >= ~0.975
        # Direct computation (no truncation needed near x=1)
        res = pio2_hi - (2.0 * (s + s * asin_R(z)) - pio2_lo)
    else:
        # Truncate lower 32 bits of s for extended precision computation
        # This mirrors the C SET_LOW_WORD(f, 0) idiom
        s_bits = struct.unpack('<Q', np.float64(s).tobytes())[0]
        f_bits = s_bits & 0xffffffff00000000
        f = struct.unpack('<d', struct.pack('<Q', f_bits))[0]
        c = (z - f * f) / (s + f)
        p = 2.0 * s * asin_R(z) - (pio2_lo - 2.0 * c)
        q = pio4_hi - 2.0 * f
        res = pio4_hi - (p - q)

    if x < 0:
        return float(-res)
    return float(res)
