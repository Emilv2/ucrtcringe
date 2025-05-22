import numpy as np

def asinf_R(z):
    # coefficients for R(x^2)
   #p1 = 1.66666672e-01
    # p1 = 1.6666666e-01
    # p2 = -5.11644611e-02
    # p3 = -1.21124933e-02
    # p4 = -3.58742251e-03
    # q1 = -7.56982703e-01
    p1 = 1.6666666e-01
    p2 = -5.11644600e-02
    p3 = -1.21122106e-02
    p4 = -3.58744047e-03
    p5 = -2.36896778e-05
    q1 = -7.56982663e-01
    # [-5.11644600e-02 -1.21122106e-02 -3.58744047e-03 -2.36896778e-05 -7.56982663e-01 -2.42458589e-10

    p = z * (p1 + z * (p2 + z * (p3 + z * (p4 + z * p5))))
    q = 1.0 + z * q1
    return p / q

def asinf(x):
    # pio2 = 1.570796326794896558e+00
    # pio4_hi = 0.785398125648
    # pio2_lo = 7.54978941586e-08
    pio2 = 1.57079633e+00
    pio4_hi = 7.85398134e-01
    pio2_lo = 7.54978942e-08

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
        return float(-res)
    return float(res)
