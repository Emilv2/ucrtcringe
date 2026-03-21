# ucrtcringe

Reverse engineering Windows UCRT math functions (starting with `asin`/`asinf`)
to produce compatible implementations for Wine.

Licensed under LGPL-2.1 (see LICENSE).

## Current Approach

The current pipeline uses **black-box curve fitting**:

1. **Capture native output** — `asin_test.c` / `math_test.c` are cross-compiled
   with MinGW and run under Wine to call the real UCRT functions. Input/output
   pairs are written to `.bin` files covering all interesting ranges.

2. **Fit polynomial coefficients** — `asin_fit.py` / `fit.py` use
   `scipy.optimize.curve_fit` to fit rational‐polynomial coefficients
   (P(z)/Q(z)) to the captured data, starting from Sun/FreeBSD libm initial
   guesses.

3. **Implement in Python** — `asin.py` / `asinf.py` implement the full
   function (range reduction, bit manipulation, polynomial evaluation) using
   the fitted coefficients.

4. **Verify** — `asin_compare.py` / `compare.py` compare the Python
   implementation against native output at the bit level (ULP differences).
   `test_asin.py` provides a self-test suite.

### Limitations

- `curve_fit` minimises least-squares error, but UCRT coefficients are
  typically chosen for **minimax** (minimise the maximum error) or by other
  criteria. This means `curve_fit` may converge to different coefficients that
  give similar accuracy but are not bit-for-bit identical.
- The polynomial structure (degree of P and Q) is assumed from FreeBSD libm
  and may differ in the actual Windows implementation.
- Range thresholds (e.g. the `0.975` sub-split) are educated guesses; the
  real boundaries may be slightly different.

## Alternative Approaches

If curve fitting does not produce bit-exact results, the following
alternatives can be tried (roughly ordered from easiest to hardest):

### 1. Differential testing against known open-source implementations

Compare UCRT output against **every known open-source `asin`**:

- glibc (`sysdeps/ieee754/dbl-64/e_asin.c`)
- musl (`src/math/asin.c`)
- FreeBSD libm / OpenBSD libm
- CORE-MATH (`src/binary64/asin/`)
- Arm Optimized Routines (`math/asin.c`)
- LLVM libc (`src/math/generic/asin.cpp`)

If any of these match UCRT bit-for-bit, we are done — use those coefficients
directly. This is the fastest path because it requires no fitting at all.
Run the comparison with:

```python
# pseudo-code
for impl in [glibc_asin, musl_asin, freebsd_asin, ...]:
    diffs = count_ulp_diffs(ucrt_data, impl)
    print(f"{impl.__name__}: {diffs} differences")
```

### 2. Direct disassembly / decompilation

Use a disassembler (Ghidra, IDA, Binary Ninja) to read the `asin` code
directly from `ucrtbase.dll`:

1. Open `ucrtbase.dll` (from a Windows installation or Wine) in Ghidra.
2. Navigate to the `asin` export.
3. Read the polynomial coefficients from the `.rdata` / constant pool.
4. Reconstruct the control flow (range checks, special cases).

**Advantages**: gives exact coefficients and exact range thresholds.

**Disadvantages**: resulting code may be considered a derived work, which
conflicts with LGPL-2.1 clean-room goals. Using disassembly as a
**verification tool** (confirming polynomial degree and range structure, but
not copying code) can mitigate this.

### 3. Remez exchange algorithm

The standard technique for deriving math library polynomials is the
**Remez algorithm**, which finds the minimax (Chebyshev) approximation —
the polynomial that minimises the **maximum** error, not the least-squares
error.

Tools:
- [Sollya](https://www.sollya.org/) — `remez()` command
- [lolremez](https://github.com/samhocevar/lolremez)
- [CORE-MATH tools](https://core-math.gitlabpages.inria.fr/)

```
# Sollya example: fit P5/Q4 for asin on [0, 0.25]
> P = remez(asin(sqrt(x))/sqrt(x) - 1, 5, [0;0.25], 1/sqrt(x));
```

If UCRT used a Remez-derived polynomial (likely), re-running Remez with the
same degree and interval should reproduce the exact same coefficients.

### 4. Coefficient extraction via binary search

Given that each coefficient is a `double` (64-bit), its exact value can be
found by binary search:

1. Fix all other coefficients at their best-known values.
2. Binary-search through possible IEEE 754 bit patterns for one coefficient.
3. Pick the value that minimises the max ULP difference against native data.
4. Repeat for each coefficient, cycling until convergence.

This is slow but guaranteed to find the exact representable double that the
native implementation uses, assuming the polynomial structure is correct.

### 5. Integer relation detection (LLL / PSLQ)

Lattice basis reduction algorithms like **LLL** or **PSLQ** can discover
exact algebraic relationships between floating-point values:

```python
from mpmath import pslq
# Find if the coefficient is a rational combination of known constants
pslq([coefficient, 1, pi, pi**2, log(2)])
```

This can identify coefficients that are exact rational numbers or simple
expressions involving π, ln2, etc., which narrows down the search space.

### 6. Exhaustive table extraction

Some math library implementations use **lookup tables** in addition to (or
instead of) polynomials. If timing analysis (`--timing` mode) reveals
unexpected patterns, the implementation may be table-driven.

To extract tables:
- Sample the function at table boundaries (typically powers of 2 or
  uniformly spaced in a reduced argument).
- Compare against known table-driven implementations (e.g. glibc's
  `asin` uses a degree-12 polynomial, no tables; some `exp` implementations
  use 256-entry tables).

### 7. Iterative coefficient refinement (hill climbing)

Instead of `curve_fit` (which may get stuck in a local minimum), use
a discrete search:

1. Start from the best-fit coefficients.
2. For each coefficient, try incrementing/decrementing by 1 ULP.
3. Accept the change if it reduces the total number of bit-level mismatches.
4. Repeat until no single-ULP change improves the match.

This is essentially a coordinate-descent optimisation in the space of
representable doubles, directly targeting bit-exact matching rather than
least-squares error.

## Files

| File | Description |
|------|-------------|
| `asin.py` | Double-precision `asin` implementation (Python) |
| `asinf.py` | Single-precision `asinf` implementation (Python) |
| `asin_test.c` | C test harness for capturing native double `asin` data |
| `math_test.c` | C test harness for capturing native float `asinf` data |
| `asin_compare.py` | Compare Python `asin` against native UCRT output |
| `compare.py` | Compare Python `asinf` against native UCRT output |
| `asin_fit.py` | Curve fitting for double `asin` coefficients |
| `fit.py` | Curve fitting for float `asinf` coefficients |
| `test_asin.py` | Self-test suite (validates Python code without Wine) |

## Quick Start

```bash
# Run the self-tests (no Wine required)
python3 test_asin.py

# Full pipeline (requires Wine + MinGW cross-compiler)
python3 asin_compare.py           # capture native data + compare
python3 asin_fit.py               # fit coefficients to captured data
python3 asin_compare.py --timing  # detect range boundaries via timing
```
