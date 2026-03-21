#include <windows.h>
#include <stdio.h>
#include <stdint.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

typedef double (__cdecl *asin_func)(double);

/* Helper to reinterpret uint64_t as double */
double uint64_to_double(uint64_t u) {
    union {
        uint64_t u;
        double d;
    } x;
    x.u = u;
    return x.d;
}

int main(int argc, char *argv[]) {
    FILE *out;
    HMODULE hLib;

    int native = 0;
    int timing_mode = 0;
    const char *filename = "asin_test_out.bin";
    if (argc > 1) {
        filename = argv[1];
    }
    if (argc > 2) {
        char* p;
        errno = 0;
        long arg = strtol(argv[2], &p, 10);
        if (*p != '\0' || errno != 0) {
            return 1;
        }
        if (arg < INT_MIN || arg > INT_MAX) {
            return 1;
        }
        native = arg;
        printf("native: %d\n", native);
    }
    if (argc > 3 && strcmp(argv[3], "--timing") == 0) {
        timing_mode = 1;
    }

    out = fopen(filename, "wb");
    if (!out) {
        perror("fopen");
        return 1;
    }

    if (native == 0) {
        hLib = LoadLibraryA("ucrtbase_builtin.dll");
    } else {
        hLib = LoadLibraryA("ucrtbase.dll");
    }

    if (!hLib) {
        printf("Failed to load ucrtbase.dll\n");
        return 1;
    }

    asin_func my_asin = (asin_func)GetProcAddress(hLib, "asin");
    if (!my_asin) {
        printf("Failed to get address of asin\n");
        FreeLibrary(hLib);
        return 1;
    }

    if (timing_mode) {
        /*
         * Timing mode: measure execution time at evenly spaced inputs.
         * Timing discontinuities reveal range boundaries in the implementation.
         * Uses QueryPerformanceCounter for high-resolution timing.
         */
        LARGE_INTEGER freq, start, end;
        int num_samples = 10000;
        int iterations = 10000;
        QueryPerformanceFrequency(&freq);

        for (int i = 0; i <= num_samples; i++) {
            double input = (double)i / num_samples; /* 0 to 1 */
            volatile double result;
            QueryPerformanceCounter(&start);
            for (int j = 0; j < iterations; j++) {
                result = my_asin(input);
            }
            QueryPerformanceCounter(&end);
            double elapsed = (double)(end.QuadPart - start.QuadPart) / freq.QuadPart;
            fwrite(&input, sizeof(double), 1, out);
            fwrite(&elapsed, sizeof(double), 1, out);
        }
    } else {
        /*
         * Data collection mode: iterate through double-precision inputs
         * by stepping through upper 32 bits (lower 32 bits set to 0).
         * This gives dense, systematic coverage of each range.
         *
         * For doubles we cannot iterate all 2^64 bit patterns.
         * Iterating upper 32 bits gives ~27M points in the main range,
         * which is sufficient for polynomial coefficient fitting.
         */

        /* Range 1: tiny values - sparse sampling
         * 0x00100000 (smallest normal) to 0x3E500000 (~2^-26)
         * Step by 0x1000 to keep count manageable */
        for (uint32_t hi = 0x00100000; hi < 0x3E500000; hi += 0x1000) {
            uint64_t bits = ((uint64_t)hi << 32);
            double input = uint64_to_double(bits);
            double result = my_asin(input);
            fwrite(&input, sizeof(double), 1, out);
            fwrite(&result, sizeof(double), 1, out);
        }

        /* Range 2: 2^-26 to 0.5 - dense sampling (every upper 32-bit value)
         * This is the main polynomial approximation range */
        for (uint32_t hi = 0x3E500000; hi < 0x3FE00000; hi++) {
            uint64_t bits = ((uint64_t)hi << 32);
            double input = uint64_to_double(bits);
            double result = my_asin(input);
            fwrite(&input, sizeof(double), 1, out);
            fwrite(&result, sizeof(double), 1, out);
        }

        /* Range 3: 0.5 to 1.0 - dense sampling
         * This range uses sqrt-based range reduction */
        for (uint32_t hi = 0x3FE00000; hi < 0x3FF00000; hi++) {
            uint64_t bits = ((uint64_t)hi << 32);
            double input = uint64_to_double(bits);
            double result = my_asin(input);
            fwrite(&input, sizeof(double), 1, out);
            fwrite(&result, sizeof(double), 1, out);
        }

        /* Special values */
        {
            double special[] = {1.0, -1.0, 0.0, -0.0};
            int i;
            for (i = 0; i < 4; i++) {
                double result = my_asin(special[i]);
                fwrite(&special[i], sizeof(double), 1, out);
                fwrite(&result, sizeof(double), 1, out);
            }
        }
    }

    fclose(out);
    FreeLibrary(hLib);
    return 0;
}
