#include <windows.h>
#include <stdio.h>
#include <stdint.h>
#include <math.h>
#include <stdlib.h>

typedef float (__cdecl *asinf_func)(float);

// Helper to reinterpret uint32_t as float
float uint32_to_float(uint32_t u) {
    union {
        uint32_t u;
        float f;
    } x;
    x.u = u;
    return x.f;
}

int main(int argc, char *argv[]) {
    FILE *out;
    HMODULE hLib;

    int native = 0;
    const char *filename = "math_test_out.bin";
    if (argc > 1) {
        filename = argv[1];
    }
    if (argc > 2) {
        char* p;
        errno = 0; // not 'int errno', because the '#include' already defined it
        long arg = strtol(argv[2], &p, 10);
        if (*p != '\0' || errno != 0) {
            return 1; // In main(), returning non-zero means failure
        }

        if (arg < INT_MIN || arg > INT_MAX) {
            return 1;
        }
        native = arg;

        // Everything went well, print it as a regular number plus a newline
        printf("native: %d\n", native);
        }

    out = fopen(filename, "wb");
    if (!out) {
        perror("fopen");
        return 1;
    }

    if (native == 0) {
        hLib = LoadLibraryA("/home/emil/code/ucrtcringe/dll/ucrtbase_builtin.dll");
    } else {
        hLib = LoadLibraryA("/home/emil/code/ucrtcringe/dll/ucrtbase.dll");
    }

    if (!hLib) {
        printf("Failed to load ucrtbase.dll\n");
        return 1;
    }
    
    asinf_func my_asinf = (asinf_func)GetProcAddress(hLib, "asinf");
    if (!my_asinf) {
        printf("Failed to get address of asin\n");
        FreeLibrary(hLib);
        return 1;
    }

    // focus on 0.00024414065 <= x < 0.5 for now
    // this has a different (and likely simpler) implementation
    for (uint32_t i = 0x39800001; ; ++i) {
        if (i == 0x3f000000) break;

        float input = uint32_to_float(i);
        float result_asin = my_asinf(input);

        // Write input, asin (all as float32)
        fwrite(&input, sizeof(float), 1, out);
        fwrite(&result_asin, sizeof(float), 1, out);
    }

    fclose(out);
    FreeLibrary(hLib);
    return 0;
}
