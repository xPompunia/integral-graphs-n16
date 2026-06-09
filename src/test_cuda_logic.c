/**
@file test_cuda_logic.c
@brief Weryfikacja logiki CUDA na CPU (bez GPU).

Uzywa DOKLADNIE tego samego rdzenia (integral_bits.h: ib_g6_to_bits +
eigentest_bits) co jadro CUDA. Pozwala sprawdzic poprawnosc sciezki binarnej
(kodowanie bitowe + solver) zanim zainstalujemy CUDA Toolkit.

Kompilacja:
  gcc -O3 -march=native test_cuda_logic.c -o test_cuda_logic -lm
*/
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "integral_bits.h"

int main(void)
{
    char line[64];
    long long total = 0, integral = 0;
    while (fgets(line, sizeof(line), stdin)) {
        line[strcspn(line, "\r\n")] = '\0';
        if (line[0] == '\0') continue;
        total++;
        int n = line[0] - 63;
        int nwords = ib_words_for(n);
        uint32_t bits[IB_MAXWORDS];
        ib_g6_to_bits(line, n, bits, nwords);
        if (eigentest_bits(bits, n)) { integral++; fputs(line, stdout); fputc('\n', stdout); }
    }
    fprintf(stderr, "[test_cuda_logic] przebadane=%lld calkowite=%lld\n", total, integral);
    return 0;
}
