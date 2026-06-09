/**
@file sito_cuda.cu
@brief Sito spektralne - wersja CUDA (wiele grafow jednoczesnie na GPU).

Model obliczen (zgodnie z zaleceniem prowadzacego dla problemow grafowych):
  * przetwarzamy WIELE instancji grafu jednoczesnie - 1 watek CUDA = 1 graf,
  * struktura grafu kodowana BINARNIE przy przekazaniu do jadra: dolnotrojkatna
    macierz sasiedztwa (n*(n-1)/2 bitow) pakowana w slowa 32-bitowe,
  * uklad SoA (Structure of Arrays): bity grafu o numerze idx leza w
    d_bits[w*num + idx] -> sasiednie watki czytaja sasiednie adresy
    (dostep scalony / coalesced).

Algorytm testu calkowitosci (eigentest_bits) jest we wspolnym naglowku
integral_bits.h i jest IDENTYCZNY jak w wersji sekwencyjnej i OpenMP
(integral.h) -> wyniki wszystkich wersji sa bit-w-bit takie same.

Przygotowanie danych na hoscie (odczyt + dekodowanie graph6 -> bitmapa) jest
zrownoleglone przez OpenMP - to czesc CPU, ktora przygotowuje wsad dla GPU.
Wlasciwe sito (solver) pozostaje w calosci na GPU. Bez tego zrownoleglenia
sekwencyjny odczyt na hoscie dominowalby czas (efekt Amdahla).

Kompilacja (RTX 3060 Ti = Ampere, sm_86):
  nvcc -O3 -arch=sm_86 -Xcompiler -fopenmp sito_cuda.cu -o sito_cuda

Uruchomienie:
  ./geng -cq 16 46:46 | ./sito_cuda                 # wypis na stdout
  ./geng -cq 16 46:46 | ./sito_cuda wynik.g6        # zapis do pliku
  ./geng -cq 16 46:46 | ./sito_cuda wynik.g6 256    # rozmiar bloku = 256
  ./geng -cq  9       | ./sito_cuda - 1024 | wc -l  # '-' = stdout (test OEIS)
*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <omp.h>
#include <cuda_runtime.h>
#ifdef _WIN32
#include <io.h>
#include <fcntl.h>
#define READ _read
#else
#include <unistd.h>
#define READ read
#endif
#include "integral_bits.h"

#define BUFSIZE 32                  /* dlugosc napisu graph6 (n<=22)            */
#define CHUNK   (16 * 1024 * 1024)  /* rozmiar bloku odczytu hosta              */
#define MAXLINE 64

/* makro kontroli bledow CUDA (poprawne __FILE__/__LINE__) */
#define CUDA_CHECK(call)                                                        \
    do {                                                                       \
        cudaError_t _err = (call);                                            \
        if (_err != cudaSuccess) {                                            \
            fprintf(stderr, "Blad CUDA %s:%d - %s\n",                         \
                    __FILE__, __LINE__, cudaGetErrorString(_err));            \
            exit(EXIT_FAILURE);                                              \
        }                                                                    \
    } while (0)


/* jadro: kazdy watek testuje jeden graf (solver bez zmian - integral_bits.h) */
__global__ void sieve_kernel(const uint32_t *__restrict__ d_bits,
                             int *__restrict__ d_res,
                             int n, int nwords, int num)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num) return;

    uint32_t bits[IB_MAXWORDS];
    for (int w = 0; w < nwords; w++)
        bits[w] = d_bits[(size_t)w * num + idx];   /* dostep scalony (SoA) */

    d_res[idx] = eigentest_bits(bits, n);
}


/* przetworzenie jednej porcji na GPU; zwraca liczbe znalezionych grafow.
   lines[] -> wskazniki na napisy graph6 (do wypisania trafien). */
static long long process_batch(char **lines, uint32_t *h_bits, int *h_res,
                               uint32_t *d_bits, int *d_res,
                               int count, int n, int nwords, int block,
                               FILE *out, float *kernel_ms_accum)
{
    CUDA_CHECK(cudaMemcpy(d_bits, h_bits,
                          (size_t)nwords * count * sizeof(uint32_t),
                          cudaMemcpyHostToDevice));

    int grid = (count + block - 1) / block;

    cudaEvent_t e0, e1;
    CUDA_CHECK(cudaEventCreate(&e0));
    CUDA_CHECK(cudaEventCreate(&e1));
    CUDA_CHECK(cudaEventRecord(e0));
    sieve_kernel<<<grid, block>>>(d_bits, d_res, n, nwords, count);
    CUDA_CHECK(cudaEventRecord(e1));
    CUDA_CHECK(cudaEventSynchronize(e1));
    float ms = 0; CUDA_CHECK(cudaEventElapsedTime(&ms, e0, e1));
    *kernel_ms_accum += ms;
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventDestroy(e0));
    CUDA_CHECK(cudaEventDestroy(e1));

    CUDA_CHECK(cudaMemcpy(h_res, d_res, (size_t)count * sizeof(int),
                          cudaMemcpyDeviceToHost));

    long long found = 0;
    for (int i = 0; i < count; i++)
        if (h_res[i]) { fputs(lines[i], out); fputc('\n', out); found++; }
    return found;
}


int main(int argc, char *argv[])
{
    FILE *out = stdout;
    if (argc > 1 && strcmp(argv[1], "-") != 0) {
        out = fopen(argv[1], "w");
        if (!out) { perror(argv[1]); return EXIT_FAILURE; }
    }
    int block = 1024;                            /* domyslnie max watkow w bloku */
    if (argc > 2) {
        block = atoi(argv[2]);
        if (block < 1 || block > 1024) { fprintf(stderr, "blok 1..1024; ustawiam 1024\n"); block = 1024; }
    }
#ifdef _WIN32
    _setmode(0, _O_BINARY);                       /* bez translacji \r\n */
#endif

    /* maks. liczba grafow w jednej porcji (CHUNK / min. dlugosc linii) */
    const int CAP = CHUNK / 4 + 16;

    char     *buf    = (char *)    malloc(CHUNK + MAXLINE + 1);
    char    **lines  = (char **)   malloc((size_t)CAP * sizeof(char *));
    uint32_t *h_bits = (uint32_t *)malloc((size_t)CAP * IB_MAXWORDS * sizeof(uint32_t));
    int      *h_res  = (int *)     malloc((size_t)CAP * sizeof(int));
    uint32_t *d_bits = NULL; int *d_res = NULL;
    CUDA_CHECK(cudaMalloc(&d_bits, (size_t)CAP * IB_MAXWORDS * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_res,  (size_t)CAP * sizeof(int)));

    size_t carry = 0;
    int n = 0, nwords = 0;
    long long total = 0, integral = 0;
    float kernel_ms = 0.0f;

    cudaEvent_t w0, w1;
    CUDA_CHECK(cudaEventCreate(&w0));
    CUDA_CHECK(cudaEventCreate(&w1));
    CUDA_CHECK(cudaEventRecord(w0));

    long r;
    while ((r = READ(0, buf + carry, CHUNK)) > 0) {
        size_t avail = carry + (size_t)r;

        /* koniec ostatniej kompletnej linii */
        size_t proc = 0;
        for (size_t p = avail; p > 0; p--)
            if (buf[p - 1] == '\n') { proc = p; break; }

        /* podzial na linie in-place -> tablica wskaznikow */
        int cnt = 0; size_t start = 0;
        for (size_t p = 0; p < proc; p++) {
            if (buf[p] == '\n') {
                size_t end = p;
                if (end > start && buf[end - 1] == '\r') end--;
                buf[end] = '\0';
                if (end > start) lines[cnt++] = buf + start;
                start = p + 1;
            }
        }
        if (cnt == 0) { carry = avail - proc; memmove(buf, buf + proc, carry); continue; }

        if (n == 0) { n = lines[0][0] - 63; nwords = ib_words_for(n); }
        total += cnt;

        /* RÓWNOLEGŁE dekodowanie graph6 -> bitmapa (SoA, skok = cnt) na CPU */
        #pragma omp parallel for schedule(static) default(none) \
                shared(lines, h_bits, cnt, n, nwords)
        for (int gi = 0; gi < cnt; gi++) {
            uint32_t tmp[IB_MAXWORDS];
            ib_g6_to_bits(lines[gi], n, tmp, nwords);
            for (int w = 0; w < nwords; w++)
                h_bits[(size_t)w * cnt + gi] = tmp[w];
        }

        integral += process_batch(lines, h_bits, h_res, d_bits, d_res,
                                  cnt, n, nwords, block, out, &kernel_ms);

        carry = avail - proc;
        memmove(buf, buf + proc, carry);
    }

    CUDA_CHECK(cudaEventRecord(w1));
    CUDA_CHECK(cudaEventSynchronize(w1));
    float wall_ms = 0; CUDA_CHECK(cudaEventElapsedTime(&wall_ms, w0, w1));

    if (out != stdout) fclose(out);
    free(buf); free(lines); free(h_bits); free(h_res);
    CUDA_CHECK(cudaFree(d_bits));
    CUDA_CHECK(cudaFree(d_res));

    fprintf(stderr, "[sito_cuda] n=%d blok=%d watki_host=%d przebadane=%lld calkowite=%lld "
                    "czas_jadra=%.3fs czas_calk=%.3fs\n",
            n, block, omp_get_max_threads(), total, integral,
            kernel_ms / 1000.0f, wall_ms / 1000.0f);
    return EXIT_SUCCESS;
}
