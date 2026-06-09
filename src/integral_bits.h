/**
@file integral_bits.h
@brief Rdzen sita dzialajacy na BINARNIE zakodowanym grafie (dla CUDA + testu CPU).

DOKLADNIE ten sam algorytm numeryczny co integral.h (Householder + bisekcja Sturma
z wczesnym odrzuceniem) - jedyna roznica to zrodlo danych: macierz sasiedztwa jest
odtwarzana z upakowanej BITMAPY (slowa 32-bit) zamiast z napisu graph6. Dzieki
wspolnemu naglowkowi kod jadra CUDA i jego weryfikacja na CPU korzystaja z tej samej
logiki.

Kwalifikator IB_QUAL:
  * pod nvcc funkcja jest __host__ __device__ (wolana z jadra i z hosta),
  * pod zwyklym kompilatorem C jest static inline (do testu na CPU).

Arytmetyka: double (wymagana). Wariant float zostal porzucony - progi zbieznosci
bisekcji (2.91e-16, 7.28e-17) sa dobrane pod precyzje double; we float petla bisekcji
sie nie zbiega. Algorytm bisekcji Sturma wymaga podwojnej precyzji.
*/
#ifndef INTEGRAL_BITS_H
#define INTEGRAL_BITS_H

#include <math.h>
#include <stdint.h>

#ifdef __CUDACC__
#define IB_QUAL static inline __host__ __device__
#else
#define IB_QUAL static inline
#endif

#define IB_NMAX  16
#define IB_ASIZE (IB_NMAX*(IB_NMAX-1)/2 + IB_NMAX + 1)
#define IB_MAXWORDS 8
#define IB_INT_TOL 1e-4

/* liczba slow 32-bit na dolny trojkat dla n wierzcholkow */
static inline int ib_words_for(int n) { return (n * (n - 1) / 2 + 31) / 32; }

/* HOST: graph6 -> bity dolnego trojkata (kolejnosc jak a[] w integral.h) */
static inline void ib_g6_to_bits(const char *buf, int n, uint32_t *out, int nwords)
{
    for (int w = 0; w < nwords; w++) out[w] = 0u;
    int bitg = 32, poz = 1, bitidx = 0;
    for (int i = 1; i < n; i++)
        for (int j = 0; j < i; j++) {
            if (bitg == 0) { bitg = 32; poz++; }
            if ((buf[poz] - 63) & bitg) out[bitidx >> 5] |= (1u << (bitidx & 31));
            bitidx++;
            bitg >>= 1;
        }
}

/* RDZEN: test calkowitosci widma, dane = bitmapa dolnego trojkata */
IB_QUAL int eigentest_bits(const uint32_t *bits, int n)
{
    int i, j, k, k3, k4, L, L1, z, cond;
    double eps, g, h, ma, mn, norm, s, t, u, w;
    double d[IB_NMAX + 1], e[IB_NMAX + 1], e2[IB_NMAX + 1], Lb[IB_NMAX + 1], x[IB_NMAX + 1];
    double a[IB_ASIZE];
    int poz2 = 1, bitidx = 0;

    a[0] = 0.0;
    /* bitmapa -> dolnotrojkatna spakowana macierz sasiedztwa (indeksowanie od 1) */
    for (i = 0; i < n; i++)
        for (j = 0; j <= i; j++) {
            if (i == j) { a[poz2++] = 0.0; }
            else {
                a[poz2++] = ((bits[bitidx >> 5] >> (bitidx & 31)) & 1u) ? 1.0 : 0.0;
                bitidx++;
            }
        }

    const int k1 = 1, k2 = n;
    i = 0;
    for (L = 1; L <= n; L++) { i += L; d[L] = a[i]; }

    /* --- Redukcja Householdera do postaci trojprzekatniowej --- */
    for (L = n; L >= 2; L--) {
        i--; j = i; h = a[j]; s = 0;
        for (k = L - 2; k >= 1; k--) { i--; g = a[i]; s += g * g; }
        i--;
        if (s == 0) { e[L] = h; e2[L] = h * h; a[j] = 0.0; }
        else {
            s += h * h; e2[L] = s; g = sqrt(s); if (h >= 0.0) g = -g;
            e[L] = g;
            s = 1.0 / (s - h * g);
            a[j] = h - g; h = 0.0; L1 = L - 1; k3 = 1;
            for (j = 1; j <= L1; j++) {
                k4 = k3; g = 0;
                for (k = 1; k <= L1; k++) {
                    g += a[k4] * a[i + k];
                    if (k < j) z = 1; else z = k;
                    k4 += z;
                }
                k3 += j; g *= s; e[j] = g; h += a[i + j] * g;
            }
            h *= 0.5 * s; k3 = 1;
            for (j = 1; j <= L1; j++) {
                s = a[i + j]; g = e[j] - h * s; e[j] = g;
                for (k = 1; k <= j; k++) { a[k3] += -s * e[k] - a[i + k] * g; k3++; }
            }
        }
        h = d[L]; d[L] = a[i + L]; a[i + L] = h;
    }

    h = d[1]; d[1] = a[1]; a[1] = h; e[1] = 0.0; e2[1] = 0.0; s = d[n];
    t = fabs(e[n]); mn = s - t; ma = s + t;
    for (i = n - 1; i >= 1; i--) {
        u = fabs(e[i]); h = t + u; t = u; s = d[i];
        u = s - h; if (u < mn) mn = u;
        u = s + h; if (u > ma) ma = u;
    }
    for (i = 1; i <= n; i++) { Lb[i] = mn; x[i] = ma; }
    norm = fabs(mn); s = fabs(ma); if (s > norm) norm = s;
    w = ma; eps = 7.28e-17 * norm;

    /* --- Bisekcja kolejnych wartosci wlasnych z wczesnym odrzuceniem --- */
    for (k = k2; k >= k1; k--) {
        s = mn; i = k;
        do { cond = 0; g = Lb[i];
             if (s < g) s = g; else { i--; if (i >= k1) cond = 1; }
        } while (cond);
        g = x[k]; if (w > g) w = g;
        while (w - s > 2.91e-16 * (fabs(s) + fabs(w)) + eps) {
            if (floor(w + IB_INT_TOL) < s - IB_INT_TOL) return 0;  /* brak liczby calk. */
            L1 = 0; g = 1.0; t = 0.5 * (s + w);
            for (i = 1; i <= n; i++) {
                if (g != 0) g = e2[i] / g; else g = fabs(6.87e15 * e[i]);
                g = d[i] - t - g;
                if (g < 0) L1++;
            }
            if (L1 < k1) { s = t; Lb[k1] = s; }
            else if (L1 < k) { s = t; Lb[L1 + 1] = s; if (x[L1] > t) x[L1] = t; }
            else w = t;
        }
        u = 0.5 * (s + w); x[k] = u;
        if (!((ceil(u) - u < IB_INT_TOL) || (u - floor(u) < IB_INT_TOL))) return 0;
    }
    return 1;
}

#endif /* INTEGRAL_BITS_H */
