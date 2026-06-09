/**
@file integral.h
@brief Wspolny rdzen sita spektralnego: test calkowitosci widma grafu.

Algorytm (jak w sito5 prowadzacego, oparty na kodzie Pascala A.Marciniaka):
  1. Redukcja Householdera macierzy symetrycznej do postaci trojprzekatniowej.
  2. Bisekcja (ciagi Sturma) wyznaczajaca kolejne wartosci wlasne.
  3. WCZESNE ODRZUCENIE: gdy przedzial bisekcji [s,w] nie moze juz zawierac
     liczby calkowitej (floor(w) < s), graf jest natychmiast odrzucany.

Optymalizacje wzgledem sito5:
  * typ `double` (SSE2) zamiast `long double` (80-bit x87) - szybsza arytmetyka,
    zgodna z CUDA (brak long double na GPU).
  * funkcja w pelni reentrantna (wylacznie zmienne lokalne) => bezpieczna watkowo;
    kazdy watek dostaje wlasny komplet struktur pomocniczych na stosie.

@param BUFOR napis w formacie graph6 (czytane sa tylko bajty kodujace graf)
@return 1 gdy wszystkie wartosci wlasne sa calkowite, 0 w przeciwnym razie
*/
#ifndef INTEGRAL_H
#define INTEGRAL_H

#include <math.h>

#define NMAX 20
#define INT_TOL 1e-4   /* tolerancja "bliskosci" do liczby calkowitej (jak 10e-5 w sito5) */

static inline int eigentest(const char *BUFOR)
{
    int i, j, k, k3, k4, L, L1, z, cond;
    double eps, g, h, ma, mn, norm, s, t, u, w;
    double d[NMAX + 1], e[NMAX + 1], e2[NMAX + 1], Lb[NMAX + 1], x[NMAX + 1];
    double a[NMAX * (NMAX - 1) / 2 + NMAX + 1];
    int n, bit = 32, poz = 1, poz2 = 1;

    n = BUFOR[0] - 63;
    a[0] = 0.0;

    /* graph6 -> dolnotrojkatna spakowana macierz sasiedztwa (indeksowanie od 1) */
    for (i = 0; i < n; i++)
        for (j = 0; j <= i; j++) {
            if (i == j) { a[poz2++] = 0.0; }
            else {
                if (bit == 0) { bit = 32; poz++; }
                a[poz2++] = ((BUFOR[poz] - 63) & bit) ? 1.0 : 0.0;
                bit >>= 1;
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
            if (floor(w + INT_TOL) < s - INT_TOL) return 0;  /* brak liczby calk. w przedziale */
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
        if (!((ceil(u) - u < INT_TOL) || (u - floor(u) < INT_TOL))) return 0;
    }
    return 1;
}

#endif /* INTEGRAL_H */
