// Zintegrowany generator + sito spektralne grafów całkowitych
//
// geng z nauty jest wbudowany jako biblioteka (poprzez OUTPROC callback).
// Każdy wątek OpenMP odpala własną instancję geng_main z innym
// res/mod (-r) - geng dzieli przestrzeń generacji deterministycznie
// pomiędzy partycje, więc wątki nie generują tego samego grafu.
//
// OUTPROC jest wywoływany przez geng dla każdego wygenerowanego grafu.
// Wykonujemy w nim sito spektralne in-place (bez pipe'ów, bez kopiowania).
//
// Kompilacja (Apple Clang + libomp + nauty):
//   clang -c -O3 -DUSE_TLS -DMAXN=WORDSIZE \
//         -DOUTPROC=process_graph -DGENG_MAIN=geng_main \
//         -I/opt/homebrew/Cellar/nauty/2.9.3/include/nauty geng.c -o geng.o
//   clang++ -O3 -Xclang -fopenmp \
//         -I/opt/homebrew/opt/libomp/include \
//         -I/opt/homebrew/Cellar/nauty/2.9.3/include/nauty \
//         program_omp.cpp geng.o \
//         /opt/homebrew/Cellar/nauty/2.9.3/lib/libnautyTL1.a \
//         -L/opt/homebrew/opt/libomp/lib -lomp -o program_omp

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <chrono>
#include <vector>
#include <string>
#include <omp.h>

extern "C" {
#include "nauty.h"
}

#define MAX_NODES 16
#define TRI_SIZE (MAX_NODES*(MAX_NODES+1)/2 + 1)


// =================== Sito spektralne ===================

static void matrix_to_lower_triangle(int n, const int adj[MAX_NODES][MAX_NODES], double *packed)
{
    int idx = 1;
    for (int row = 0; row < n; row++)
        for (int col = 0; col <= row; col++)
            packed[idx++] = (double)adj[row][col];
}

static int all_eigenvalues_integral(int n, const double *eigenvals)
{
    for (int i = 1; i <= n; i++) {
        double v = eigenvals[i];
        if (fabs(v - round(v)) > 1e-5)
            return 0;
    }
    return 1;
}

// Algorytm Householdera + bisekcja na trójprzekątniowej.
// "pure" - operuje wyłącznie na lokalnych argumentach -> thread-safe.
static int compute_sym_eigenvalues(int n, double *mat, int lo, int hi, double *result)
{
    int i, j, k, p, q, L, L1, sign;
    double lambda, eps, g, h, max_val, min_val, norm, s, t, u, w;
    int cond;
    double diag[MAX_NODES+1], off[MAX_NODES+1], off2[MAX_NODES+1], lower[MAX_NODES+1];

    if (!((1 <= lo) && (lo <= hi) && (hi <= n)))
        return 1;

    i = 0;
    for (L = 1; L <= n; L++) { i += L; diag[L] = mat[i]; }

    for (L = n; L >= 2; L--) {
        i--; j = i; h = mat[j]; s = 0;
        for (k = L - 2; k >= 1; k--) { i--; g = mat[i]; s += g * g; }
        i--;
        if (s == 0) {
            off[L] = h; off2[L] = h * h; mat[j] = 0.0;
        } else {
            s += h * h; off2[L] = s;
            g = sqrt(s);
            if (h >= 0.0) g = -g;
            off[L] = g;
            s = 1.0 / (s - h * g);
            mat[j] = h - g;
            h = 0.0; L1 = L - 1; p = 1;
            for (j = 1; j <= L1; j++) {
                q = p; g = 0;
                for (k = 1; k <= L1; k++) {
                    g += mat[q] * mat[i + k];
                    if (k < j) sign = 1; else sign = k;
                    q += sign;
                }
                p += j; g *= s; off[j] = g;
                h += mat[i + j] * g;
            }
            h *= 0.5 * s; p = 1;
            for (j = 1; j <= L1; j++) {
                s = mat[i + j]; g = off[j] - h * s; off[j] = g;
                for (k = 1; k <= j; k++) {
                    mat[p] += -s * off[k] - mat[i + k] * g;
                    p++;
                }
            }
        }
        h = diag[L]; diag[L] = mat[i + L]; mat[i + L] = h;
    }

    h = diag[1]; diag[1] = mat[1]; mat[1] = h;
    off[1] = 0.0; off2[1] = 0.0;
    s = diag[n]; t = fabs(off[n]);
    min_val = s - t; max_val = s + t;
    for (i = n - 1; i >= 1; i--) {
        u = fabs(off[i]); h = t + u; t = u; s = diag[i];
        u = s - h; if (u < min_val) min_val = u;
        u = s + h; if (u > max_val) max_val = u;
    }
    for (i = 1; i <= n; i++) { lower[i] = min_val; result[i] = max_val; }
    norm = fabs(min_val); s = fabs(max_val);
    if (s > norm) norm = s;
    w = max_val; lambda = norm;

    for (k = hi; k >= lo; k--) {
        eps = 7.28e-17 * norm; s = min_val; i = k;
        do {
            cond = 0; g = lower[i];
            if (s < g) s = g; else { i--; if (i >= lo) cond = 1; }
        } while (cond);
        g = result[k]; if (w > g) w = g;
        while (w - s > 2.91e-16 * (fabs(s) + fabs(w)) + eps) {
            L1 = 0; g = 1.0; t = 0.5 * (s + w);
            for (i = 1; i <= n; i++) {
                if (g != 0) g = off2[i] / g;
                else        g = fabs(6.87e15 * off[i]);
                g = diag[i] - t - g;
                if (g < 0) L1++;
            }
            if (L1 < lo) { s = t; lower[lo] = s; }
            else if (L1 < k) { s = t; lower[L1 + 1] = s;
                               if (result[L1] > t) result[L1] = t; }
            else w = t;
        }
        result[k] = 0.5 * (s + w);
    }
    return 1;
}


// =================== Encoding graph6 (thread-safe) ===================
// Konwersja macierzy sąsiedztwa do napisu graph6 dla 1 <= n <= 62.
// Bity są pakowane kolumnowo: bit dla (row, col) gdzie col < row.
static void encode_graph6_from_adj(int n, const int adj[MAX_NODES][MAX_NODES], char *out)
{
    out[0] = (char)(n + 63);
    int char_pos = 1, shift = 5, buf = 0;
    for (int row = 1; row < n; row++) {
        for (int col = 0; col < row; col++) {
            if (adj[row][col]) buf |= (1 << shift);
            if (shift == 0) { out[char_pos++] = (char)(buf + 63); shift = 5; buf = 0; }
            else            { shift--; }
        }
    }
    if (shift != 5) out[char_pos++] = (char)(buf + 63);
    out[char_pos] = '\0';
}


// =================== Per-thread state + callback ===================

struct ThreadResult {
    long long total = 0;
    long long integral = 0;
    std::vector<std::string> integral_graphs;
};

static std::vector<ThreadResult> g_thread_results;

// Globalny licznik do pomiaru przepustowosci w stalym czasie (timeout).
// Co PROGRESS_STEP grafow wypisuje biezacy licznik na stderr - dzieki temu
// przerwany przez timeout pomiar zostawia uzyteczna liczbe.
#include <atomic>
static std::atomic<long long> g_progress{0};
#define PROGRESS_STEP 2000000

// OUTPROC callback wywoływany przez geng_main dla każdego grafu.
// Działa w kontekście wątku OMP, który odpalił geng_main.
extern "C" void process_graph(FILE * /*outfile*/, graph *g, int n)
{
    int tid = omp_get_thread_num();
    ThreadResult &r = g_thread_results[tid];
    r.total++;

    long long cur = ++g_progress;
    if (cur % PROGRESS_STEP == 0)
        fprintf(stderr, "PROGRESS %lld\n", cur);

    if (n <= 0 || n > MAX_NODES) return;

    // nauty graph -> adj matrix. Dla MAXN=WORDSIZE, m=1: g[i] to setword
    // z bitami "MSB-first" - bit (WORDSIZE-1) to wierzchołek 0.
    int adj[MAX_NODES][MAX_NODES];
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++) adj[i][j] = 0;

    for (int i = 0; i < n; i++) {
        setword row = g[i];
        for (int j = 0; j < n; j++)
            if (row & bit[j]) adj[i][j] = 1;
    }

    double packed[TRI_SIZE];
    matrix_to_lower_triangle(n, adj, packed);

    double eigenvals[MAX_NODES + 1];
    compute_sym_eigenvalues(n, packed, 1, n, eigenvals);

    if (all_eigenvalues_integral(n, eigenvals)) {
        r.integral++;
        char g6_buf[64];
        encode_graph6_from_adj(n, adj, g6_buf);
        r.integral_graphs.emplace_back(g6_buf);
        #pragma omp critical
        {
            printf("%s\n", g6_buf);
            fflush(stdout);
            if (getenv("EXIT_AFTER_FIRST")) {
                fprintf(stderr, "Found first integral graph, exiting.\n");
                exit(0);
            }
        }
    }
}


// =================== main ===================

extern "C" int geng_main(int argc, char *argv[]);

int main(int argc, char *argv[])
{
    int n = (argc > 1) ? atoi(argv[1]) : 16;
    int k = (argc > 2) ? atoi(argv[2]) : 46;

    int n_threads = omp_get_max_threads();
    g_thread_results.assign(n_threads, ThreadResult{});

    // Opcjonalnie: przetworz tylko WYCINEK przestrzeni - partycje 0..L-1 z modulu M.
    // Pozwala uruchomic ograniczony, policzalny test na n=16 k=46.
    //   program_omp n k            -> cala przestrzen (L = M = n_threads*16)
    //   program_omp n k L M        -> partycje 0..L-1 z modulu M (wycinek L/M)
    const int oversub = 16;
    int partitions = n_threads * oversub;   // modulus M
    int run_parts  = partitions;            // ile partycji faktycznie liczymy (L)
    if (argc > 4) { run_parts = atoi(argv[3]); partitions = atoi(argv[4]); }

    fprintf(stderr, "Searching integral connected graphs: n=%d, k=%d, threads=%d, partycje=%d/%d\n",
            n, k, n_threads, run_parts, partitions);

    auto t_start = std::chrono::high_resolution_clock::now();

    // Więcej partycji niż wątków -> dynamiczne load balancing.
    // Geng's res/mod nie jest zbalansowane przez liczbę grafów (różnice 10x+
    // między partycjami), więc tworzymy nadmiar zadań i rozdzielamy dynamicznie.
    #pragma omp parallel for schedule(dynamic, 1)
    for (int p = 0; p < run_parts; p++) {
        char nstring[16], estring[16], res_str[24];
        snprintf(nstring, sizeof(nstring), "%d", n);
        snprintf(estring, sizeof(estring), "%d:%d", k, k);
        snprintf(res_str, sizeof(res_str), "%d/%d", p, partitions);

        // geng -cq <n> <k>:<k> <p>/<partitions>
        char *gargv[7];
        gargv[0] = (char *)"geng";
        gargv[1] = (char *)"-cq";
        gargv[2] = nstring;
        gargv[3] = estring;
        gargv[4] = res_str;
        gargv[5] = NULL;

        geng_main(5, gargv);
    }

    auto t_end = std::chrono::high_resolution_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(t_end - t_start);

    // Zbiór statystyk z wszystkich wątków (grafy już zostały wydrukowane w callbacku)
    long long total = 0, integral = 0;
    for (auto &r : g_thread_results) {
        total    += r.total;
        integral += r.integral;
    }

    fprintf(stderr, "Threads used     : %d\n", n_threads);
    fprintf(stderr, "Total graphs gen : %lld\n", total);
    fprintf(stderr, "Integral spectra : %lld\n", integral);
    fprintf(stderr, "Elapsed time     : %.3f s\n", elapsed.count() / 1000.0);

    // Per-thread breakdown
    for (int i = 0; i < n_threads; i++)
        fprintf(stderr, "  thread %2d: total=%lld, integral=%lld\n",
                i, g_thread_results[i].total, g_thread_results[i].integral);

    return 0;
}
