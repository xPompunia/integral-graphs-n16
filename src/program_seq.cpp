#include <stdlib.h>
#include <stdio.h>
#include <stdbool.h>
#include <unistd.h>
#include <math.h>
#include <string.h>
#include <chrono>

#define MAX_NODES 16
#define TRI_SIZE  (MAX_NODES*(MAX_NODES+1)/2 + 1)


// Decode graph6-encoded string into an adjacency matrix (N <= MAX_NODES)
void decode_graph6(const char *encoded, int *node_count, int adj[MAX_NODES][MAX_NODES])
{
    *node_count = encoded[0] - 63;
    if (*node_count <= 0 || *node_count > MAX_NODES)
        return;

    for (int r = 0; r < *node_count; r++)
        for (int c = 0; c < *node_count; c++)
            adj[r][c] = 0;

    int char_pos = 1;
    int shift = 5; // graph6 packs 6 bits per character

    for (int row = 1; row < *node_count; row++) {
        for (int col = 0; col < row; col++) {
            if (((encoded[char_pos] - 63) >> shift) & 1)
                adj[row][col] = adj[col][row] = 1;

            if (shift == 0) {
                shift = 5;
                char_pos++;
            } else {
                shift--;
            }
        }
    }
}


// Pack adjacency matrix into a lower-triangular 1-indexed vector
void matrix_to_lower_triangle(int n, int adj[MAX_NODES][MAX_NODES], double *packed)
{
    int idx = 1;
    for (int row = 0; row < n; row++)
        for (int col = 0; col <= row; col++)
            packed[idx++] = (double)adj[row][col];
}


// Check whether all computed eigenvalues are (approximately) integers
int all_eigenvalues_integral(int n, double *eigenvals)
{
    for (int i = 1; i <= n; i++) {
        double v = eigenvals[i];
        if (fabs(v - round(v)) > 1e-5)
            return 0;
    }
    return 1;
}


// Compute eigenvalues of a symmetric matrix using Householder + bisection
int compute_sym_eigenvalues(int n, double *mat, int lo, int hi, double *result)
{
    int i, j, k, p, q, L, L1, sign;
    double eps, g, h, max_val, min_val, norm, s, t, u, w;
    int cond;
    double diag[MAX_NODES+1], off[MAX_NODES+1], off2[MAX_NODES+1], lower[MAX_NODES+1];

    if (!((1 <= lo) && (lo <= hi) && (hi <= n)))
        return 1;

    // Copy diagonal
    i = 0;
    for (L = 1; L <= n; L++) {
        i += L;
        diag[L] = mat[i];
    }

    // Householder reduction to tridiagonal form
    for (L = n; L >= 2; L--) {
        i--;
        j = i;
        h = mat[j];
        s = 0;

        for (k = L - 2; k >= 1; k--) {
            i--;
            g = mat[i];
            s += g * g;
        }
        i--;

        if (s == 0) {
            off[L]  = h;
            off2[L] = h * h;
            mat[j]  = 0.0;
        } else {
            s += h * h;
            off2[L] = s;
            g = sqrt(s);
            if (h >= 0.0) g = -g;
            off[L] = g;

            s = 1.0 / (s - h * g);
            mat[j] = h - g;
            h = 0.0;
            L1 = L - 1;
            p = 1;

            for (j = 1; j <= L1; j++) {
                q = p;
                g = 0;
                for (k = 1; k <= L1; k++) {
                    g += mat[q] * mat[i + k];
                    if (k < j) sign = 1; else sign = k;
                    q += sign;
                }
                p += j;
                g *= s;
                off[j] = g;
                h += mat[i + j] * g;
            }

            h *= 0.5 * s;
            p = 1;

            for (j = 1; j <= L1; j++) {
                s = mat[i + j];
                g = off[j] - h * s;
                off[j] = g;
                for (k = 1; k <= j; k++) {
                    mat[p] += -s * off[k] - mat[i + k] * g;
                    p++;
                }
            }
        }

        h = diag[L];
        diag[L] = mat[i + L];
        mat[i + L] = h;
    }

    h = diag[1];
    diag[1] = mat[1];
    mat[1] = h;
    off[1]  = 0.0;
    off2[1] = 0.0;

    // Determine initial eigenvalue bounds (Gershgorin)
    s = diag[n];
    t = fabs(off[n]);
    min_val = s - t;
    max_val = s + t;

    for (i = n - 1; i >= 1; i--) {
        u = fabs(off[i]);
        h = t + u;
        t = u;
        s = diag[i];
        u = s - h; if (u < min_val) min_val = u;
        u = s + h; if (u > max_val) max_val = u;
    }

    for (i = 1; i <= n; i++) {
        lower[i]  = min_val;
        result[i] = max_val;
    }

    norm = fabs(min_val);
    s = fabs(max_val);
    if (s > norm) norm = s;

    w = max_val;

    // Bisection for each requested eigenvalue
    for (k = hi; k >= lo; k--) {
        eps = 7.28e-17 * norm;
        s = min_val;
        i = k;

        do {
            cond = 0;
            g = lower[i];
            if (s < g) {
                s = g;
            } else {
                i--;
                if (i >= lo) cond = 1;
            }
        } while (cond);

        g = result[k];
        if (w > g) w = g;

        while (w - s > 2.91e-16 * (fabs(s) + fabs(w)) + eps) {
            L1 = 0;
            g  = 1.0;
            t  = 0.5 * (s + w);

            for (i = 1; i <= n; i++) {
                if (g != 0) g = off2[i] / g;
                else        g = fabs(6.87e15 * off[i]);
                g = diag[i] - t - g;
                if (g < 0) L1++;
            }

            if (L1 < lo) {
                s = t;
                lower[lo] = s;
            } else if (L1 < k) {
                s = t;
                lower[L1 + 1] = s;
                if (result[L1] > t) result[L1] = t;
            } else {
                w = t;
            }
        }

        result[k] = 0.5 * (s + w);
    }

    return 1;
}


int main(int argc, char *argv[])
{
    // Argumenty opcjonalne: remainder divisor
    // pozwalają na podzielenie pracy między wiele procesów
    // (proces o numerze `remainder` z `divisor` procesów)
    int remainder = 0;
    int divisor   = 1;

    if (argc == 3) {
        remainder = atoi(argv[1]);
        divisor   = atoi(argv[2]);
    }

    char  *line     = NULL;
    size_t line_cap = 0;

    long long total_graphs    = 0;
    long long integral_graphs = 0;

    auto t_start = std::chrono::high_resolution_clock::now();

    while (getline(&line, &line_cap, stdin) != -1) {
        if (total_graphs % divisor == remainder) {
            line[strcspn(line, "\r\n")] = 0;

            if (strlen(line) == 0) {
                total_graphs++;
                continue;
            }

            int node_count;
            int adj[MAX_NODES][MAX_NODES];
            decode_graph6(line, &node_count, adj);

            if (node_count > 0 && node_count <= MAX_NODES) {
                double packed[TRI_SIZE];
                matrix_to_lower_triangle(node_count, adj, packed);

                double eigenvals[MAX_NODES + 1];
                compute_sym_eigenvalues(node_count, packed, 1, node_count, eigenvals);

                if (all_eigenvalues_integral(node_count, eigenvals)) {
                    printf("%s\n", line);
                    fflush(stdout);
                    integral_graphs++;
                }
            }
        }
        total_graphs++;
    }

    auto t_end   = std::chrono::high_resolution_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(t_end - t_start);

    fprintf(stderr, "Elapsed time     : %.3f s\n", elapsed.count() / 1000.0);
    fprintf(stderr, "Total graphs read: %lld\n", total_graphs);
    fprintf(stderr, "Integral spectra : %lld\n", integral_graphs);

    free(line);
    return 0;
}
