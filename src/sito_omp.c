/**
@file sito_omp.c
@brief Sito spektralne - wersja OpenMP (zoptymalizowana, potokowa przez stdin).

Strategia (zgodnie z podpowiedziami prowadzacego):
  * I/O: blokowy odczyt read() duzymi porcjami (CHUNK) zamiast fgets()
    linia-po-linii; parsowanie linii in-place (bez kopiowania napisow).
  * Dekompozycja: jedna porcja danych -> tablica wskaznikow na linie ->
    rownolegla petla #pragma omp for. Kazdy graf to niezalezne zadanie.
  * Szeregowanie: schedule(runtime) - sterowane zmienna OMP_SCHEDULE.
    Wczesne odrzucenie powoduje duza zmiennosc czasu pojedynczego grafu,
    wiec szeregowanie dynamiczne/guided rownowazy obciazenie watkow.
  * Struktury pomocnicze: komplet tablic solvera jest lokalny w eigentest()
    (na stosie watku) -> 1 komplet na 1 watek, bez falszywego wspoldzielenia.
  * Wyjscie bez sekcji krytycznej w petli goracej: watki zapisuja wynik 1/0
    do rozlacznych komorek tablicy hit[] (brak wyscigu), a wypis odbywa sie
    szeregowo po petli -> kolejnosc wynikow jest deterministyczna.

Kompilacja:
  gcc -O3 -march=native -fopenmp sito_omp.c -o sito_omp -lm

Uruchomienie:
  OMP_NUM_THREADS=16 ./geng -cq 16 46:46 | ./sito_omp wynik.g6
*/
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>
#ifdef _WIN32
#include <io.h>
#include <fcntl.h>
#define READ _read
#else
#include <unistd.h>
#define READ read
#endif
#include "integral.h"

#define CHUNK   (16 * 1024 * 1024)   /* rozmiar bloku odczytu */
#define MAXLINE 1024

int main(int argc, char *argv[])
{
    FILE *out = stdout;
    if (argc > 1) {
        out = fopen(argv[1], "w");
        if (!out) { perror(argv[1]); return EXIT_FAILURE; }
    }
#ifdef _WIN32
    _setmode(0, _O_BINARY);           /* bez translacji \r\n na wejsciu */
#endif
    /* domyslnie 'guided' (najlepsze w testach); nadpisywalne przez OMP_SCHEDULE */
    if (!getenv("OMP_SCHEDULE")) omp_set_schedule(omp_sched_guided, 0);

    char  *buf   = (char *)malloc(CHUNK + MAXLINE + 1);
    char **lines = NULL;
    size_t lines_cap = 0;
    size_t carry = 0;
    long long total = 0, integral = 0;

    double t0 = omp_get_wtime();

    long r;
    while ((r = READ(0, buf + carry, CHUNK)) > 0) {
        size_t avail = carry + (size_t)r;

        /* znajdz koniec ostatniej kompletnej linii */
        size_t proc = 0;
        for (size_t p = avail; p > 0; p--)
            if (buf[p - 1] == '\n') { proc = p; break; }

        /* policz linie i (raz) powieksz tablice wskaznikow */
        size_t nlines = 0;
        for (size_t p = 0; p < proc; p++) if (buf[p] == '\n') nlines++;
        if (nlines > lines_cap) {
            lines_cap = nlines;
            lines = (char **)realloc(lines, lines_cap * sizeof(char *));
        }

        /* podziel [0,proc) na linie in-place (znaki '\0') */
        size_t li = 0, start = 0;
        for (size_t p = 0; p < proc; p++) {
            if (buf[p] == '\n') {
                size_t end = p;
                if (end > start && buf[end - 1] == '\r') end--;
                buf[end] = '\0';
                if (end > start) lines[li++] = buf + start;
                start = p + 1;
            }
        }
        nlines = li;
        total += (long long)nlines;

        /* rownolegle sito; wyniki do rozlacznych komorek hit[] */
        char *hit = (char *)calloc(nlines ? nlines : 1, 1);
        long long batch = 0;
        #pragma omp parallel for schedule(runtime) \
                default(none) shared(lines, hit, nlines) reduction(+:batch)
        for (long long gi = 0; gi < (long long)nlines; gi++) {
            if (eigentest(lines[gi])) { hit[gi] = 1; batch++; }
        }

        /* szeregowy, deterministyczny wypis trafien */
        for (size_t gi = 0; gi < nlines; gi++)
            if (hit[gi]) { fputs(lines[gi], out); fputc('\n', out); }
        free(hit);
        integral += batch;

        /* przenies niekompletna koncowke na poczatek bufora */
        carry = avail - proc;
        memmove(buf, buf + proc, carry);
    }

    /* ostatnia linia bez znaku nowej linii */
    if (carry > 0) {
        if (buf[carry - 1] == '\r') carry--;
        buf[carry] = '\0';
        if (carry > 0) {
            total++;
            if (eigentest(buf)) { integral++; fputs(buf, out); fputc('\n', out); }
        }
    }

    double t1 = omp_get_wtime();
    if (out != stdout) fclose(out);
    free(buf); free(lines);
    fprintf(stderr, "[sito_omp] watki=%d przebadane=%lld calkowite=%lld czas=%.3fs\n",
            omp_get_max_threads(), total, integral, t1 - t0);
    return EXIT_SUCCESS;
}
