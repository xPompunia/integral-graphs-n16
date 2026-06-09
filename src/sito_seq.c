/**
@file sito_seq.c
@brief Sito spektralne - wersja sekwencyjna (zoptymalizowana).

Czyta grafy w formacie graph6 ze standardowego wejscia i wypisuje na stdout
te, ktorych widmo (zbior wartosci wlasnych macierzy sasiedztwa) jest calkowite.

Roznice wzgledem sito5 prowadzacego:
  * wczesne odrzucenie zachowane (jak w sito5),
  * arytmetyka `double` zamiast `long double` (szybsza, zgodna z CUDA),
  * wspolny rdzen (integral.h) uzywany takze przez wersje OpenMP i CUDA.

Kompilacja:
  gcc -O3 -march=native sito_seq.c -o sito_seq -lm

Uruchomienie:
  ./geng -cq 16 46:46 | ./sito_seq            # wypis na stdout
  ./geng -cq 16 46:46 | ./sito_seq wynik.g6   # zapis do pliku
*/
#include <stdio.h>
#include <stdlib.h>
#include "integral.h"

#define BUFSIZE 1024

int main(int argc, char *argv[])
{
    FILE *out = stdout;
    if (argc > 1) {
        out = fopen(argv[1], "w");
        if (!out) { perror(argv[1]); return EXIT_FAILURE; }
    }

    char buf[BUFSIZE];
    long long total = 0, integral = 0;

    while (fgets(buf, BUFSIZE, stdin)) {
        if (buf[0] == '\n' || buf[0] == '\0') continue;
        total++;
        if (eigentest(buf)) { integral++; fputs(buf, out); }
    }

    if (out != stdout) fclose(out);
    fprintf(stderr, "[sito_seq] przebadane=%lld calkowite=%lld\n", total, integral);
    return EXIT_SUCCESS;
}
