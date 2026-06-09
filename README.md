# Generowanie spójnych grafów całkowitych (n=16, k=46) — OpenMP / CUDA

Projekt z programowania równoległego. Generujemy spójne grafy rzędu `n=16`
o rozmiarze `k=46` pakietem **nauty (`geng`)**, a następnie **sito spektralne**
odrzuca grafy, których widmo macierzy sąsiedztwa nie jest całkowite.
Trzy wersje algorytmu: **sekwencyjna**, **OpenMP** i **CUDA** — porównane
wydajnościowo z kodami prowadzącego (`sito5`, `sito8`).

## Szybki start (po sklonowaniu repo)

```bash
bash scripts/setup.sh        # rozpakuj nauty, zbuduj biblioteki + wszystkie wersje sita
./bin/geng.exe -cq 8 | ./bin/sito_seq.exe | wc -l   # test: powinno dac 22
```

`setup.sh` rozpakowuje nauty z `build/nauty2_8_9.tar.gz`, buduje `geng`/`labelg`/
`nauty.a`/`nautyTL.a` i wszystkie wersje sita do `bin/`. CUDA budowana tylko, gdy
`nvcc` jest w PATH (na macOS pomijana automatycznie).

**Regeneracja danych benchmarkowych** (duże pliki `.g6` nie są w repo — patrz niżej):
```bash
./bin/geng.exe -cq 16 46:46 | head -n 50000000 > data/bench16_46_50M.g6
bash scripts/bench_full.sh   # benchmark seq/OMP/CUDA/sito5 -> data/bench_full.tsv
```

## Struktura katalogów

```
.
├── README.md, SPRAWOZDANIE.txt, NOTATKA_DO_NAUKI.md, WYNIKI_BENCHMARK.md  # dokumentacja
├── src/        # źródła C/C++/CUDA + nagłówki nauty + geng.c
├── scripts/    # setup, budowanie, benchmarki, wizualizacja (Python)
├── data/       # małe dane/wyniki w repo; duże .g6 generowane lokalnie
├── bin/        # zbudowane programy (ignorowane przez git — odtwarzalne)
└── build/      # nauty2_8_9.tar.gz (w repo) + rozpakowane źródła (ignorowane)
```

**Co jest w repozytorium, a co nie** (patrz `.gitignore`):
- **W repo:** źródła (`src/`), skrypty (`scripts/`), dokumentacja, archiwum nauty
  (`build/nauty2_8_9.tar.gz`), małe pliki wynikowe (`data/grafy16L.graph6`,
  `data/wyniki_0*.g6`, tabele `.tsv`). Całość ~4 MB.
- **Ignorowane (odtwarzalne):** `bin/` (z `setup.sh`), rozpakowane nauty (z tarballa),
  duże `data/bench16_46_*.g6` (z `geng`).

## Pliki źródłowe (`src/`)

| Plik                | Rola |
|---------------------|------|
| `integral.h`        | Wspólny rdzeń sita (wejście: napis graph6). Householder + bisekcja Sturma z wczesnym odrzuceniem, typ `double`. |
| `integral_bits.h`   | Rdzeń sita dla danych **binarnych** (bitmapa). Używany przez CUDA i przez weryfikator CPU. |
| `sito_seq.c`        | Wersja **sekwencyjna** (stdin → stdout/plik). |
| `sito_omp.c`        | Wersja **OpenMP**: blokowy `read()`, podział linii in-place, `#pragma omp for schedule(runtime) default(none)`, prywatne bufory na wątek, deterministyczny wypis. |
| `sito_cuda.cu`      | Wersja **CUDA**: 1 wątek = 1 graf, kodowanie binarne SoA, parametr rozmiaru bloku. |
| `test_cuda_logic.c` | Weryfikacja logiki CUDA na CPU (bez GPU) — ten sam `integral_bits.h`. |
| `program_omp.cpp`   | Wersja zintegrowana: `geng` jako biblioteka (callback `OUTPROC`), zrównoleglona **generacja + sito** bez potoku. |
| `program_seq.cpp`   | Wariant sekwencyjny zintegrowany (oryginał; uwaga: `getline` nie działa na MSYS2/UCRT). |
| `sito5.c`, `sito8.cu` | Kody referencyjne prowadzącego (CPU: sekw.+OpenMP; GPU). |
| `geng.c`, `*.h`     | Źródło generatora `geng` i nagłówki nauty (kopia z `build/`). |

## Skrypty (`scripts/`)

| Plik | Rola |
|------|------|
| `build_all.sh`     | Buduje wszystko do `bin/` (CUDA tylko gdy `nvcc` w PATH). |
| `build_geng.sh`    | Buduje `geng.exe` + `labelg.exe` + `nauty.a` w `build/nauty2_8_9/`. |
| `build_tls.sh`     | Buduje `nautyTL.a` + `geng_lib.o` (dla `program_omp`). |
| `bench_sizes.sh`   | Pomiar czas vs rozmiar (100k…20M) → `data/bench_sizes.tsv`. |
| `bench_cuda.sh`    | Benchmark rozmiaru bloku CUDA + poprawność. |
| `g6_to_dokuwiki.py`, `todokuwiki.py` | Konwersja graph6 → blok `graphviz` (DokuWiki) wraz z widmem (networkx + numpy). |

## Budowanie

```bash
bash scripts/build_all.sh        # buduje wszystko do bin/; CUDA tylko gdy nvcc w PATH
```

### nauty (geng, labelg) — już zbudowane w `build/nauty2_8_9/`
Gdyby trzeba odtworzyć (na nowej maszynie, np. macOS):
```bash
cd build/nauty2_8_9 && ./configure && cd ../..
bash scripts/build_geng.sh    # geng.exe + labelg.exe + nauty.a
bash scripts/build_tls.sh     # nautyTL.a + geng_lib.o (dla program_omp)
```

### CUDA (zainstalowane: nvcc 13.3 + MSVC 14.42)
RTX 3060 Ti = architektura **Ampere → `sm_86`**. Na Windows nvcc wymaga hosta MSVC
(`cl.exe`); `build_all.sh` sam dodaje `-ccbin` do katalogu Visual Studio, gdy `cl`
nie jest w PATH. **Na macOS CUDA jest pomijana** (brak nvcc) — pozostałe wersje
budują się normalnie.

**Wariant `float` (`-DUSE_FLOAT`) porzucony:** progi zbieżności bisekcji są dobrane pod
`double`; we `float` pętla się nie zbiega (wolniejsza i mniej dokładna). Algorytm
bisekcji Sturma wymaga podwójnej precyzji.

## Uruchamianie (programy w `bin/`, dane w `data/`)

```bash
# potokowo (sito czyta graph6 ze stdin):
./bin/geng.exe -cq 16 46:46 | ./bin/sito_seq.exe  data/wynik.g6
./bin/geng.exe -cq 16 46:46 | ./bin/sito_omp.exe  data/wynik.g6
./bin/geng.exe -cq 16 46:46 | ./bin/sito_cuda.exe data/wynik.g6 32

# zintegrowana generacja + sito (bez potoku):
./bin/program_omp.exe 16 46            # n=16, k=46

# liczba wątków OMP / rozmiar bloku CUDA są zmienialne:
OMP_NUM_THREADS=8 ./bin/sito_omp.exe data/wynik.g6 < data/dane.g6
```

## Weryfikacja poprawności

```bash
# zgodność z OEIS A064731 (spójne grafy całkowite): 3,6,7,22,24 dla n=5..9
for n in 5 6 7 8 9; do ./bin/geng.exe -cq $n | ./bin/sito_seq.exe | wc -l; done

# etykietowanie kanoniczne (różne etykiety ⇒ grafy nieizomorficzne):
./bin/labelg.exe -q data/wyniki_0.g6

# wizualizacja graph6 → graphviz (DokuWiki):
while read g; do python3 scripts/g6_to_dokuwiki.py "$g"; done < data/wyniki_0.g6
```
Wszystkie wersje (seq, OpenMP, CUDA) dają identyczne liczby oraz wynik
**bajt-w-bajt** zgodny między sobą.

## Wyniki

Zobacz `WYNIKI_BENCHMARK.md` (pełne tabele) oraz `SPRAWOZDANIE.txt` (sprawozdanie
DokuWiki). Skrót na **20 mln** grafów n=16 k=46:
- sekwencyjna: **2,45×** szybsza od `sito5` (`double` zamiast `long double`),
- OpenMP (16 wątków): **2,49×** szybsza od `sito5_omp`, ~20× od `sito5` sekwencyjnego,
- CUDA: jądro GPU najszybsze obliczeniowo (1,49 s), **~688×** szybsza od `sito8`
  (które używa `<<<1,1>>>` na graf); czas całkowity ograniczony przez I/O hosta (Amdahl),
- skalowalność OpenMP do 16 wątków: ~8,2×; najlepszy rozmiar bloku CUDA: 32.
