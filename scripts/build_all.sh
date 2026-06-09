#!/usr/bin/env bash
# Buduje wszystkie wersje sita. Uruchom z dowolnego miejsca:  bash scripts/build_all.sh
# Wymaga: gcc (MSYS2/MinGW na Windows lub clang/gcc na macOS/Linux).
# CUDA budowane tylko jesli nvcc jest w PATH (na macOS pomijane).
set -e

# katalog glowny projektu = rodzic katalogu tego skryptu
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC="$ROOT/src"
BIN="$ROOT/bin"
NDIR="$ROOT/build/nauty2_8_9"
mkdir -p "$BIN"
CF="-O3 -march=native"
INC="-I$SRC"

echo "== sito5 (referencja prowadzacego: sekwencyjna + OpenMP) =="
gcc $CF $INC          "$SRC/sito5.c" -o "$BIN/sito5.exe"     -lm
gcc $CF $INC -fopenmp "$SRC/sito5.c" -o "$BIN/sito5_omp.exe" -lm

echo "== moje wersje: sekwencyjna + OpenMP =="
gcc $CF $INC          "$SRC/sito_seq.c" -o "$BIN/sito_seq.exe" -lm
gcc $CF $INC -fopenmp "$SRC/sito_omp.c" -o "$BIN/sito_omp.exe" -lm

echo "== weryfikator logiki CUDA na CPU =="
gcc $CF $INC          "$SRC/test_cuda_logic.c" -o "$BIN/test_cuda_logic.exe" -lm

echo "== wersja zintegrowana (geng jako biblioteka, OpenMP) =="
if [ -f "$NDIR/geng_lib.o" ] && [ -f "$NDIR/nautyTL.a" ]; then
  g++ $CF $INC -fopenmp -DUSE_TLS=1 -DMAXN=WORDSIZE \
      "$SRC/program_omp.cpp" "$NDIR/geng_lib.o" "$NDIR/nautyTL.a" -o "$BIN/program_omp.exe"
else
  echo "   (pomijam program_omp - brak geng_lib.o/nautyTL.a; uruchom scripts/build_tls.sh)"
fi

echo "== narzedzia nauty (geng, labelg) =="
if [ -f "$NDIR/geng.exe" ]; then cp "$NDIR/geng.exe" "$BIN/"; fi
if [ -f "$NDIR/labelg.exe" ]; then cp "$NDIR/labelg.exe" "$BIN/"; fi

echo "== CUDA =="
if command -v nvcc >/dev/null 2>&1; then
  ARCH="${CUDA_ARCH:-sm_86}"   # RTX 3060 Ti = Ampere
  # Na Windows nvcc potrzebuje hosta MSVC (cl.exe). Jesli cl nie jest w PATH,
  # wskaz katalog Hostx64/x64 zmienna CCBIN; ponizej domyslna sciezka VS 2022.
  CCBIN_DEFAULT="/c/Program Files/Microsoft Visual Studio/2022/Community/VC/Tools/MSVC/14.42.34433/bin/Hostx64/x64"
  CCB=()                                   # tablica - poprawne cytowanie sciezek ze spacjami
  if ! command -v cl >/dev/null 2>&1; then
    CLDIR="${CCBIN:-$CCBIN_DEFAULT}"
    export PATH="$CLDIR:$PATH"             # nvcc musi znalezc cl.exe w PATH
    CCB=(-ccbin "$CLDIR")
  fi
  # sito_cuda uzywa OpenMP do dekodowania graph6 na hoscie. Pod MSVC flaga to
  # '-openmp'. PROBLEM: powloki MSYS2/Git-Bash zamieniaja argumenty z '/' na
  # sciezki Windows; '-openmp' (z myslnikiem) tego unika, a '-ccbin' ze sciezka
  # nadal jest poprawnie konwertowany. Dlatego uzywamy '-Xcompiler -openmp'.
  if nvcc -O3 -arch=$ARCH $INC "${CCB[@]}" -Xcompiler -openmp "$SRC/sito_cuda.cu" -o "$BIN/sito_cuda.exe" \
     && nvcc -O3 -arch=$ARCH $INC "${CCB[@]}"                 "$SRC/sito8.cu"     -o "$BIN/sito8.exe"; then
    echo "   zbudowano sito_cuda.exe (moja, double + OpenMP host) i sito8.exe (referencja)"
  else
    echo "   (BLAD budowania CUDA - sprawdz sciezke do MSVC; reszta wersji zbudowana)"
  fi
else
  echo "   (pomijam CUDA - brak nvcc w PATH; na macOS to normalne)"
fi

echo "== GOTOWE =="
ls -la "$BIN"
