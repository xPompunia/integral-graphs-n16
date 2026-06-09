#!/usr/bin/env bash
# Build geng.exe + nauty.a without `make` (MSYS2/MinGW).
# Mirrors nauty's makefile: library objects (general MAXN) + geng.c with -DMAXN=WORDSIZE.
set -e
# uruchamiaj z dowolnego miejsca - przejdz do katalogu zrodel nauty
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/build/nauty2_8_9"
CF="-O3 -march=native"
LIBSRC="nauty nautil naugraph nausparse gtools nautinv naugroup naututil naurng schreier gtnauty gutil1 gutil2"

# labelg (etykietowanie kanoniczne) tez tu zbudujemy, jesli jeszcze nie ma
echo ">> compiling library objects"
for s in $LIBSRC; do
  echo "   cc $s.c"
  gcc $CF -c "$s.c" -o "$s.o"
done

echo ">> archiving nauty.a"
ar cr nauty.a nauty.o nautil.o naugraph.o nausparse.o gtools.o nautinv.o \
   naugroup.o naututil.o naurng.o schreier.o gtnauty.o gutil1.o gutil2.o

GTOOLSO="gtools.o naugroup.o nautinv.o gutil1.o gutil2.o gtnauty.o naututil.o"

echo ">> linking geng.exe"
gcc $CF -DMAXN=WORDSIZE -o geng.exe geng.c $GTOOLSO nauty.a

echo ">> linking labelg.exe (etykietowanie kanoniczne)"
gcc $CF -c traces.c -o traces.o
# naututil.o jest juz w $GTOOLSO - nie powtarzac (multiple definition)
gcc $CF -o labelg.exe labelg.c traces.o $GTOOLSO nauty.a

echo ">> OK: $(ls -la geng.exe labelg.exe | awk '{print $5, $9}')"
