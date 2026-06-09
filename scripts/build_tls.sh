#!/usr/bin/env bash
# Build thread-local nauty (nautyTL.a) + geng-as-library object (geng_lib.o)
# for the integrated parallel generator+sieve (program_omp).
set -e
# uruchamiaj z dowolnego miejsca - przejdz do katalogu zrodel nauty
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/build/nauty2_8_9"
CF="-O3 -march=native -DUSE_TLS=1"
LIBSRC="nauty nautil naugraph nausparse gtools nautinv naugroup naututil naurng schreier gtnauty gutil1 gutil2"

echo ">> compiling thread-local library objects"
for s in $LIBSRC; do
  echo "   cc(TLS) $s.c"
  gcc $CF -c "$s.c" -o "tls_$s.o"
done

echo ">> archiving nautyTL.a"
ar cr nautyTL.a tls_nauty.o tls_nautil.o tls_naugraph.o tls_nausparse.o tls_gtools.o \
   tls_nautinv.o tls_naugroup.o tls_naututil.o tls_naurng.o tls_schreier.o \
   tls_gtnauty.o tls_gutil1.o tls_gutil2.o

echo ">> compiling geng.c as library (OUTPROC=process_graph, GENG_MAIN=geng_main)"
gcc $CF -DMAXN=WORDSIZE -DOUTPROC=process_graph -DGENG_MAIN=geng_main \
    -c geng.c -o geng_lib.o

echo ">> OK"
ls -la nautyTL.a geng_lib.o
