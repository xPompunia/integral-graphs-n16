#!/usr/bin/env bash
# Pomiar czasu calkowitego (z I/O) seq/OMP/CUDA dla roznych rozmiarow zbioru.
# Wynik -> data/bench_sizes.tsv. Uruchom: bash scripts/bench_sizes.sh
# Wymaga: bin/sito_seq.exe, bin/sito_omp.exe, bin/sito_cuda.exe oraz
#         data/bench16_46_20M.g6 (pelna probka, z ktorej wycinamy mniejsze).
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BIN="$ROOT/bin"; DATA="$ROOT/data"
OUT="$DATA/bench_sizes.tsv"
FULL="$DATA/bench16_46_20M.g6"
echo -e "size\tseq_s\tomp_s\tcuda_total_s\tcuda_kernel_s" > "$OUT"

timeit() {  # $1 = polecenie (string); echo czas w s
  local s e
  s=$(date +%s.%N); eval "$1" >/dev/null 2>/dev/null; e=$(date +%s.%N)
  awk "BEGIN{printf \"%.3f\", $e-$s}"
}

for sz in 100000 500000 1000000 5000000 10000000 20000000; do
  IN="$DATA/in_${sz}.g6"
  head -n "$sz" "$FULL" > "$IN"
  seq=$(timeit "$BIN/sito_seq.exe < $IN")
  omp=$(timeit "$BIN/sito_omp.exe < $IN")
  s=$(date +%s.%N); "$BIN/sito_cuda.exe" nul 32 < "$IN" 2>"$DATA/cstats.txt"; e=$(date +%s.%N)
  cuda=$(awk "BEGIN{printf \"%.3f\", $e-$s}")
  ker=$(grep -o 'czas_jadra=[0-9.]*' "$DATA/cstats.txt" | cut -d= -f2)
  echo -e "${sz}\t${seq}\t${omp}\t${cuda}\t${ker}" >> "$OUT"
  echo "done size=$sz : seq=$seq omp=$omp cuda=$cuda ker=$ker"
  rm -f "$IN"
done
rm -f "$DATA/cstats.txt"
echo "ALL_DONE -> $OUT"
