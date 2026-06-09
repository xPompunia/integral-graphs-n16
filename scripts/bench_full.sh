#!/usr/bin/env bash
# Pelny benchmark na roznych rozmiarach zbioru: seq/OMP/CUDA + sito5/sito5_omp.
# (sito8 mierzony osobno - jest bardzo wolny). Wynik -> data/bench_full.tsv
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BIN="$ROOT/bin"; DATA="$ROOT/data"
OUT="$DATA/bench_full.tsv"
FULL="$DATA/bench16_46_50M.g6"
echo -e "size\tseq\tomp\tcuda_total\tcuda_kernel\tsito5\tsito5_omp" > "$OUT"

t(){ local s e; s=$(date +%s.%N); eval "$1" >/dev/null 2>/dev/null; e=$(date +%s.%N); awk "BEGIN{printf \"%.3f\",$e-$s}"; }

for sz in 1000000 5000000 10000000 20000000 50000000; do
  IN="$DATA/in_$sz.g6"; head -n "$sz" "$FULL" > "$IN"
  seq=$(t "$BIN/sito_seq.exe < $IN")
  omp=$(t "$BIN/sito_omp.exe < $IN")
  s=$(date +%s.%N); "$BIN/sito_cuda.exe" nul 32 < "$IN" 2>"$DATA/cs.txt"; e=$(date +%s.%N)
  cuda=$(awk "BEGIN{printf \"%.3f\",$e-$s}")
  ker=$(grep -o 'czas_jadra=[0-9.]*' "$DATA/cs.txt" | cut -d= -f2)
  s5=$(t "$BIN/sito5.exe < $IN")
  s5o=$(t "$BIN/sito5_omp.exe 100000 < $IN")
  echo -e "${sz}\t${seq}\t${omp}\t${cuda}\t${ker}\t${s5}\t${s5o}" >> "$OUT"
  echo "size=$sz seq=$seq omp=$omp cuda=$cuda(ker=$ker) sito5=$s5 sito5_omp=$s5o"
  rm -f "$IN"
done
rm -f "$DATA/cs.txt"
echo "DONE -> $OUT"
