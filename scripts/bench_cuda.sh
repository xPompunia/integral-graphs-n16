#!/usr/bin/env bash
# Benchmark CUDA: rozmiar bloku (czas jadra i calkowity) + poprawnosc.
# Wynik -> data/bench_cuda_result.txt. Uruchom: bash scripts/bench_cuda.sh
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BIN="$ROOT/bin"; DATA="$ROOT/data"
RES="$DATA/bench_cuda_result.txt"
IN="$DATA/bench16_46.g6"     # 2 mln grafow
: > "$RES"

echo "=== poprawnosc: CUDA vs SEQ na 2M ===" >> "$RES"
"$BIN/sito_seq.exe"  < "$IN" 2>/dev/null > "$DATA/seq_out.txt"
"$BIN/sito_cuda.exe" "$DATA/cuda_out.txt" 32 < "$IN" 2>/dev/null
if diff -q "$DATA/seq_out.txt" "$DATA/cuda_out.txt" >/dev/null 2>&1; then
  echo "CUDA (double): IDENTYCZNE z SEQ ($(wc -l < "$DATA/cuda_out.txt") grafow)" >> "$RES"
else
  echo "CUDA: ROZNICE z SEQ!" >> "$RES"
fi

echo "" >> "$RES"
echo "=== double: rozmiar bloku (2 mln grafow) ===" >> "$RES"
for b in 32 64 128 256 512 1024; do
  "$BIN/sito_cuda.exe" nul $b < "$IN" 2> "$DATA/tmp_stats.txt"
  k=$(grep -o 'czas_jadra=[0-9.]*' "$DATA/tmp_stats.txt" | cut -d= -f2)
  t=$(grep -o 'czas_calk=[0-9.]*'  "$DATA/tmp_stats.txt" | cut -d= -f2)
  printf "  blok=%-5s kernel=%ss  total=%ss\n" "$b" "$k" "$t" >> "$RES"
done
rm -f "$DATA/seq_out.txt" "$DATA/cuda_out.txt" "$DATA/tmp_stats.txt"
echo "" >> "$RES"; echo "DONE -> $RES" >> "$RES"
cat "$RES"
