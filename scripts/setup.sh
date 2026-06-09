#!/usr/bin/env bash
# Przygotowanie projektu od zera (po sklonowaniu repo).
# Rozpakowuje nauty z archiwum, kopiuje do niego skrypty budowania,
# buduje biblioteki nauty (geng, labelg, nautyTL) i wszystkie wersje sita.
#
# Uzycie:  bash scripts/setup.sh
set -e
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

NDIR="build/nauty2_8_9"
TARBALL="build/nauty2_8_9.tar.gz"

# 1) Rozpakuj nauty (jesli jeszcze nie rozpakowane)
if [ ! -d "$NDIR" ]; then
  echo ">> rozpakowuje nauty z $TARBALL"
  tar -xzf "$TARBALL" -C build/
  # configure generuje naglowki (nauty.h itd.) z szablonow *-h.in
  ( cd "$NDIR" && ./configure >/dev/null 2>&1 || true )
fi

# 2) Skopiuj nasze skrypty budowania nauty do katalogu nauty
cp scripts/build_geng.sh "$NDIR/" 2>/dev/null || true
cp scripts/build_tls.sh  "$NDIR/" 2>/dev/null || true

# 3) Zbuduj biblioteki nauty: geng.exe + labelg.exe + nauty.a
echo ">> buduje geng/labelg/nauty.a"
bash scripts/build_geng.sh

# 4) Zbuduj wersje thread-local: nautyTL.a + geng_lib.o (dla GEN)
echo ">> buduje nautyTL.a + geng_lib.o"
bash scripts/build_tls.sh

# 5) Zbuduj wszystkie wersje sita do bin/
echo ">> buduje wersje sita (build_all.sh)"
bash scripts/build_all.sh

echo ""
echo ">> GOTOWE. Programy w bin/. Test poprawnosci:"
echo "   ./bin/geng.exe -cq 8 | ./bin/sito_seq.exe | wc -l   # powinno dac 22"
