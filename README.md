# integral-graphs-n16

Generating and counting connected integral graphs of order `n=16`, size `k=46` —
sequential, OpenMP, and CUDA. Graphs are generated with **nauty (`geng`)**, then a
**spectral sieve** rejects graphs whose adjacency-matrix spectrum is not integral.
The three implementations are benchmarked against the reference codes (`sito5`,
`sito8`).

## Quick start

```bash
bash scripts/setup.sh        # unpack nauty, build the libraries + all sieve versions
./bin/geng.exe -cq 8 | ./bin/sito_seq.exe | wc -l   # sanity check: should print 22
```

`setup.sh` unpacks nauty from `build/nauty2_8_9.tar.gz`, builds `geng`/`labelg`/
`nauty.a`/`nautyTL.a`, and compiles every sieve version into `bin/`. CUDA is built
only when `nvcc` is in PATH (skipped automatically on macOS).

**Regenerating benchmark data** (large `.g6` files are not in the repo — see below):
```bash
./bin/geng.exe -cq 16 46:46 | head -n 50000000 > data/bench16_46_50M.g6
bash scripts/bench_full.sh   # benchmark seq/OMP/CUDA/sito5 -> data/bench_full.tsv
```

## Directory layout

```
.
├── README.md, SPRAWOZDANIE.txt, WYNIKI_BENCHMARK.md  # docs (report and benchmarks in Polish)
├── src/        # C/C++/CUDA sources + nauty headers + geng.c
├── scripts/    # setup, builds, benchmarks, visualization (Python)
├── data/       # small data/results in the repo; large .g6 generated locally
├── bin/        # built binaries (git-ignored — reproducible)
└── build/      # nauty2_8_9.tar.gz (in repo) + unpacked sources (ignored)
```

**What is tracked and what is not** (see `.gitignore`):
- **In the repo:** sources (`src/`), scripts (`scripts/`), docs, the nauty archive
  (`build/nauty2_8_9.tar.gz`), small result files (`data/grafy16L.graph6`,
  `data/wyniki_0*.g6`, `.tsv` tables). About 4 MB total.
- **Ignored (reproducible):** `bin/` (from `setup.sh`), the unpacked nauty tree
  (from the tarball), large `data/bench16_46_*.g6` (from `geng`).

## Sources (`src/`)

| File                | Role |
|---------------------|------|
| `integral.h`        | Shared sieve core (input: graph6 string). Householder + Sturm bisection with early rejection, `double` precision. |
| `integral_bits.h`   | Sieve core for **binary** input (bitmap). Used by CUDA and the CPU verifier. |
| `sito_seq.c`        | **Sequential** version (stdin → stdout/file). |
| `sito_omp.c`        | **OpenMP** version: block `read()`, in-place line splitting, `#pragma omp for schedule(runtime) default(none)`, per-thread private buffers, deterministic output. |
| `sito_cuda.cu`      | **CUDA** version: 1 thread = 1 graph, binary SoA encoding, configurable block size. |
| `test_cuda_logic.c` | Verifies the CUDA logic on the CPU (no GPU) — same `integral_bits.h`. |
| `program_omp.cpp`   | Integrated version: `geng` as a library (`OUTPROC` callback), parallel **generation + sieve** without a pipe. |
| `program_seq.cpp`   | Integrated sequential variant (original; note: `getline` does not work on MSYS2/UCRT). |
| `stream_omp.cpp`    | Older streaming OpenMP variant, kept for benchmark comparison. |
| `sito5.c`, `sito8.cu` | Instructor's reference codes (CPU: seq + OpenMP; GPU). |
| `geng.c`, `*.h`     | `geng` generator source and nauty headers (copied from `build/`). |

## Scripts (`scripts/`)

| File | Role |
|------|------|
| `build_all.sh`     | Builds everything into `bin/` (CUDA only when `nvcc` is in PATH). |
| `build_geng.sh`    | Builds `geng.exe` + `labelg.exe` + `nauty.a` in `build/nauty2_8_9/`. |
| `build_tls.sh`     | Builds `nautyTL.a` + `geng_lib.o` (for `program_omp`). |
| `bench_sizes.sh`   | Time vs input size (100k…20M) → `data/bench_sizes.tsv`. |
| `bench_cuda.sh`    | CUDA block-size benchmark + correctness check. |
| `g6_to_dokuwiki.py`, `todokuwiki.py` | graph6 → `graphviz` block (DokuWiki) with the spectrum (networkx + numpy). |

## Building

```bash
bash scripts/build_all.sh        # builds everything into bin/; CUDA only when nvcc is in PATH
```

### nauty (geng, labelg) — already built in `build/nauty2_8_9/`
To rebuild from scratch (on a new machine, e.g. macOS):
```bash
cd build/nauty2_8_9 && ./configure && cd ../..
bash scripts/build_geng.sh    # geng.exe + labelg.exe + nauty.a
bash scripts/build_tls.sh     # nautyTL.a + geng_lib.o (for program_omp)
```

### CUDA (tested with nvcc 13.3 + MSVC 14.42)
RTX 3060 Ti = Ampere architecture → `sm_86`. On Windows, nvcc needs the MSVC host
compiler (`cl.exe`); `build_all.sh` adds `-ccbin` pointing at the Visual Studio
directory when `cl` is not in PATH. **On macOS CUDA is skipped** (no nvcc) — the
remaining versions build normally.

**The `float` variant (`-DUSE_FLOAT`) was abandoned:** the bisection convergence
thresholds are tuned for `double`; with `float` the loop does not converge (slower
and less accurate). Sturm bisection needs double precision here.

## Running (binaries in `bin/`, data in `data/`)

```bash
# pipeline (the sieve reads graph6 from stdin):
./bin/geng.exe -cq 16 46:46 | ./bin/sito_seq.exe  data/wynik.g6
./bin/geng.exe -cq 16 46:46 | ./bin/sito_omp.exe  data/wynik.g6
./bin/geng.exe -cq 16 46:46 | ./bin/sito_cuda.exe data/wynik.g6 32

# integrated generation + sieve (no pipe):
./bin/program_omp.exe 16 46            # n=16, k=46

# OMP thread count / CUDA block size are configurable:
OMP_NUM_THREADS=8 ./bin/sito_omp.exe data/wynik.g6 < data/dane.g6
```

## Correctness

```bash
# matches OEIS A064731 (connected integral graphs): 3,6,7,22,24 for n=5..9
for n in 5 6 7 8 9; do ./bin/geng.exe -cq $n | ./bin/sito_seq.exe | wc -l; done

# canonical labelling (distinct labels => non-isomorphic graphs):
./bin/labelg.exe -q data/wyniki_0.g6

# visualization graph6 -> graphviz (DokuWiki):
while read g; do python3 scripts/g6_to_dokuwiki.py "$g"; done < data/wyniki_0.g6
```

All versions (seq, OpenMP, CUDA) produce identical counts and **byte-for-byte**
identical output.

## Results

See `WYNIKI_BENCHMARK.md` (full tables) and `SPRAWOZDANIE.txt` (DokuWiki report),
both in Polish. Summary for **20M** graphs, n=16 k=46:
- sequential: **2.45×** faster than `sito5` (`double` instead of `long double`),
- OpenMP (16 threads): **2.49×** faster than `sito5_omp`, ~20× faster than sequential `sito5`,
- CUDA: the GPU kernel is the fastest compute-wise (1.49 s), **~688×** faster than
  `sito8` (which launches `<<<1,1>>>` per graph); total time is host-I/O bound (Amdahl),
- OpenMP scaling up to 16 threads: ~8.2×; best CUDA block size: 32.
