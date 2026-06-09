# Wyniki benchmarków — sito spektralne (n=16, k=46)

**Maszyna:** Windows 11, CPU 16 wątków logicznych, GPU NVIDIA RTX 3060 Ti (Ampere,
sm_86, 8 GB), gcc 13.2 (MSYS2/UCRT) `-O3 -march=native`, nvcc 13.3 + MSVC 14.42.
**Wejście:** `bench16_46_20M.g6` — 20 000 000 spójnych grafów n=16, k=46
(`geng -cq 16 46:46 | head -n 20000000`). Trafień (grafów całkowitych) w próbce: 3.
**Uwaga:** podczas pomiarów GPU wstrzymano animowaną tapetę (Wallpaper Engine),
która stale obciążała kartę w ~50% i zaburzała wyniki.

Wszystkie pomiary na **tym samym** zbiorze 20 mln grafów (przepustowość = grafy/s).

## Tabela zbiorcza (20 mln grafów)

| Wersja | Czas [s] | Przepustowość [graf/s] | Przyspieszenie |
|---|---:|---:|---:|
| `sito5` (ref. prowadzącego, sekw., long double) | 87.26 | 229 203 | 1.00× (baza) |
| `sito_seq` (moja, sekw., double + early-reject) | 35.67 | 560 759 | **2.45×** |
| `sito5_omp` (ref., OpenMP) | 10.86 | 1 841 316 | 8.03× |
| `sito_omp` (moja, OpenMP 16 wątków) | 4.37 | 4 576 162 | **19.97×** |
| `sito_cuda` (moja, GPU, blok=32) — czas całkowity | 6.26 | 3 193 814 | 13.9× |
| `sito_cuda` — **samo jądro GPU** | 1.49 | 13 458 950 | **58.7×** |
| `sito8` (ref. prowadzącego, GPU `<<<1,1>>>`/graf) | ~4305* | 4 645 | 0.02× |

\* sito8 zmierzono na 200 000 grafów (43.06 s → 4 645 graf/s) i ekstrapolowano
na 20 mln (≈72 min) — uruchomienie pełnego zbioru było niepraktyczne.

## Porównanie CUDA: moja wersja vs sito8 (cel zadania)

| Metryka | sito8 | sito_cuda (moja) | Stosunek |
|---|---:|---:|---:|
| Przepustowość [graf/s] | 4 645 | 3 193 814 | **≈688× szybciej** |
| Konfiguracja jądra | `<<<1,1>>>` na 1 graf | `<<<grid,blok>>>`, cała porcja naraz | — |
| Synchronizacja | `cudaDeviceSynchronize()` po każdym grafie | 1× na porcję 1 mln | — |
| Transfer H→D | 1 graf na raz | porcjami po 1 mln (binarna bitmapa) | — |

**Dlaczego sito8 jest tak wolny:** uruchamia jądro z jednym wątkiem (`<<<1,1>>>`)
osobno dla każdego grafu i synchronizuje po każdym. Wykorzystuje ~0,001% rdzeni GPU
i płaci pełny narzut uruchomienia+synchronizacji 20 mln razy. To celowo słaby punkt
odniesienia. Moja wersja koduje grafy binarnie (bitmapa dolnego trójkąta, układ SoA →
dostęp scalony), kopiuje porcjami po 1 mln i przetwarza tysiące grafów równolegle.

## Analiza CUDA: jądro vs narzut (prawo Amdahla)

| Składowa | Czas [s] | Udział |
|---|---:|---:|
| Samo jądro GPU (obliczenia) | 1.49 | 24 % |
| Narzut hosta: odczyt I/O + dekodowanie graph6 + transfer | 4.77 | 76 % |
| **Razem** | 6.26 | 100 % |

Jądro GPU (1.49 s) jest **szybsze niż 16 wątków CPU** (sito_omp 4.37 s). Jednak
przygotowanie danych na hoście (odczyt 20 mln linii + kodowanie do bitmap, sekwencyjnie
na 1 wątku CPU) dominuje czas całkowity. To klasyczny efekt Amdahla: gdy część
równoległa staje się bardzo szybka, część sekwencyjna (I/O) wyznacza granicę.

## Zagęszczenie pomiarów: czas vs rozmiar zbioru (punkt graniczny)

Czas całkowity [s] (z I/O), jednolita metodyka, wejście zamrożone w pliku `.g6`:

| Liczba grafów | SEQ | OMP (16w) | CUDA całość | CUDA jądro | Narzut hosta CUDA |
|---:|---:|---:|---:|---:|---:|
| 100 000 | 0.253 | 0.099 | 0.208 | 0.016 | 92.3 % |
| 500 000 | 0.930 | 0.150 | 0.316 | 0.043 | 86.4 % |
| 1 000 000 | 1.745 | 0.245 | 0.449 | 0.081 | 82.0 % |
| 5 000 000 | 8.690 | 1.002 | 1.697 | 0.384 | 77.4 % |
| 10 000 000 | 18.013 | 1.831 | 3.249 | 0.765 | 76.5 % |
| 20 000 000 | 34.874 | 3.571 | 6.201 | 1.511 | 75.6 % |

**Punkt graniczny:**
- CUDA (całość) > SEQ już od **~100 tys.** grafów (próg GPU vs 1 rdzeń CPU jest niski).
- CUDA (całość) **nie pokonuje** OpenMP w całym zakresie (OMP/CUDA ≈ 0.5–0.6×), bo
  **75–92 % czasu CUDA to serializowany I/O hosta**. Punkt przecięcia CUDA-całość z
  OpenMP leży **powyżej 20 mln** grafów.
- **Samo jądro** GPU jest **2.4–6.2× szybsze** niż 16 wątków CPU dla każdego rozmiaru ≥ 1 mln.
- Udział narzutu hosta maleje z rozmiarem (92 % → 76 %) — kierunek właściwy, ale granica
  I/O jest twarda. Aby przesunąć punkt graniczny w dół: równoległy odczyt + `cudaMemcpyAsync`
  ze strumieniami (nakładanie transferu na obliczenia).

## Sito5 vs moja wersja (sekwencyjna + OpenMP)

| Porównanie | Stosunek |
|---|---:|
| `sito_seq` vs `sito5` (sekwencyjnie) | **2.45× szybciej** |
| `sito_omp` vs `sito5_omp` (OpenMP) | **2.49× szybciej** |
| `sito_omp` vs `sito5` (równoległa vs sekw. baza) | **19.97×** |

Źródło zysku: arytmetyka `double` (SSE2) zamiast `long double` (x87, 80-bit) +
zachowane wczesne odrzucenie + lepsze I/O (blokowy `read`, podział linii in-place).

## Rozmiar bloku CUDA (double, 2 mln grafów, GPU bezczynne)

| Blok | Jądro [s] | Razem [s] |
|---:|---:|---:|
| 32 | **0.153** | 0.614 |
| 64 | 0.220 | 0.666 |
| 128 | 0.226 | 0.674 |
| 256 | 0.215 | 0.669 |
| 512 | 0.206 | 0.656 |
| 1024 | 0.212 | 0.655 |

Najlepszy **blok=32**. Jądro jest bardzo "rejestrochłonne" (każdy wątek trzyma na
stosie pięć tablic `double[17]` + macierz `double[137]`), więc małe bloki zmniejszają
nacisk na rejestry i poprawiają zajętość (occupancy). To uzasadnia, że nie należy
arbitralnie ustawiać 256 — optymalna wartość zależy od zasobów jądra.

## Skalowalność sito_omp (2 mln grafów, GPU/CPU bezczynne)

| Wątki | Czas [s] | Przyspieszenie | Efektywność |
|---:|---:|---:|---:|
| 1 | 3.291 | 1.00× | 100 % |
| 2 | 1.684 | 1.95× | 98 % |
| 4 | 0.959 | 3.43× | 86 % |
| 8 | 0.587 | 5.61× | 70 % |
| 12 | 0.461 | 7.14× | 59 % |
| 16 | 0.402 | 8.19× | 51 % |

Próg nasycenia ~8–12 wątków — koszt serializowanego I/O staje się istotny.

## Poprawność

- Zgodność z OEIS A064731 (spójne grafy całkowite n=5..9 → 3,6,7,22,24): OK dla
  `sito5`, `sito_seq`, `sito_omp`, `sito_cuda`, `sito8`.
- `sito_omp` oraz `sito_cuda` (double) dają wynik **bajt-w-bajt identyczny** z
  `sito_seq` na pełnym zbiorze (deterministyczna kolejność, ten sam algorytm
  numeryczny: Householder + bisekcja Sturma).
- Na próbce 100k wszystkie cztery wersje znalazły te same 3 grafy całkowite.

## Kluczowy wniosek metodyczny

Przewaga CUDA zależy od **rozmiaru zbioru** (amortyzacja narzutu uruchomienia i
transferu PCIe) oraz od **jakości punktu odniesienia**. Przy bardzo szybkim
algorytmie sekwencyjnym (z wczesnym odrzuceniem) i tylko 2 mln grafów narzut hosta
dominuje. Dopiero na 20 mln grafów GPU w pełni się opłaca, a wobec celowo słabego
`sito8` (`<<<1,1>>>`/graf) moja wersja jest ~688× szybsza.
