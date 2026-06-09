# Notatka do nauki — projekt: spójne grafy całkowite (OpenMP / CUDA)

Dokument do **samodzielnej nauki i obrony projektu**. Tłumaczy wspólną bazę
algorytmiczną, a potem szczegółowo każdą wersję: jakie dyrektywy, jakie parametry,
co dokładnie zmieniono w kodzie, gdzie jest zrównoleglenie i **dlaczego** takie
decyzje. Na końcu jest sekcja „przewidywane pytania + odpowiedzi".

---

## 0. Problem w jednym zdaniu

Generujemy wszystkie spójne grafy rzędu $n=16$ o rozmiarze $k=46$ (pakiet **nauty**,
program `geng`), a następnie **sito spektralne** odrzuca te, których widmo (zbiór
wartości własnych macierzy sąsiedztwa) **nie jest całkowite**. Graf nazywamy
*całkowitym*, gdy wszystkie wartości własne są liczbami całkowitymi.

Trzy wersje sita do porównania: **sekwencyjna (SEQ)**, **OpenMP (OMP)**, **CUDA**,
plus wariant zintegrowany **GEN** (geng wkompilowany jako biblioteka). Punkty
odniesienia prowadzącego: `sito5` (CPU) i `sito8` (GPU).

---

## 1. WSPÓLNA BAZA — to samo we wszystkich wersjach

### 1.1. Format wejścia: graph6

`geng` wypisuje grafy w formacie tekstowym **graph6** (rozszerzenie `.g6`).
Jedna linia = jeden graf. Kodowanie:
- **Pierwszy bajt** koduje liczbę wierzchołków: `n = bajt[0] - 63`. Dla n=16 → bajt `'O'` (79 - 63 = 16).
- **Kolejne bajty** kodują **górny trójkąt** macierzy sąsiedztwa, po 6 bitów na bajt
  (do każdego dodaje się 63, żeby był drukowalny ASCII). Bity czyta się od najstarszego.

Dla n=16 linia graph6 ma **21 znaków** (1 bajt na n + 20 bajtów na $\binom{16}{2}=120$
bitów: $\lceil 120/6 \rceil = 20$).

### 1.2. Algorytm sprawdzania całkowitości (rdzeń obliczeniowy)

To jest **serce projektu**, identyczne we wszystkich wersjach. Składa się z trzech kroków:

**Krok 1 — Redukcja Householdera do postaci trójprzekątniowej.**
Macierz sąsiedztwa jest symetryczna. Householder przekształca ją (zachowując widmo!)
do macierzy **trójprzekątniowej** (niezerowe tylko: przekątna `d[]` i poddiagonala `e[]`).
To standardowy pierwszy krok większości solverów wartości własnych — redukuje problem
$\mathcal{O}(n^3)$ do prostej macierzy, na której łatwo liczyć.

**Krok 2 — Bisekcja oparta na ciągach Sturma.**
Mając macierz trójprzekątniową, liczbę wartości własnych mniejszych od dowolnego $t$
można policzyć w $\mathcal{O}(n)$ (liczba zmian znaku w ciągu Sturma — w kodzie to
licznik `L1`). Bisekcja zacieśnia przedział $[s, w]$ wokół każdej kolejnej wartości
własnej, aż przedział jest dostatecznie wąski.

**Krok 3 — WCZESNE ODRZUCENIE (kluczowa optymalizacja).**
W trakcie bisekcji, **gdy tylko** przedział $[s,w]$ nie może już zawierać liczby
całkowitej (`floor(w + tol) < s - tol`), graf jest **natychmiast** odrzucany — bez
liczenia pozostałych wartości własnych:

```c
while (w - s > 2.91e-16 * (fabs(s) + fabs(w)) + eps) {
    if (floor(w + INT_TOL) < s - INT_TOL) return 0;  // brak liczby calk. w przedziale
    ...
}
```

**Dlaczego to ważne:** dla k=46 ogromna większość grafów NIE jest całkowita i odpada
po sprawdzeniu pierwszej–drugiej wartości własnej. Wczesne odrzucenie sprawia, że
średni czas na graf jest dużo mniejszy niż pełne $\mathcal{O}(n^3)$. To samo robi `sito5`.

### 1.3. Reprezentacja macierzy — spakowana dolnotrójkątna

Macierz przechowujemy jako **spakowaną dolnotrójkątną** tablicę `double a[]`
(indeksowanie od 1, jak w oryginalnym kodzie Pascala A. Marciniaka). Dla n=16 to
`double a[137]` ($\binom{16}{2} + 16 + 1$). Oszczędza pamięć (połowa macierzy +
przekątna) i jest cache-friendly.

### 1.4. Typ `double` — wspólny mianownik

**Wszystkie wersje używają `double`** (64-bit, IEEE 754). To świadoma decyzja:
- `sito5` używa `long double` (80-bit x87) — **wolny** na nowoczesnym x86.
- `double` liczy się na jednostkach SSE2 — szybciej, a precyzja w zupełności wystarcza
  (tolerancja `INT_TOL = 1e-4`).
- **`long double` NIE istnieje na GPU** — gdyby SEQ używał `long double`, nie dałoby
  się współdzielić rdzenia z CUDA. `double` pozwala mieć **jeden algorytm** na CPU i GPU.

### 1.5. Dwa pliki nagłówkowe z rdzeniem

| Plik | Wejście | Używany przez |
|---|---|---|
| `integral.h` → `eigentest(napis)` | napis graph6 | SEQ, OMP |
| `integral_bits.h` → `eigentest_bits(bitmapa)` | bitmapa (słowa 32-bit) | CUDA (+ test CPU) |

**Algorytm numeryczny jest identyczny** — różni się tylko *źródło danych*: `integral.h`
dekoduje graf z napisu graph6 w locie, `integral_bits.h` odtwarza go z binarnej bitmapy.
Powód rozdzielenia: GPU dostaje dane binarnie (patrz sekcja CUDA), więc potrzebuje
wersji czytającej z bitmapy. Dzięki wspólnemu nagłówkowi **wynik GPU jest bajt-w-bajt
identyczny z CPU**.

W `integral_bits.h` jest makro `IB_QUAL`:
```c
#ifdef __CUDACC__
#define IB_QUAL static inline __host__ __device__   // kompilacja przez nvcc: dziala na CPU i GPU
#else
#define IB_QUAL static inline                        // zwykly gcc: tylko CPU (do testu)
#endif
```
To pozwala skompilować **tę samą funkcję** raz na GPU (w jądrze) i raz na CPU
(weryfikator `test_cuda_logic`), bez duplikowania kodu.

### 1.6. Weryfikacja poprawności (wspólna)

- **OEIS A064731** (liczba spójnych grafów całkowitych): dla n=5..9 musi wyjść
  3, 6, 7, 22, 24. Wszystkie wersje to spełniają.
- **Bajt-w-bajt**: wyjście OMP i CUDA jest identyczne z SEQ na pełnej próbce.
- **Zbiór `grafy16L.graph6`** (375 znanych grafów całkowitych n=16): sito akceptuje
  wszystkie 375 (zero fałszywych odrzuceń).

---

## 2. WERSJA SEKWENCYJNA (SEQ) — `sito_seq.c`

### 2.1. Co robi
Najprostsza pętla: czyta linie ze `stdin` przez `fgets`, dla każdej woła `eigentest`,
i jeśli graf jest całkowity — wypisuje go.

```c
while (fgets(buf, BUFSIZE, stdin)) {
    if (buf[0] == '\n' || buf[0] == '\0') continue;
    total++;
    if (eigentest(buf)) { integral++; fputs(buf, out); }
}
```

### 2.2. Co zmieniono względem `sito5`
- **`double` zamiast `long double`** — to jest cała różnica algorytmiczna. Ten sam
  algorytm, ta sama logika wczesnego odrzucenia, tylko szybszy typ.

### 2.3. Wynik
- **2,48× szybsza od `sito5`** (50 mln: 86,7 s vs 214,8 s), przy **identycznych** wynikach.
- To pokazuje czystą cenę `long double` vs `double` (bo reszta kodu jest taka sama).

### 2.4. Po co w ogóle wersja sekwencyjna
Jest **bazą odniesienia** (przyspieszenie liczymy względem niej) i **wzorcem
poprawności** — OMP i CUDA muszą dać dokładnie to samo wyjście.

---

## 3. WERSJA OpenMP (OMP) — `sito_omp.c`

### 3.1. Główna idea zrównoleglenia
**Dekompozycja danych:** jeden graf = jedno niezależne zadanie. Grafy są od siebie
całkowicie niezależne (sprawdzenie jednego nie zależy od innych) — to **idealnie
równoległy** problem (ang. *embarrassingly parallel*). Dzielimy strumień na porcje
i każdą porcję przetwarzamy równolegle.

### 3.2. Dyrektywy OpenMP — dokładnie

```c
#pragma omp parallel for schedule(runtime) \
        default(none) shared(lines, hit, nlines) reduction(+:batch)
for (long long gi = 0; gi < (long long)nlines; gi++) {
    if (eigentest(lines[gi])) { hit[gi] = 1; batch++; }
}
```

Rozbiór każdej klauzuli i **dlaczego**:

- **`parallel for`** — tworzy zespół wątków i rozdziela iteracje pętli między nie.
  Jeden graf = jedna iteracja = jedno zadanie.

- **`schedule(runtime)`** — sposób przydziału iteracji do wątków sterowany zmienną
  środowiskową `OMP_SCHEDULE` (w kodzie domyślnie ustawiamy `guided`). **Dlaczego nie
  `static`?** Bo przez **wczesne odrzucenie** czas pojedynczego grafu jest bardzo
  zmienny — graf odrzucony po 1. wartości własnej liczy się ~10× krócej niż graf
  całkowity (pełne widmo). Przy `static` (równe bloki z góry) jeden wątek mógłby
  dostać same „ciężkie" grafy i reszta by na niego czekała. `guided`/`dynamic`
  **równoważy obciążenie** — wątek bierze nową porcję, gdy skończy poprzednią.

- **`default(none)`** — **wymóg prowadzącego**. Zmusza do jawnego zadeklarowania
  każdej zmiennej jako `shared` lub `private`. Zapobiega przypadkowym błędom
  współdzielenia (najczęstsze źródło bugów w OpenMP). Bez tego kompilator domyślnie
  współdzieli — łatwo o wyścig.

- **`shared(lines, hit, nlines)`** — te zmienne są **wspólne** dla wątków:
  - `lines` — tablica wskaźników na grafy (tylko do odczytu w pętli),
  - `hit` — tablica wyników; **każdy wątek pisze do INNEJ komórki** `hit[gi]` (brak konfliktu),
  - `nlines` — liczba grafów (tylko odczyt).

- **`reduction(+:batch)`** — bezpieczne sumowanie licznika trafień. Każdy wątek ma
  swoją prywatną kopię `batch`, a na końcu OpenMP sumuje je atomowo. Brak wyścigu,
  brak `atomic` w pętli gorącej.

- **Zmienne lokalne `eigentest`** — wszystkie tablice solvera (`d`, `e`, `a`...) są
  **lokalne w funkcji** = na stosie każdego wątku. To daje **automatyczną prywatność**
  i **reentrancję** (funkcja jest bezpieczna wątkowo bez żadnych dodatkowych zabiegów).
  Kluczowe: każdy wątek ma swój komplet struktur, brak fałszywego współdzielenia
  (*false sharing*).

### 3.3. Optymalizacja I/O — blokowy `read()` zamiast `fgets`

To druga warstwa optymalizacji (oprócz `double`):

```c
#define CHUNK (16 * 1024 * 1024)   // 16 MiB
while ((r = READ(0, buf + carry, CHUNK)) > 0) {
    // ... podzial na linie in-place ...
}
```

- **`read()` zamiast `fgets()`**: jedno wywołanie systemowe pobiera 16 MiB naraz,
  zamiast wołać `fgets` raz na linię (miliony wywołań). Mniej narzutu syscalli.
- **Podział linii in-place**: zamiast kopiować każdą linię do osobnego bufora,
  wstawiamy `'\0'` w miejsce `'\n'` i zapamiętujemy **wskaźniki** do początków linii
  w tablicy `lines[]`. Zero kopiowania napisów.
- **`carry`**: ostatnia (niekompletna) linia na granicy bloku jest przenoszona na
  początek bufora i sklejana z następnym blokiem.
- **`_setmode(0, _O_BINARY)`** (Windows): wyłącza translację `\r\n`→`\n`, żeby `read`
  dał surowe bajty (inaczej liczby bajtów by się nie zgadzały).

### 3.4. Wypis bez sekcji krytycznej w pętli gorącej

**To ważny szczegół projektowy.** Naiwne podejście (jak w `sito5_omp`) drukuje trafienie
przez `printf` **z wnętrza** równoległej pętli → to serializuje wątki na `stdout`
(albo wymaga `#pragma omp critical`, co też serializuje). My robimy inaczej:

```c
// 1) w pętli równoległej: tylko zapis 1/0 do rozłącznych komórek
for (long long gi = 0; gi < nlines; gi++)
    if (eigentest(lines[gi])) { hit[gi] = 1; batch++; }

// 2) PO pętli, szeregowo: wypis trafień w kolejności
for (size_t gi = 0; gi < nlines; gi++)
    if (hit[gi]) { fputs(lines[gi], out); fputc('\n', out); }
```

Korzyści:
- **Brak wyścigu** — każdy wątek pisze do innej komórki `hit[gi]` (izolacja indeksów).
- **Brak sekcji krytycznej** w pętli gorącej → wątki się nie blokują.
- **Determinizm** — wypis po pętli zachowuje kolejność wejścia, więc wynik jest
  **bajt-w-bajt identyczny** z wersją sekwencyjną (łatwa weryfikacja poprawności).

### 3.5. Skalowalność (zmierzona, próbka 2 mln)

| Wątki | 1 | 2 | 4 | 8 | 12 | 16 |
|---|---|---|---|---|---|---|
| Czas [s] | 3,29 | 1,68 | 0,96 | 0,59 | 0,46 | 0,40 |
| Przyspieszenie | 1,0× | 1,95× | 3,43× | 5,61× | 7,14× | 8,19× |
| Efektywność | 100% | 98% | 86% | 70% | 59% | 51% |

**Interpretacja (ważne do obrony):**
- Do 4 wątków efektywność ~90% (prawie liniowo).
- Powyżej 8 wątków **nasycenie** — przyspieszenie rośnie wolno. Powód: **serializowane
  I/O** (odczyt + podział linii + wypis robi 1 wątek) staje się istotną częścią czasu,
  gdy obliczenie jest już szybkie. To **prawo Amdahla**: część sekwencyjna ogranicza
  maksymalne przyspieszenie.
- CPU ma 10 rdzeni fizycznych / 16 logicznych (HyperThreading) — powyżej 10 wątków
  działają już rdzenie logiczne, które dają mniejszy zysk.

### 3.6. Wynik
- **24,6× szybsza od `sito5`** (50 mln: 8,7 s vs 214,8 s). Na ten zysk składa się:
  ~2,5× z `double` + ~10× ze zrównoleglenia (16 wątków).

---

## 4. WERSJA CUDA — `sito_cuda.cu`

### 4.1. Model obliczeń: 1 wątek GPU = 1 graf

GPU ma tysiące rdzeni, ale **pojedynczy rdzeń jest wolny** (zegar ~1,4 GHz vs ~4,9 GHz
CPU). Siła GPU to **szerokość** (masowa równoległość), nie szybkość pojedynczego wątku.
Dlatego: każdy wątek CUDA przetwarza **jeden cały graf** (uruchamia pełny solver),
a tysiące wątków robi to **jednocześnie**. To zalecany przez prowadzącego model dla
problemów grafowych: „wiele instancji jednocześnie".

### 4.2. Kodowanie binarne grafu (kluczowe dla GPU)

Zamiast przekazywać napisy graph6 na GPU, kodujemy graf **binarnie** — dolnotrójkątna
macierz sąsiedztwa jako bity spakowane w słowa 32-bit. Dla n=16: 120 bitów = 4 słowa.

```c
// HOST: graph6 -> bity dolnego trojkata
static inline void ib_g6_to_bits(const char *buf, int n, uint32_t *out, int nwords) {
    for (int w = 0; w < nwords; w++) out[w] = 0u;
    int bitg = 32, poz = 1, bitidx = 0;
    for (int i = 1; i < n; i++)
        for (int j = 0; j < i; j++) {
            if (bitg == 0) { bitg = 32; poz++; }
            if ((buf[poz] - 63) & bitg) out[bitidx >> 5] |= (1u << (bitidx & 31));
            bitidx++; bitg >>= 1;
        }
}
```

**Dlaczego binarnie:** mniej danych do transferu przez PCIe (4 słowa zamiast 21 bajtów
napisu) i wygodny układ pamięci na GPU (patrz SoA niżej).

### 4.3. Układ SoA (Structure of Arrays) — dostęp scalony

To **najważniejsza optymalizacja pamięci na GPU.** Bity grafu o numerze `idx` leżą
w pamięci urządzenia pod adresem `d_bits[w * num + idx]`, gdzie `w` to numer słowa.

- **AoS (Array of Structures, źle):** dane grafu 0, potem grafu 1... → sąsiednie wątki
  (idx=0, 1, 2...) czytałyby odległe adresy.
- **SoA (Structure of Arrays, dobrze):** słowo 0 wszystkich grafów, potem słowo 1...
  → wątki o sąsiednich `idx` czytają **sąsiednie adresy**.

```c
// w jadrze: wątek idx czyta swoje słowa
for (int w = 0; w < nwords; w++)
    bits[w] = d_bits[(size_t)w * num + idx];   // dostep scalony (coalesced)
```

**Dostęp scalony (coalesced memory access):** gdy 32 wątki warpu czytają 32 sąsiednie
adresy, GPU łączy to w **jedną** transakcję pamięci zamiast 32 osobnych. To może dać
nawet kilkukrotne przyspieszenie odczytu z pamięci globalnej. To jest właśnie powód,
dla którego dane są w układzie SoA, a nie AoS.

### 4.4. Jądro (kernel) — dyrektywy i indeksowanie

```c
__global__ void sieve_kernel(const uint32_t *__restrict__ d_bits,
                             int *__restrict__ d_res,
                             int n, int nwords, int num)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;   // globalny numer wątku
    if (idx >= num) return;                            // straż: nadmiarowe wątki nic nie robią

    uint32_t bits[IB_MAXWORDS];
    for (int w = 0; w < nwords; w++)
        bits[w] = d_bits[(size_t)w * num + idx];       // SoA, dostep scalony

    d_res[idx] = eigentest_bits(bits, n);              // ten sam solver co na CPU
}
```

- **`__global__`** — funkcja-jądro, wołana z hosta, wykonywana na GPU.
- **`blockIdx.x * blockDim.x + threadIdx.x`** — standardowy wzór na globalny indeks
  wątku w siatce 1D. `blockDim.x` = rozmiar bloku (liczba wątków w bloku).
- **`if (idx >= num) return`** — siatka jest zaokrąglana w górę do wielokrotności
  rozmiaru bloku, więc ostatnie wątki mogą wykraczać poza dane → ta straż je wyłącza.
- **`__restrict__`** — obietnica, że wskaźniki nie nakładają się (nie aliasują).
  Pozwala kompilatorowi agresywniej optymalizować (trzymać wartości w rejestrach).
- **`uint32_t bits[IB_MAXWORDS]`** — wątek kopiuje swój graf (4 słowa) do małej tablicy
  prywatnej, potem liczy. Solver `eigentest_bits` jest dokładnie ten sam co na CPU
  (duże tablice solvera trafiają do pamięci lokalnej — patrz 4.6).

### 4.5. Wynik: „tablica flag 1/0; trafienia wypisywane na hoście"

To opisuje **jak GPU zwraca wynik** — i jest świadomą decyzją projektową (podział
pracy CPU↔GPU). Warto to dobrze rozumieć do obrony.

**„Tablica flag 1/0":** GPU **nie wypisuje grafów**. Każdy wątek, po sprawdzeniu swojego
grafu, zapisuje tylko **jedną liczbę: 1 albo 0** — do tablicy wyników `d_res`, pod swój
własny indeks:
```c
// w jadrze (GPU): watek idx zapisuje wynik dla grafu idx
d_res[idx] = eigentest_bits(bits, n);   // 1 = calkowity, 0 = nie
```
Dla porcji np. 4 mln grafów GPU produkuje tablicę 4 mln liczb:
```
graf:   0  1  2  3  4  5  6  ...
d_res:  0  0  0  1  0  0  0  ...   (graf nr 3 okazal sie calkowity)
```
To jest surowa odpowiedź „tak/nie" dla każdego grafu — bez żadnego tekstu.

**„Trafienia wypisywane na hoście":** GPU pracował na **bitmapach binarnych** — nie ma
już oryginalnych napisów graph6. Dlatego po obliczeniu:
1. Kopiujemy tablicę flag z GPU na CPU (D→H):
```c
cudaMemcpy(h_res, d_res, count * sizeof(int), cudaMemcpyDeviceToHost);
```
2. Host (CPU) przegląda flagi i tam, gdzie jest 1, wypisuje **oryginalny napis graph6**
   (który CPU cały czas trzymał w pamięci w tablicy `lines[]`):
```c
for (int i = 0; i < count; i++)
    if (h_res[i]) { fputs(lines[i], out); fputc('\n', out); }
```

**Krótko: GPU mówi „KTÓRY" (indeksy 1/0), CPU mówi „CO" (faktyczny napis grafu).**

**Dlaczego akurat tak (uzasadnienie):**
- **GPU robi to, w czym jest dobry** — masowo równolegle liczy „tak/nie". Każdy wątek
  pisze do **innej komórki** `d_res[idx]` → zero konfliktów, zero synchronizacji,
  idealny (rozłączny) dostęp do pamięci.
- **CPU robi to, w czym jest dobry** — sekwencyjne I/O (wypis do pliku). Wypis jest
  **bardzo rzadki** (na 50 mln grafów tylko 3 trafienia!), więc nie ma sensu go
  zrównoleglać ani komplikować.
- **Minimalny transfer przez PCIe** — z GPU wraca tylko tablica małych `int`-ów (1/0),
  a nie miliony napisów. Gdyby GPU miało zwracać teksty trafień, wątki nie wiedziałyby
  z góry, ile ich będzie ani gdzie pisać — wymagałoby to atomowych liczników
  i kompaktowania wyników (dużo komplikacji dla 3 trafień).

**Analogia:** GPU to **tysiąc egzaminatorów**, każdy sprawdza jeden test i stawia tylko
**✓ albo ✗** w swojej kratce na wspólnej liście (nie przepisują całych testów — byłoby
wolno i chaotycznie). Po wszystkim **jeden sekretarz** (CPU) przegląda listę i przepisuje
do raportu tylko te nieliczne testy z ✓. Każdy robi to, w czym jest najszybszy.

### 4.6. Rozmiar bloku — strojenie i wybór 32

Liczba wątków w bloku jest **parametrem wywołania** (nie zaszyta na sztywno).
Prowadzący wprost ostrzegał: **nie ustawiać arbitralnie 256.** Zmierzyliśmy (2 mln, czas całkowity):

| Blok | 32 | 64 | 128 | 256 | 512 | 1024 |
|---|---|---|---|---|---|---|
| Czas [s] | **0,267** | 0,297 | 0,309 | 0,308 | 0,299 | 0,296 |

**Wybór: blok = 32.** Różnica jest realna i powtarzalna, ale **niewielka (~10%)** —
od najszybszego (32) do najwolniejszego (128). Poniżej dokładny mechanizm, prostym
językiem. Liczby pochodzą z kompilatora (`nvcc -Xptxas -v`), nie ze zgadywania.

**Ile faktycznie zużywa jądro (zmierzone):**
```
Used 48 registers, 1808 bytes stack frame, 0 bytes spill
```
- **48 rejestrów na wątek.** Rejestry to najszybsza pamięć GPU — „blat roboczy" wątku.
- **1808 bajtów „stack frame" na wątek.** Tu trafiają duże tablice solvera (`a[137]`,
  `d, e, e2, Lb, x`). **Ważne:** one NIE są w rejestrach, tylko w **pamięci lokalnej**
  (local memory) — to pamięć prywatna wątku, ale fizycznie leży w wolnej pamięci
  globalnej GPU (cache'owana w L1/L2).
- **0 bajtów spill** — kompilator nie musiał awaryjnie wyrzucać rejestrów do pamięci
  (to dobrze; spill bardzo spowalnia).

**Mechanizm 1 — ograniczona pula rejestrów (efekt UMIARKOWANY).**
Multiprocesor GPU (SM) to jakby „warsztat z ograniczoną liczbą blatów". RTX 3060 Ti
(Ampere) ma **65 536 rejestrów na SM** do podziału między wszystkie wątki działające
tam jednocześnie:
```
65 536 rejestrów ÷ 48 rejestrów/wątek ≈ 1 365 wątków naraz na SM
```
Sprzętowe maksimum to 1 536 wątków/SM, więc rejestry ograniczają nas do ~89% obsady
(*occupancy*). To **trochę** ogranicza, ale nie dramatycznie — więc rejestry to NIE
jest główny powód, dla którego blok=32 wygrywa.

**Mechanizm 2 — efekt „ogona" + rozbieżność wątków (GŁÓWNY powód).**
Kluczowy fakt o GPU: **blok wątków trzyma zasoby SM, dopóki NAJWOLNIEJSZY wątek w
bloku nie skończy.** Cały blok zwalnia się naraz, nie wątek po wątku.

A przez **wczesne odrzucenie** czasy wątków są skrajnie nierówne: graf niecałkowity
odpada po 1. wartości własnej (szybko), a graf „prawie całkowity" liczy pełne widmo
(~10× dłużej).

Analogia — **autobus, który czeka na ostatniego pasażera:**
> Blok = autobus. Autobus zwalnia parking (zasoby SM) dopiero, gdy WSZYSCY wysiądą.
> Blok=1024 to autobus na 1024 osoby: gdy 1023 wysiadło po sekundzie, a jeden liczy
> pełne widmo przez 10 s — autobus stoi i blokuje parking przez 10 s.
> Blok=32 to mały autobus: skończył warp → od razu zwalnia parking, SM bierze następny.
> „Maruderzy" trzymają w zakładnikach znacznie mniej innych wątków.

To jest **efekt ogona** (*tail effect*): w dużym bloku prawie zawsze trafi się wolny
wątek, który trzyma cały blok (i jego zasoby) przy życiu, choć reszta już skończyła.
Blok=32 (dokładnie jeden warp) to **najmniejsza jednostka**, jaką GPU zwalnia — więc
zasoby krążą najszybciej i „maruderzy" blokują najmniej.

**Sedno (jedno zdanie do obrony):** optymalny rozmiar bloku nie jest uniwersalną stałą
— zależy od (1) ile rejestrów/pamięci zużywa jądro i (2) jak bardzo różnią się czasy
wątków. Przy naszym jądrze (48 rej., 1808 B lokalnej + skrajna rozbieżność przez
wczesne odrzucenie) małe bloki = szybsze krążenie zasobów SM = najkrótszy czas.
Dlatego NIE należy zaszywać arbitralnie 256 — trzeba zmierzyć (co zrobiliśmy).

### 4.7. Przetwarzanie porcjami (batching) + zdarzenia do pomiaru

```c
const int CAP = CHUNK / 4 + 16;        // ile grafow w porcji
// bufory alokowane RAZ przed petla (bez realokacji)
CUDA_CHECK(cudaMalloc(&d_bits, ...));
CUDA_CHECK(cudaMalloc(&d_res, ...));

while ((r = READ(0, buf + carry, CHUNK)) > 0) {
    // ... podzial na linie ...
    // dekodowanie (OpenMP) ...
    cudaMemcpy(d_bits, h_bits, ..., cudaMemcpyHostToDevice);   // H -> D
    sieve_kernel<<<grid, block>>>(d_bits, d_res, n, nwords, cnt);
    cudaDeviceSynchronize();
    cudaMemcpy(h_res, d_res, ..., cudaMemcpyDeviceToHost);     // D -> H
    // wypis trafien
}
```

- **Bufory alokowane RAZ** przed pętlą (`cudaMalloc` poza pętlą) — prowadzący ostrzegał
  przed wielokrotną alokacją/zwalnianiem. Alokacja GPU jest droga.
- **`<<<grid, block>>>`** — konfiguracja uruchomienia: `grid = ceil(cnt / block)` bloków,
  każdy po `block` wątków.
- **Jedna synchronizacja na porcję** (`cudaDeviceSynchronize`) — w przeciwieństwie do
  `sito8`, które synchronizuje po **każdym** grafie (patrz sekcja 6).
- **`cudaEvent`** — do pomiaru czasu **samego jądra** (oddzielnie od transferu).
- **`CUDA_CHECK`** — makro opakowujące każde wywołanie CUDA, sprawdza kod błędu i
  przerywa z czytelnym komunikatem (poprawne `__FILE__`/`__LINE__`).

### 4.8. OpenMP na hoście — dekodowanie graph6 (kluczowa optymalizacja)

To jest **najciekawsza decyzja w wersji CUDA** i warto ją dobrze zrozumieć.

**Problem (prawo Amdahla):** jądro GPU jest tak szybkie, że wąskim gardłem staje się
**przygotowanie danych na hoście** — odczyt i dekodowanie 50 mln napisów graph6 do
bitmap, robione sekwencyjnie na 1 wątku CPU. Zmierzyliśmy: samo dekodowanie graph6
to ~2,4 s na 20 mln grafów. Przy sekwencyjnym hoście CUDA total = 6,3 s, z czego
~76% to host I/O — GPU czeka.

**Rozwiązanie:** zrównoleglić **dekodowanie na hoście** przez OpenMP, zostawiając
**solver w całości na GPU**:

```c
#pragma omp parallel for schedule(static) default(none) \
        shared(lines, h_bits, cnt, n, nwords)
for (int gi = 0; gi < cnt; gi++) {
    uint32_t tmp[IB_MAXWORDS];
    ib_g6_to_bits(lines[gi], n, tmp, nwords);          // dekodowanie graph6 -> bitmapa
    for (int w = 0; w < nwords; w++)
        h_bits[(size_t)w * cnt + gi] = tmp[w];         // zapis w ukladzie SoA
}
```

- **`schedule(static)`** — tu wystarczy statyczne (dekodowanie każdego grafu trwa
  tyle samo, brak zmienności jak przy solverze; więc równe bloki są optymalne i mają
  najmniejszy narzut).
- Kompilacja wymaga **`nvcc -Xcompiler /openmp`** (na Windows; `-fopenmp` na Linux/Mac)
  — przekazuje flagę OpenMP do hosta MSVC.

**Wynik:** czas hosta spadł z ~4,8 s do ~0,7 s, CUDA total z 6,3 s do 2,3 s (na 20 mln).
**Teraz CUDA bije OpenMP** (50 mln: 5,8 s vs 8,7 s = 1,51×).

**Ważne dla obrony:** to NIE zmienia algorytmu — algorytm (Householder + bisekcja)
działa w całości na GPU. OpenMP użyto tylko do **przygotowania wsadu** (warstwa CPU).
To uczciwe i zgodne z zasadą „jeden algorytm we wszystkich wersjach".

### 4.9. Dlaczego porzucono wariant `float`

Próbowaliśmy `float` (szybszy na GeForce), ale **się nie zbiega**. Progi zbieżności
bisekcji (`2.91e-16`, `7.28e-17`) są dobrane pod precyzję `double` (epsilon ~2e-16).
We `float` (epsilon ~1e-7) warunek pętli `while (w - s > prog)` praktycznie nigdy nie
staje się fałszywy → pętla iteruje znacznie dłużej, jest **wolniejsza i mniej dokładna**.

**Wniosek (ciekawy do obrony):** nie każdy algorytm zyskuje na niższej precyzji.
Algorytm bisekcji Sturma **wymaga** `double` — to przykład, że dobór typu danych
musi uwzględniać naturę algorytmu, nie tylko „mniej bitów = szybciej".

### 4.10. Wynik
- **najszybsza ze wszystkich**: 50 mln w 5,8 s (1,51× szybsza od OpenMP).
- **~1900× szybsza od `sito8`** (sito8: ~3 h na 50 mln).

---

## 5. WERSJA ZINTEGROWANA GEN — `program_omp.cpp`

### 5.1. Po co w ogóle wersja zintegrowana — problem potoku

Wszystkie poprzednie wersje (SEQ/OMP/CUDA) czytają grafy z **potoku**: `geng | sito`.
To dwa osobne procesy połączone „rurą" (pipe). Problem zmierzony:

- `geng` generuje grafy **jednowątkowo** → ~194 tys. graf/s.
- nasze sito (OMP) potrafi przerobić **>4 mln graf/s**.

Czyli sito jest ~20× szybsze niż generacja — **sito stoi i czeka na dane**. W potoku
**generacja jest wąskim gardłem**, a 15 z 16 rdzeni de facto się marnuje (sito jest
„za szybkie"). Dodatkowo potok kosztuje: geng koduje graf do napisu graph6 → przesyła
przez rurę → sito dekoduje napis z powrotem do macierzy. Dużo zbędnej pracy.

**Pomysł GEN:** skoro to generacja jest wolna, **zrównoleglmy generację**. Ale `geng`
to osobny program — jak go zmusić, żeby działał na 16 wątkach i oddawał grafy wprost
do naszego sita, bez rury i bez kodowania do tekstu? Odpowiedź: **wkompilować geng do
naszego programu jako bibliotekę.**

### 5.2. Co znaczy „geng jako biblioteka" — trzy makra

Normalnie `geng` to samodzielny program: ma własny `main()`, sam parsuje argumenty,
sam wypisuje grafy na ekran. Autor nauty (B. McKay) przewidział jednak, że ktoś może
chcieć użyć geng **wewnątrz** własnego programu — i wbudował w `geng.c` trzy „przełączniki"
preprocesora. Włączamy je przy kompilacji:

```bash
gcc -DUSE_TLS=1 -DMAXN=WORDSIZE \
    -DOUTPROC=process_graph -DGENG_MAIN=geng_main -c geng.c -o geng_lib.o
```

**Makro 1: `-DGENG_MAIN=geng_main` — „nie bądź programem, bądź funkcją".**
W `geng.c` funkcja główna nazywa się `GENG_MAIN(...)`. Bez tego makra preprocesor
zostawia ją jako `main` (samodzielny program). Z makrem zamienia ją na `geng_main` —
zwykłą funkcję, którą **my** wołamy z naszego `main()`. Czyli zamiast „uruchom program
geng" robimy „wywołaj funkcję geng_main(argc, argv)" — z tymi samymi argumentami, jakie
podałbyś w terminalu. W kodzie:
```c
extern "C" int geng_main(int argc, char *argv[]);   // deklarujemy funkcje z biblioteki
...
char *gargv[6] = {"geng", "-cq", "16", "46:46", "0/2000", NULL};
geng_main(5, gargv);                                 // to samo co: geng -cq 16 46:46 0/2000
```

**Makro 2: `-DOUTPROC=process_graph` — „nie wypisuj, oddaj mi graf".**
Domyślnie geng dla każdego wygenerowanego grafu woła wewnętrzną funkcję wypisującą
(`writeg6x` — koduje do graph6 i drukuje). Makro `OUTPROC` przekierowuje to wywołanie
do **naszej** funkcji (callback). W `geng.c`:
```c
static TLS_ATTR void (*outproc)(FILE*,graph*,int);   // wskaznik na funkcje wyjscia
...
outproc = OUTPROC;                                   // = process_graph (nasza)
...
(*outproc)(outfile, gcan, nx);   // dla KAZDEGO grafu geng wola nasz callback
```
Czyli geng, zamiast drukować graf, **wywołuje naszą funkcję `process_graph`** i podaje
jej graf **bezpośrednio jako strukturę nauty w pamięci** (typ `graph*`), a nie jako
napis. To eliminuje kodowanie/dekodowanie graph6 — w callbacku od razu mamy macierz.

Nasz callback (uproszczony):
```c
extern "C" void process_graph(FILE * /*outfile*/, graph *g, int n)
{
    // g[] to wiersze macierzy sasiedztwa jako bity (setword) - prosto z nauty
    int adj[MAX_NODES][MAX_NODES];
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++) adj[i][j] = (g[i] & bit[j]) ? 1 : 0;
    // ... ten sam solver (Householder + bisekcja) ...
    if (graf_calkowity) { /* wypisz trafienie */ }
}
```

**Makro 3: `-DUSE_TLS=1` — „daj każdemu wątkowi własną kopię stanu".**
To **najważniejszy** i najmniej oczywisty element. `geng` używa wielu **zmiennych
globalnych** (np. `outproc`, `outfile`, `connec`, liczniki...). Gdyby 16 wątków wołało
`geng_main` jednocześnie, **wszystkie pisałyby do tych samych globalnych zmiennych** →
katastrofa (wyścigi, pomieszany stan). `USE_TLS` rozwiązuje to przez **thread-local
storage**: każda globalna zmienna geng dostaje **osobną kopię na każdy wątek**.

Mechanizm w `nauty.h`:
```c
#ifdef USE_TLS
#define TLS_ATTR _Thread_local      // C11: kazdy watek ma wlasna kopie
#else
#define TLS_ATTR                    // pusto: jedna wspolna zmienna (tylko 1 watek)
#endif
```
A w `geng.c` każda globalna jest oznaczona `TLS_ATTR`:
```c
static TLS_ATTR void (*outproc)(FILE*,graph*,int);
static TLS_ATTR FILE *outfile;
static TLS_ATTR int connec;
// ... itd.
```
Słowo kluczowe `_Thread_local` (standard C11) mówi kompilatorowi: „ta zmienna nie jest
jedna — jest po jednej na każdy wątek". Dzięki temu 16 instancji `geng_main` działa
**niezależnie**, każda w swojej bańce stanu, bez deptania sobie po danych.

**Po co osobna biblioteka `nautyTL.a`?** `geng` korzysta z funkcji nauty (z plików
`nauty.c`, `nautil.c`, `naugraph.c`...). One też mają globalne zmienne. Więc **całą
nauty trzeba przekompilować z `-DUSE_TLS=1`** i spakować do osobnego archiwum
`nautyTL.a` (TL = thread-local). To robi skrypt `build_tls.sh`: kompiluje ~13 plików
nauty z flagą TLS i `ar`-em skleja w bibliotekę. Zwykła `nauty.a` (bez TLS) nadaje się
tylko do jednowątkowego `geng.exe`; do GEN potrzebna jest wersja TLS.

Składanie całości (link):
```bash
g++ -fopenmp -DUSE_TLS=1 -DMAXN=WORDSIZE \
    program_omp.cpp geng_lib.o nautyTL.a -o program_omp.exe
#            ^nasz kod    ^geng-biblioteka  ^nauty thread-local
```

### 5.3. Zrównoleglenie generacji: podział res/mod

Teraz mamy `geng_main` jako funkcję wątkowo-bezpieczną. Jak rozdzielić **generację**
między 16 wątków, żeby się nie dublowała? Tu wchodzi wbudowana w geng funkcja **res/mod**.

**Co robi res/mod:** `geng n k:k res/mod` dzieli **całą** przestrzeń grafów na `mod`
rozłącznych, równych klas i generuje **tylko** klasę numer `res`. Np.:
- `geng 16 46:46 0/4` → klasa 0 z 4 (ćwiartka grafów),
- `geng 16 46:46 1/4` → klasa 1 z 4, itd.

Klasy 0/4, 1/4, 2/4, 3/4 **razem dają całość**, bez powtórzeń i bez luk (podział
deterministyczny, wbudowany w algorytm geng). To idealne do zrównoleglenia: każdy
wątek bierze inną klasę.

Nasza pętla:
```c
int partitions = n_threads * 16;       // np. 16 watkow * 16 = 256 partycji (modulus)

#pragma omp parallel for schedule(dynamic, 1)
for (int p = 0; p < partitions; p++) {
    char res[24];
    snprintf(res, sizeof(res), "%d/%d", p, partitions);   // "p/256"
    char *gargv[6] = {"geng", "-cq", nstr, estr, res, NULL};
    geng_main(5, gargv);               // ten watek generuje klase p/256
}
```
Każda iteracja pętli woła `geng_main` z innym `res/mod` → generuje inny kawałek
przestrzeni → woła nasz callback `process_graph` na każdym grafie z tego kawałka.
16 wątków = 16 jednoczesnych generacji różnych klas. **Generacja przestała być
jednowątkowa** — to jest cały zysk GEN.

**Dlaczego `schedule(dynamic, 1)`, a nie `static`:**
Klasy `res/mod` są **nierównomierne** — geng nie gwarantuje, że każda klasa ma tyle
samo grafów (różnice potrafią być 10×+). Gdyby użyć `static` (z góry przydzielone
równe bloki iteracji), jeden wątek mógłby trafić na same „grube" klasy i liczyć długo,
gdy reszta już skończyła i czeka. `dynamic, 1` przydziela klasy **pojedynczo, w locie**:
wątek skończył swoją klasę → bierze następną wolną. To **równoważy obciążenie**.

**Dlaczego oversubskrypcja (16× więcej partycji niż wątków):**
Gdybyśmy zrobili tylko 16 partycji (= liczba wątków), `dynamic` nie miałby czym
wyrównywać — każdy wątek dostałby dokładnie jedną klasę i znów jeden „gruby" wątek
trzymałby resztę. Robiąc **256 drobnych** partycji, dajemy schedulerowi dużo małych
kawałków do żonglowania → nierównomierności się **uśredniają** (wątek, który dostał
chudą klasę, szybko bierze następną). To klasyczna technika: *więcej, mniejszych zadań
= lepsze równoważenie* (kosztem minimalnego narzutu na uruchomienie geng_main co klasę).

### 5.4. Pomiar: stały czas, liczba przeanalizowanych grafów

Cała przestrzeń n=16 k=46 to setki milionów grafów (godziny). Dlatego porównanie
zrobiono **na stały czas (20 s)** — ile grafów każda wersja zdążyła przeanalizować:

| Wersja | Grafy w 20 s | Przepustowość |
|---|---|---|
| Potok `geng \| sito_omp` | 7,4 mln | 370 tys./s |
| **GEN** | **36 mln** | **1,8 mln/s** |

**GEN ~4,9× szybszy.** Bo potok jest ograniczony jednowątkową generacją, a GEN
zrównolegla generację na 16 wątków.

### 5.5. Znana wada GEN (warto o niej wiedzieć)

Mechanizm wczesnego zakończenia (`EXIT_AFTER_FIRST`) woła `exit()` z **wnętrza regionu
OpenMP**, gdy inne wątki są w `geng_main`. To prowadzi do **zakleszczenia** przy
sprzątaniu wątków (proces staje się nieusuwalnym „zombie"). Wniosek: GEN należy
uruchamiać do końca lub przerywać sygnałem z zewnątrz, a nie kończyć z callbacku.
To uczciwa rzecz do wspomnienia — pokazuje, że rozumiesz pułapki `exit()` w kodzie
równoległym.

---

## 6. KODY REFERENCYJNE — dlaczego są wolne

### 6.1. `sito5` (CPU)
Ten sam algorytm co nasz (Householder + bisekcja + wczesne odrzucenie), ale w
**`long double`** (80-bit x87). To jedyna różnica — i daje 2,48× spowolnienia względem
naszego `double`. Nasza wersja OMP to ten sam kod zrównoleglony + szybszy typ.

### 6.2. `sito8` (GPU) — antywzorzec

```c
while (fgets(BUFFOR, BUFSIZE-1, stdin)) {
    cudaMemcpy(cuda_bufor, BUFFOR, BUFSIZE, cudaMemcpyHostToDevice);
    test<<<1,1>>>(cuda_bufor);     // JEDEN blok, JEDEN watek
    cudaDeviceSynchronize();        // synchronizacja po KAZDYM grafie
}
```

`sito8` to **celowo zły** punkt odniesienia (do pobicia):
- **`<<<1,1>>>`** — jeden blok, jeden wątek = używa **~0,001% rdzeni GPU**. Cała moc
  GPU leży odłogiem; działa jak (wolny) pojedynczy rdzeń.
- **`cudaMemcpy` + `cudaDeviceSynchronize` po KAŻDYM grafie** — pełny narzut transferu
  i synchronizacji 50 mln razy.

**Zmierzona przepustowość: ~4 536 graf/s** (~215 µs/graf). Zweryfikowane niezależnie:
trzy różne sprawozdania podają 2,7k–8,8k graf/s na różnych GPU — ten sam rząd wielkości.

Rozbicie 215 µs/graf (zmierzone): sam narzut wywołania pustego jądra `<<<1,1>>>` + sync
to ~9 µs, reszta (~206 µs) to **obliczenie na jednym wolnym wątku GPU**. Dla porównania:
ten sam solver na 1 rdzeniu CPU (`sito_seq`) to ~1,7 µs/graf. Pojedynczy rdzeń GPU jest
~125× wolniejszy od rdzenia CPU dla szeregowego kodu skalarnego (niższy zegar, brak
zaawansowanego ILP, wolny `sqrt`/dzielenie w `double`). **Nasza CUDA bije sito8 ~1900×**,
bo używa tysięcy wątków zamiast jednego.

**Ważna lekcja:** GPU nie jest „szybkie" — jest **szerokie**. Sukces na GPU = masowa
równoległość, nie szybki pojedynczy wątek.

---

## 7. PODSUMOWANIE WYNIKÓW (50 mln grafów, n=16 k=46)

| Wersja | Czas | vs sito5 | Co decyduje |
|---|---|---|---|
| `sito5` (ref.) | 214,8 s | 1,0× | long double, 1 wątek |
| **SEQ** | 86,7 s | 2,48× | double zamiast long double |
| **OMP** (16 w.) | 8,7 s | 24,6× | + zrównoleglenie 16 wątków |
| **CUDA** | **5,8 s** | **37,2×** | + tysiące wątków GPU, host-decode OpenMP |
| `sito8` (ref.) | ~3 h | 0,02× | <<<1,1>>>, sync per graf |

GEN (generacja+sito) ~4,9× szybszy end-to-end od potoku.

---

## 8. PRZEWIDYWANE PYTANIA + ODPOWIEDZI (do obrony)

**P: Dlaczego CUDA jest tylko 1,5× szybsza od OpenMP, skoro GPU ma tysiące rdzeni?**
O: Bo nasz algorytm ma **wczesne odrzucenie** — większość grafów odpada błyskawicznie,
więc samo liczenie jest bardzo lekkie. Wąskim gardłem staje się przygotowanie danych
na hoście (odczyt+dekodowanie graph6) i transfer przez PCIe, które rosną liniowo
i nie da się ich w pełni ukryć. Samo jądro GPU jest ~2,3× szybsze od 16 wątków CPU,
ale całość ogranicza I/O hosta (prawo Amdahla). Gdyby algorytm był cięższy na graf,
przewaga GPU byłaby większa.

**P: Dlaczego użyłeś OpenMP wewnątrz wersji CUDA? To nie miesza technologii?**
O: OpenMP użyto **tylko** do zrównoleglenia przygotowania danych na CPU (dekodowanie
graph6). Właściwy algorytm sita (Householder + bisekcja) działa w **100% na GPU**.
To standardowa praktyka — host przygotowuje wsad, GPU liczy. Bez tego sekwencyjny
host dławiłby szybkie GPU.

**P: Dlaczego blok = 32, a nie 256/1024?**
O: Bo jądro jest rejestrochłonne (każdy wątek trzyma ~6 tablic double na stosie).
Duże bloki przekraczają pulę rejestrów SM → niska zajętość. Zmierzyłem wszystkie
rozmiary 32–1024 i 32 (rozmiar warpu) wypadł najlepiej. Prowadzący wprost ostrzegał
przed arbitralnym 256.

**P: Dlaczego `schedule(runtime)`/`guided` w OMP, a `static` przy dekodowaniu w CUDA?**
O: W sicie czas grafu jest **zmienny** (wczesne odrzucenie) → potrzebne dynamiczne
równoważenie (guided/dynamic). W dekodowaniu graph6 każdy graf trwa **tyle samo** →
static jest optymalny (najmniejszy narzut, równe bloki).

**P: Skąd pewność, że wyniki są poprawne?**
O: Trzy poziomy weryfikacji: (1) zgodność z OEIS A064731 dla n=5..9, (2) wyjście OMP
i CUDA bajt-w-bajt identyczne z sekwencyjnym, (3) sito akceptuje wszystkie 375 znanych
grafów całkowitych ze zbioru grafy16L.

**P: Dlaczego `default(none)`?**
O: Wymóg prowadzącego i dobra praktyka — zmusza do jawnego określenia, co jest shared,
a co private. Eliminuje najczęstsze źródło błędów w OpenMP (przypadkowe współdzielenie).

**P: Czym różni się Twój OMP od sito5_omp?**
O: Ten sam algorytm matematyczny, ale: (1) double zamiast long double, (2) blokowy
read() zamiast fgets, (3) podział linii in-place bez kopiowania, (4) wypis przez
tablicę hit[] zamiast printf w pętli (brak serializacji na stdout). Punkt (1) to gros
różnicy.

**P: Dlaczego nie float na GPU?**
O: Progi zbieżności bisekcji są dobrane pod double; we float pętla się nie zbiega
(jest wolniejsza i mniej dokładna). Algorytm bisekcji Sturma wymaga podwójnej precyzji.

**P: Co to jest dostęp scalony (coalesced) i czemu układ SoA?**
O: Gdy 32 wątki warpu czytają 32 sąsiednie adresy, GPU łączy to w jedną transakcję
pamięci zamiast 32. Układ SoA (d_bits[w*num+idx]) sprawia, że sąsiednie wątki (idx)
czytają sąsiednie adresy → dostęp scalony → szybszy odczyt z pamięci globalnej.

---

## 9. Mapa plików projektu

```
src/
  integral.h          - rdzen sita, wejscie: napis graph6 (SEQ, OMP)
  integral_bits.h     - rdzen sita, wejscie: bitmapa (CUDA + test CPU)
  sito_seq.c          - wersja sekwencyjna
  sito_omp.c          - wersja OpenMP (read blokowy, hit[], schedule runtime)
  sito_cuda.cu        - wersja CUDA (SoA, blok param, host-decode OpenMP)
  program_omp.cpp     - GEN: geng jako biblioteka + OpenMP res/mod
  test_cuda_logic.c   - weryfikacja logiki CUDA na CPU (bez GPU)
  sito5.c, sito8.cu   - kody referencyjne prowadzacego
scripts/              - build_all.sh, bench_*.sh, g6_to_dokuwiki.py
data/                 - probki .g6, wyniki, tabele bench
bin/                  - zbudowane programy
```
