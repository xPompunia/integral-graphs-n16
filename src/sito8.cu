/**
@file sito8.cu
Kod referencyjny prowadzacego (wersja CUDA do porownania).

UWAGA metodyczna: ta wersja uruchamia jadro w konfiguracji <<<1,1>>> (jeden blok,
jeden watek) OSOBNO dla kazdego grafu, z cudaDeviceSynchronize() po kazdym
uruchomieniu. Wykorzystuje wiec znikomy ulamek mocy GPU i placi pelny narzut
uruchomienia+synchronizacji jadra dla kazdej instancji. Jest to swiadomie slaby
punkt odniesienia ("baseline"), ktory wlasna wersja CUDA (sito_cuda.cu) ma pobic.

Wejscie obcinane jest do dlugosci glen (argv[1]); dla n=16 graph6 ma 21 znakow,
wiec uruchamiac: ... | ./sito8 21

Kompilacja:
  nvcc -O3 -arch=sm_86 sito8.cu -o sito8
*/

#include <stdio.h>
#include <stdlib.h>

#define BUFSIZE 1024
#define NMAX 20

__global__ void test(char * BUFFOR) {

  int i,j,k,k3,k4,L,L1,z;
  double /* lambda, */ eps,g,h,ma,mn,norm,s,t,u,w;
  int cond;
  double d[NMAX+1], e[NMAX+1], e2[NMAX+1], Lb[NMAX+1];
  double x[NMAX+1];
  double a[NMAX*(NMAX-1)/2 + NMAX + 1];
  int n;
  int bit, poz, poz2;
  bit = 32;
  poz = 1;
  poz2 = 1;
  n = BUFFOR[0] - 63;
  a[0] = 0.0;
  for (i = 0; i < n; i++)
   for (j = 0; j<=i; j++)
    {
	  if (i==j) {a[poz2++] = 0; }
	  else {
        if (bit == 0) { bit = 32;  poz++; }
        if ((BUFFOR[poz] - 63) & bit)
                { a[poz2++] = 1; }
               else
                { a[poz2++] = 0; }
        bit = bit >> 1;
	  }
    }

  int k1 = 1;
  int k2 = n;
   {
    i = 0;
    for (L=1;L<=n;L++) { i += L; d[L] = a[i]; }

    for (L=n;L>=2;L--)
     {
      i--; j = i; h = a[j]; s = 0;
      for (k=L-2;k>=1;k--) { i--; g = a[i]; s += g*g; }
      i--;
      if (s == 0) { e[L] = h; e2[L] = h*h; a[j] = 0.0; }
       else
        {
          s += h*h; e2[L] = s; g = sqrtf(s); if (h>=0.0) g=-g;
          e[L] = g;
          s = 1.0 / (s-h*g);
          a[j] = h - g; h = 0.0; L1 = L - 1; k3 = 1;
          for (j=1;j<=L1;j++)
           {
             k4 = k3; g = 0;
             for (k=1;k<=L1;k++) { g +=a[k4]*a[i+k];
                                   if (k<j)  z = 1; else z = k;
                                   k4 += z; }
             k3 += j; g *= s; e[j] = g; h += a[i+j]*g;
           }
          h *= 0.5*s; k3 = 1;
          for (j=1;j<=L1;j++)
           {
             s = a[i+j]; g = e[j]-h*s; e[j] = g;
             for (k=1;k<=j;k++) { a[k3] += -s*e[k]-a[i+k]*g; k3++; }
           }
        }
      h = d[L]; d[L] = a[i+L]; a[i+L] = h;
     }
    h = d[1]; d[1] = a[1]; a[1] = h; e[1] = 0.0; e2[1] = 0.0; s = d[n];
    t = fabsf(e[n]); mn = s - t; ma = s + t;
    for (i=n-1;i>=1;i--)
     {
      u = fabsf(e[i]); h = t + u; t = u; s = d[i]; u = s - h;
      if (u < mn) mn = u;
      u = s + h;
      if (u > ma) ma = u;
     }
    for (i=1;i<=n;i++) { Lb[i] = mn; x[i] = ma; }
    norm = fabsf(mn); s = fabsf(ma);
    if (s>norm) norm = s;
    w = ma; eps = 7.28e-17*norm;
    for (k=k2;k>=k1;k--)
     {
      s = mn; i = k;
      do {cond = 0; g = Lb[i];
         if (s < g) s = g; else { i--; if (i>=k1) cond = 1; }
      } while (cond);
      g = x[k];
      if (w>g) w = g;
      while (w-s>2.91e-16*(fabsf(s)+fabsf(w))+eps)
       {
         if (floorf(w+10e-5)<s-10e-5) return;
         L1 = 0; g = 1.0; t = 0.5*(s+w);
         for (i=1;i<=n;i++)
          {
            if (g!=0)  g = e2[i] / g; else g = fabsf(6.87e15*e[i]);
            g = d[i]-t-g;
            if (g<0) L1++;
          }
         if (L1<k1) { s = t; Lb[k1] = s; }
          else
           { if (L1<k)
               {
                 s = t; Lb[L1+1] = s;
                 if (x[L1]>t) x[L1] = t;
               }
              else w = t;
           }
      } // while
      u = 0.5*(s+w); x[k] = u;
	  if  (!(( ceilf(u) - u  < 10e-5 ) || ( u - floorf(u) < 10e-5 ))) { return; };
    }
  }
  printf("%s\n",BUFFOR);
}

int main(int argc, char *argv[])
{
  char BUFFOR[BUFSIZE];
  char * cuda_bufor;
  int glen = 2;

  if (argc>1) {glen=strtol(argv[1],NULL,10);}

  cudaMalloc((void**)&cuda_bufor,BUFSIZE);
  while (fgets(BUFFOR,BUFSIZE-1,stdin)) {
	BUFFOR[glen]='\0';
	cudaMemcpy(cuda_bufor,BUFFOR,BUFSIZE,cudaMemcpyHostToDevice);
    test<<<1,1>>>(cuda_bufor);
    cudaDeviceSynchronize();
  } // while
  cudaFree(cuda_bufor);
  return EXIT_SUCCESS;
}
