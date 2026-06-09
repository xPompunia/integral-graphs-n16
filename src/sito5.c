/**
@file sito5.c

Kod referencyjny prowadzacego (wersja sekwencyjna oraz OpenMP).

Kompilacja (sekwencyjna):
  gcc -O3 sito5.c -o sito5 -lm

Kompilacja (OpenMP):
  gcc -O3 sito5.c -o sito5_omp -fopenmp -lm

Uruchomienie (zliczanie bez zapamietywania znalezionych grafow):
  ./geng -c 9 2>/dev/null | ./sito5 | wc -l

Wyniki mozna porownac z https://oeis.org/A064731
*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <omp.h>
#define NMAX 20
#define BUFSIZE 1024

/**  Funkcja eigensymmatrix jest wzorowana na kodzie w jezyku Pascal A.Marciniaka */
/**
@brief Testowanie czy graf jest calkowity
@param BUFOR - lancuch tekstowy kodujacy graf w formacie graph6
@return 1 lub 0
*/
int eigensymmatrix(char * BUFOR)
{
  int i,j,k,k3,k4,L,L1,z;
  long double /* lambda, */ eps,g,h,ma,mn,norm,s,t,u,w;
  int cond;
  long double d[NMAX+1], e[NMAX+1], e2[NMAX+1], Lb[NMAX+1];
  long double x[NMAX+1];
  long double a[NMAX*(NMAX-1)/2 + NMAX + 1];
  int n;
  int bit, poz, poz2;
  bit = 32;
  poz = 1;
  poz2 = 1;
  n = BUFOR[0] - 63;
  a[0] = 0.0;
  for (i = 0; i < n; i++)
   for (j = 0; j<=i; j++)
    {
	if (i==j) {a[poz2++] = 0; }
	else {
         if (bit == 0) { bit = 32;  poz++; }
         if ((BUFOR[poz] - 63) & bit)
                { a[poz2++] = 1; }
               else
                { a[poz2++] = 0; }
         bit = bit >> 1;
	}
    }

  int k1 = 1;
  int k2 = n;
  // if ((1<=k1) && (k1<=k2) && (k2<=n))
   {
    i = 0;
    for (L=1;L<=n;L++) { i += L; d[L] = a[i]; /* printf("%Lf ",a[i]); */ }

    for (L=n;L>=2;L--)
     {
      i--; j = i; h = a[j]; s = 0;
      for (k=L-2;k>=1;k--) { i--; g = a[i]; s += g*g; }
      i--;
      if (s == 0) { e[L] = h; e2[L] = h*h; a[j] = 0.0; }
       else
        {
          s += h*h; e2[L] = s; g = sqrt(s); if (h>=0.0) g=-g;
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
    t = fabs(e[n]); mn = s - t; ma = s + t;
    for (i=n-1;i>=1;i--)
     {
      u = fabs(e[i]); h = t + u; t = u; s = d[i]; u = s - h;
      if (u < mn) mn = u;
      u = s + h;
      if (u > ma) ma = u;
     }
    for (i=1;i<=n;i++) { Lb[i] = mn; x[i] = ma; }
    norm = fabs(mn); s = fabs(ma);
    if (s>norm) norm = s;
    w = ma; /* lambda = norm; */ eps = 7.28e-17*norm;
    for (k=k2;k>=k1;k--)
     {
      /* eps = 7.28e-17*norm; */ s = mn; i = k;
      do {cond = 0; g = Lb[i];
         if (s < g) s = g; else { i--; if (i>=k1) cond = 1; }
      } while (cond);
      g = x[k];
      if (w>g) w = g;
      while (w-s>2.91e-16*(fabs(s)+fabs(w))+eps)
       {
         if (floor(w+10e-5)<s-10e-5) return 0;  // przedzial nie zawiera liczby calkowitej
         L1 = 0; g = 1.0; t = 0.5*(s+w);
         for (i=1;i<=n;i++)
          {
            if (g!=0)  g = e2[i] / g; else g = fabs(6.87e15*e[i]);
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
	  if  (!(( ceil(u) - u  < 10e-5 ) || ( u - floor(u) < 10e-5 ))) { return 0; };
    }
  }
  return 1;
}

int main(int argc, char *argv[])
{

#ifdef _OPENMP
  int t = 1;
  char ** bufory = NULL;

  if (argc > 1)  t = atoi(argv[1]);
   bufory = (char **)malloc(t*sizeof(char *));
   for (int i=0;i<t;i++) bufory[i]=(char*)malloc(BUFSIZE);

  int graphs = 0;
  int gid;

 do {
    for (graphs=0;graphs<t;graphs++) {if (fgets(bufory[graphs],BUFSIZE,stdin)==NULL) break;}
    #pragma omp parallel default(none) shared(bufory,graphs) private(gid)
    {
       #pragma omp for
	for (gid = 0;gid < graphs;gid++)
	   if (eigensymmatrix(bufory[gid])) {
   	     printf("%s",bufory[gid]);
        }
    }
 } while(graphs==t);

 for (int i=0;i<t;i++) free(bufory[i]);
 free(bufory);
#else
  char BUFOR[BUFSIZE];
  while (fgets(BUFOR,BUFSIZE,stdin)) {
   if (eigensymmatrix(BUFOR)) printf("%s",BUFOR);
  } // while
#endif
  return EXIT_SUCCESS;
}
