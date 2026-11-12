/* ======================================================================
      3DBPP PROBLEMS, Silvano Martello, David Pisinger, Daniele Vigo
   ====================================================================== */

/* This code generates instances for the three-dimensional bin-packing 
 * problem and solves them using the 3DBPP algorithm by Martello, Pisinger,
 * Vigo. 
 *
 * A description of the test instances is found in the following papers:
 *
 *   S.Martello, D.Pisinger, D.Vigo (1998)
 *   "An exact algorithm for the three-dimensional bin packing problem"
 *   submitted.
 *
 *   S.Martello, D.Pisinger, D.Vigo (1998)
 *   "The three-dimensional bin packing problem"
 *   to appear in Operations Research.
 *
 * The algorithm prompts for three arguments:
 *   n      The size of the instance, i.e. number of boxes.
 *   bindim The size of the bin, typically 40-100.
 *   type   A value between 1-9 selecting one of the instance types
 *          described in the above papers.
 *
 * Results are written to the file "3dbpp.out".
 * 
 * (c) Copyright 1998,
 *
 *   David Pisinger                        Silvano Martello, Daniele Vigo
 *   DIKU, University of Copenhagen        DEIS, University of Bologna
 *   Universitetsparken 1                  Viale Risorgimento 2
 *   Copenhagen, Denmark                   Bologna, Italy
 * 
 * This code can be used free of charge for research and academic purposes 
 * only. 
 */

#define TESTS          10     /* Number of test to run for each type */
#define MAXITEMS      201     /* Max number of items plus one */

#include <stdlib.h>
#include <stdio.h>
#include <stdarg.h>
#include <values.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <limits.h>


/* ======================================================================
				   macros
   ====================================================================== */

#define srand(x)     srand48x(x)
#define randm(x)     (lrand48x() % (long) (x))
#define rint(a,b)    (randm((b)-(a)+1) + (a))

#define TRUE  1           /* logical variables */
#define FALSE 0

#define VOL(i)                 ((i)->w * (ptype) (i)->h * (i)->d)
#define DIF(i,j)               ((int) ((j) - (i) + 1))


/* ======================================================================
				 type declarations
   ====================================================================== */

typedef short         boolean; /* logical variable      */
typedef short         ntype;   /* number of states,bins */
typedef short         itype;   /* can hold up to W,H,D  */
typedef long          stype;   /* can hold up to W*H*D  */
typedef long          ptype;   /* product multiplication */

typedef int (*funcptr) (const void *, const void *);

/* item record */
typedef struct irec {
  ntype    no;           /* item number */
  itype    w;            /* item x-size */
  itype    h;            /* item y-size */
  itype    d;            /* item z-size */
  itype    x;            /* optimal x-position */
  itype    y;            /* optimal y-position */
  itype    z;            /* optimal z-position */
  ntype    bno;          /* bin number */
  stype    vol;          /* volume of item */
} item;


/* ======================================================================
				global variables
   ====================================================================== */

FILE *trace;


/* =======================================================================
                                random
   ======================================================================= */

/* to generate the same instances as at HP9000 - UNIX, */
/* here follows C-versions of SRAND48, and LRAND48.  */

unsigned long _h48, _l48;

void srand48x(long s)
{
  _h48 = s;
  _l48 = 0x330E;
}

long lrand48x(void)
{
  _h48 = (_h48 * 0xDEECE66D) + (_l48 * 0x5DEEC);
  _l48 = _l48 * 0xE66D + 0xB;
  _h48 = _h48 + (_l48 >> 16);
  _l48 = _l48 & 0xFFFF;
  return (_h48 >> 1);
}


/* ======================================================================
				   timeused
   ====================================================================== */

void timeused(double *time)
{
  static double tstart, tend, tprev;

  if (time == NULL) {
    clock(); /* one extra call to initialize clock */
    tstart = tprev = clock();
  } else {
    tend = clock();
    if (tend < tprev) tstart -= ULONG_MAX; /* wraparound occured */
    tprev = tend;
    *time = (tend-tstart) / CLOCKS_PER_SEC; /* convert to seconds */
  }
}


/* ======================================================================
				specialbin
   ====================================================================== */

void specialbin(item *f, item *l, int W, int H, int D)
{
  item *m, *i, *j, *k;
  int w, h, d, r;

  /* vis(1,"specialbin [%d] %d,%d,%d\n", DIF(f,l), W, H, D); */
  if (f == l) { f->w = W; f->h = H; f->d = D; return; }
  if (DIF(f,l) == 5) {
    w = W/3; h = H/3; i = f+1; j = f+2; k = f+3; 
    i->w =   W-w; i->h =     h; i->d = D;
    j->w =     w; j->h =   H-h; j->d = D;
    k->w =   W-w; k->h =     h; k->d = D; 
    l->w =     w; l->h =   H-h; l->d = D;
    f->w = W-2*w; f->h = H-2*h; f->d = D;
    return;
  }

  m = f + (l-f) / 2;
  for (;;) {
    switch (rint(1,3)) {
      case 1: if (W < 2) break;
              w = rint(1,W-1); 
  	      specialbin(f, m, w, H, D);
	      specialbin(m+1, l, W-w, H, D);
	      return;
      case 2: if (H < 2) break;
	      h = rint(1,H-1); 
	      specialbin(f, m, W, h, D); 
	      specialbin(m+1, l, W, H-h, D);
	      return;
      case 3: if (D < 2) break;
	      d = rint(1,D-1);
	      specialbin(f, m, W, H, d); 
	      specialbin(m+1, l, W, H, D-d);
	      return;
    }
  }
}


/* ======================================================================
				randomtype
   ====================================================================== */

void randomtype(item *i, int W, int H, int D, int type)
{
  itype w, h, d, t;

  if (type <= 5) { /* Martello, Vigo */
    t = rint(1,10); if (t <= 5) type = t;
  }
  switch (type) {
    /* Martello, Vigo */
    case  1: w = rint(1,W/2);   h = rint(2*H/3,H); d = rint(2*D/3,D); break;
    case  2: w = rint(2*W/3,W); h = rint(1,H/2);   d = rint(2*D/3,D); break;
    case  3: w = rint(2*W/3,W); h = rint(2*H/3,H); d = rint(1,D/2);   break;
    case  4: w = rint(W/2,H);   h = rint(H/2,H);   d = rint(D/2,D);   break;
    case  5: w = rint(1,W/2);   h = rint(1,H/2);   d = rint(1,D/2);   break;

    /* Berkey, Wang */
    case 6: w = rint(1,10);    h = rint(1,10);    d = rint(1,10);    break;
    case 7: w = rint(1,35);    h = rint(1,35);    d = rint(1,35);    break;
    case 8: w = rint(1,100);   h = rint(1,100);   d = rint(1,100);   break;
  }
  i->w = w; i->h = h; i->d = d;
  i->x = 0; i->y = 0; i->z = 0;
}


/* ======================================================================
				allgood
   ====================================================================== */

boolean allgood(stype totvol, item *f, item *l)
{
  item *j, *m;
  stype vol;

  for (vol = 0, j = f, m = l+1; j != m; j++) {
    if ((j->w < 1) || (j->h < 1) || (j->d < 1)) return FALSE;
    vol += VOL(j);
  }
  return (vol == totvol);
}


/* ======================================================================
				maketest
   ====================================================================== */

void maketest(item *f, item *l, itype *W, itype *H, itype *D,
	      stype bdim, int type)
{
  register item *i, *j, *k, *m;
  int no;

  /* set bin dimensions */
  *W = bdim; *H = bdim; *D = bdim;

  /* make maxtypes item types */
  for (i = f, m = l+1, no = 1; i != m; i++, no++) {
    randomtype(i, *W, *H, *D, type);
    i->no = no;
  }

  /* make two complete bins when test */
  if (type == 9) {
    no = DIF(f,l)/3;
    k = f + no; m = k + no;
    for (;;) {
      specialbin(f  , k, *W, *H, *D); 
      specialbin(k+1, m, *W, *H, *D); 
      specialbin(m+1, l, *W, *H, *D); 
      if (allgood(3*bdim*bdim*bdim, f, l)) break;
    }
  }
}


/* ======================================================================
                                printitems
   ====================================================================== */

void printitems(item *f, item *l, itype W, itype H, itype D)
{
  item *i;
  stype vol;

  vol = 0;
  fprintf(trace,"     [  w,  h,  d]    vol\n");
  for (i = f; i != l+1; i++) {
    i->vol = VOL(i); vol += i->vol; 
    fprintf(trace,"%3d: [%3hd,%3hd,%3hd] %6ld\n",
            i->no,i->w,i->h,i->d,i->vol);
  }
  fprintf(trace,"BIN DIMENSIONS: %d %d %d\n", W, H, D);
  fprintf(trace,"VOLUME OF ITEMS: %ld\n", vol);
  fprintf(trace,"VOLUME OF BIN: %ld\n", W*(long)H*D);
}


/* ======================================================================
                                prepareitems
   ====================================================================== */

void prepareitems(item *f, item *l, int *w, int *h, int *d)
{
  item *i;
  int k;

  for (i = f, k = 0; i != l+1; i++, k++) {
    w[k] = i->w; h[k] = i->h; d[k] = i->d;
  }
}


/* ======================================================================
				main
   ====================================================================== */

int main(int argc, char *argv[])
{
  int v, n;
  itype W, H, D;
  int bdim, type;
  item *f, *l, tab[MAXITEMS];
  int w[MAXITEMS], h[MAXITEMS], d[MAXITEMS];
  int x[MAXITEMS], y[MAXITEMS], z[MAXITEMS], bno[MAXITEMS];
  int ub, lb;
  int solved, gap;
  double time, sumtime, deviation, sumdev;

  if (argc == 4) {
    n    = atoi(argv[1]);
    bdim = atoi(argv[2]);
    type = atoi(argv[3]);
    printf("3DBPP PROBLEM n %2d W %3d type %1d\n", n, bdim, type);
  } else {
    printf("3DBPP PROBLEM\n");
    printf("n = ");
    scanf("%d", &n);
    printf("bindim = ");
    scanf("%d", &bdim);
    printf("type = ");
    scanf("%d", &type);
  }

  trace = fopen("3dbpp.out", "a");
  fprintf(trace,"\n3DBPP PROBLEM %2d %3d %1d\n", n, bdim, type);
 
  sumtime = 0; sumdev = 0; gap = 0;
  for (v = 1; v <= TESTS; v++) {
    srand(v+n); /* initialize random generator */
    f = &tab[0]; l = &tab[n-1];
    maketest(f, l, &W, &H, &D, bdim, type);
    /* printitems(f, l, W, H, D); */
    prepareitems(f, l, w, h, d);
    timeused(NULL);
    binpack3d(n, W, H, D, w, h, d, x, y, z, bno, &lb, &ub, 1000);
    timeused(&time); 
    fprintf(trace,"%2d : lb %2d z %2d time %6.2lf\n", v, lb, ub, time); 
    sumtime += time;
    gap += ub - lb;
    deviation = (ub - lb) / (double) lb;
    sumdev += deviation;
    if (lb == ub) solved++;
  }
  fprintf(trace, "n      = %d\n", n);
  fprintf(trace, "bdim   = %d\n", bdim);
  fprintf(trace, "type   = %d\n", type);
  fprintf(trace, "solved = %d\n", solved);
  fprintf(trace, "gap    = %.1lf\n", gap / (double) TESTS);
  fprintf(trace, "dev    = %.2lf\n", sumdev / TESTS);
  fprintf(trace, "time   = %.2lf\n", sumtime / TESTS);
  fclose(trace);

  return 0; /* correct termination */
}
