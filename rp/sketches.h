#ifndef SKETCHES_H_INCLUDED
#define SKETCHES_H_INCLUDED


/*
   Sketches for Linear Regression

   Copyright: Jens Quedenfeld (original C++ implementation)
              Ludger Sandig (this C port)
   License: GPL v3
*/

#include <stdio.h>
#include <stdint.h>
#include <math.h>
#include <stddef.h>
#include <cstdlib>
#include <stdexcept>
#include <random>
#include <map>
#include <algorithm>
#include <vector>
#include <iostream>
#include <sstream>
#include <fstream>
#include <iterator>
#include <unordered_map>

#include <boost/array.hpp>
#include <boost/range/algorithm_ext/erase.hpp>
#include <boost/algorithm/string.hpp>
#include <boost/algorithm/string/predicate.hpp>
#include <boost/iostreams/filtering_streambuf.hpp>
#include <boost/iostreams/copy.hpp>
#include <boost/iostreams/filter/gzip.hpp>
#include <boost/iostreams/filter/bzip2.hpp>
#include "matrices.h"
#include "random_engine.h"
#include "stop_watch.h"


#define _XOPEN_SOURCE 700

/*
 * maximum sequence lenght for the BCH generators. As R only has 32 bit signed
 * integers, this is a sensible number:
 */
#define BCHDOMAIN ((1U << 31) - 1)

typedef struct
{
    unsigned int deg; /* degree of independence */
    unsigned int k;   /* number of seeds */
    uint_fast64_t s0; /* extra bit */
    uint_fast64_t *S; /* array of k full seeds */
} BCH_conf;

static BCH_conf bch_configure(unsigned int deg);
static double bch_gen(uint_fast64_t idx, BCH_conf c);
static double bch4_gen(uint_fast64_t idx, BCH_conf c);
static int bin_search(int val, const int *p, int k);
static uint_fast64_t lcg_init();
static void matprod_block(double *x, int nrx, int ncx,
                          double *y, int nry, int ncy, int ory, double *z);
static void matprod_block_xrm(double *x, int nrx, int ncx,
                              double *y, int nry, int ncy, int ory, double *z);
static uint_fast64_t ruint();
static void sample_int(int n, int max, int *out);
// SEXP sketch_cw(SEXP data, SEXP sketch_rows);
// SEXP sketch_rad(SEXP data, SEXP sketch_rows);
// SEXP sketch_srht(SEXP data, SEXP sketch_rows);
static void srht_rec(const int *p, int k, int q,
                     double *data, int d_rows, int d_cols, int d_offset,
                     double *res, int r_rows, int r_offset);

/* register .Call entrypoints with R */
// static const R_CallMethodDef CallEntries[] = {
//   {"sketch_cw",   (DL_FUNC) &sketch_cw,   2},
//   {"sketch_rad",  (DL_FUNC) &sketch_rad,  2},
//   {"sketch_srht", (DL_FUNC) &sketch_srht, 2},
//   {NULL, NULL, 0}
// };

// void
// R_init_RaProR(DllInfo *dll)
// {
//   R_registerRoutines(dll, NULL, CallEntries, NULL, NULL);
//   R_useDynamicSymbols(dll, FALSE);
// }

void *
R_alloc(size_t nElements, size_t typeSize)
{
    return std::malloc(nElements * typeSize);
}

double
runif(double lower, double upper)
{
    std::uniform_real_distribution<double> gen(lower, upper);
    return gen(RandomEngine::get());
}

double
unif_rand()
{
    return runif(0.0, 1.0);
}

BCH_conf
bch_configure(unsigned int deg)
{
    BCH_conf c;
    unsigned int i;

    c.deg = deg;
    c.k = deg / 2;
    c.s0 = deg % 2 ? ruint() % 2 : 0;
    c.S = (uint_fast64_t *)R_alloc(c.k, sizeof(uint_fast64_t));
    for (i = 0; i < c.k; i++)
        c.S[i] = ruint();
    return (c);
}

double
bch_gen(uint_fast64_t idx, BCH_conf c)
{
    uint_fast64_t idx_pow = idx;
    uint_fast64_t dp_acc = c.s0;
    uint_fast64_t dp = 0;
    unsigned int i;

    if (idx > BCHDOMAIN)
        throw std::invalid_argument("bch_gen: invalid index");
    for (i = 0; i < c.k; ++i)
    {
        dp = c.S[i] & idx_pow;
        /* calculate parity via Hamming weight (cf. Wikipedia) */
        /* while gcc has __builtin_parityll(), we don't use it for portability */
        dp -= (dp >> 1) & 0x5555555555555555;
        dp = (dp & 0x3333333333333333) + ((dp >> 2) & 0x3333333333333333);
        dp = (dp + (dp >> 4)) & 0x0f0f0f0f0f0f0f0f;
        dp = (dp * 0x0101010101010101) >> 56;
        /* parity: last bit of Hamming weight */
        dp = dp & 1;
        dp_acc = dp_acc ^ dp;
        /* next index, multiply by idx * idx, keep only bits in BCHDOMAIN */
        idx_pow = (((idx_pow * idx) & BCHDOMAIN) * idx_pow) & BCHDOMAIN;
    }
    return ((double)dp_acc) * 2 - 1;
}

/* unrolled version for 4-wise independence */
double
bch4_gen(uint_fast64_t idx, BCH_conf c)
{
    uint_fast64_t dp_acc = 0;
    uint_fast64_t dp = 0;

    if (idx > BCHDOMAIN)
        throw std::invalid_argument("bch4_gen: invalid index");

    /* 1st seed */
    dp = c.S[0] & idx;
    dp -= (dp >> 1) & 0x5555555555555555;
    dp = (dp & 0x3333333333333333) + ((dp >> 2) & 0x3333333333333333);
    dp = (dp + (dp >> 4)) & 0x0f0f0f0f0f0f0f0f;
    dp = (dp * 0x0101010101010101) >> 56;
    dp = dp & 1;
    dp_acc = dp_acc ^ dp;
    idx = (((idx * idx) & BCHDOMAIN) * idx) & BCHDOMAIN;
    /* 2nd seed */
    dp = c.S[1] & idx;
    dp -= (dp >> 1) & 0x5555555555555555;
    dp = (dp & 0x3333333333333333) + ((dp >> 2) & 0x3333333333333333);
    dp = (dp + (dp >> 4)) & 0x0f0f0f0f0f0f0f0f;
    dp = (dp * 0x0101010101010101) >> 56;
    dp = dp & 1;
    dp_acc = dp_acc ^ dp;
    return ((double)dp_acc) * 2.0 - 1.0;
}

/*
 * binary search: find k_star such that
 *          / argmin(i): p[i] >= val, if val <= p[k-1]
 * k_star = |
 *          \ k - 1                 , otherwise
 * R's findInterval() can not be used, so roll by hand:
 * - val: value to be found
 * - p: array
 * - k: length of p
 */
int bin_search(int val, const int *p, int k)
{
    int left, right, mid, k_star;

    left = 0;
    right = k - 1;
    k_star = k;
    while (1)
    {
        mid = (right + left) / 2;
        if (left == right)
        {
            if (p[mid] >= val)
                k_star = mid;
            return k_star;
        }
        else if (p[mid] == val)
        {
            return mid;
        }
        else if (p[mid] < val)
        {
            left = mid + 1;
        }
        else
        {
            right = mid;
        }
    }
}

/* 
 * R can only get us 31bit unsigned integers, so we have to stitch four 16bit
 * ones together to get a full random 64bit uint.
 */
uint_fast64_t
lcg_init()
{
    uint_fast64_t x;
    double upper;

    upper = (double)(1 << 15);
    x = (uint_fast64_t)floor(runif(0.0, upper));
    x = x | (((uint_fast64_t)floor(runif(0.0, upper))) << 16);
    x = x | (((uint_fast64_t)floor(runif(0.0, upper))) << 32);
    x = x | (((uint_fast64_t)floor(runif(0.0, upper))) << 48);
    return x;
}

/*
 * As noted in _Writing R Extensions_, _Writing portable packages_:
 *
 *   Compiled code should not call the system random number generators
 *   such as rand, drand48 and random76, but rather use the interfaces
 *   to R’s RNGs described in Random numbers. In particular, if more
 *   than one package initializes the system RNG (e.g. via srand),
 *   they will interfere with each other.  Nor should the C++11 random
 *   number library be used, nor any other third-party random number
 *   generators such as those in GSL.
 *
 * Unfortunately, R's API does not expose generating uniform random
 * integers (sample.int), so a somewhat ugly wrapper is necessary to
 * plug into R's random number generation.
 *
 * Any R-callable function using this *must* call GetRNGstate() before using
 * ruint() for the first time and hast do call PutRNGstate() before
 * return()ing.
 */

uint_fast64_t
ruint()
{
    uint_fast64_t rint;

    rint = (uint_fast64_t)floor(runif(0.0, (double)BCHDOMAIN));
    return rint;
}

/*
 * Matrix multiplication (modeled after R's simple_matprod). Pointers *x, *y,
 * *z point to already allocated matrix SEXP data. A row block offset in y can
 * be specified via the row offset ory.
 */
void matprod_block(double *x, int nrx, int ncx,
                   double *y, int nry, int ncy, int ory,
                   double *z)
{
    double sum1, sum2, sum3, sum4;
    int i, j, k;
    int t1;

    for (i = 0; i < nrx; i++)
    {
        for (k = 0; k < ncy; k++)
        {
            sum1 = sum2 = sum3 = sum4 = 0.0;
            t1 = ory + k * nry; /* common adress offset in y */
            for (j = 0; j < ncx % 4; j++)
            { /* first elements of unrolled loop */
                sum1 += x[i + j * nrx] * y[j + t1];
            }
            for (; j < ncx; j += 4)
            { /* remainder of unrolled loop */
                sum1 += x[i + j * nrx] * y[j + t1];
                sum2 += x[i + (j + 1) * nrx] * y[j + 1 + t1];
                sum3 += x[i + (j + 2) * nrx] * y[j + 2 + t1];
                sum4 += x[i + (j + 3) * nrx] * y[j + 3 + t1];
            }
            z[i + k * nrx] = sum1 + sum2 + sum3 + sum4;
        }
    }
}

/*
  like matprod_block but for x stored in *row_major* order.
 */
void matprod_block_xrm(double *x, size_t nrx, size_t ncx,
                       double *y, size_t nry, size_t ncy, size_t ory,
                       double *z)
{
    double sum1, sum2, sum3, sum4;
    size_t i, j, k;
    size_t t1, t2;

    for (i = 0; i < nrx; i++)
    {
        t2 = i * ncx; /* common adress offset in x */
        for (k = 0; k < ncy; k++)
        {
            sum1 = sum2 = sum3 = sum4 = 0.0;
            t1 = ory + k * nry; /* common adress offset in y */
            for (j = 0; j < ncx % 4; j++)
            { /* first elements of unrolled loop */
                sum1 += x[i + j * nrx] * y[j + t1];
            }
            for (; j < ncx; j += 4)
            { /* remainder of unrolled loop */
                sum1 += x[t2 + j] * y[j + t1];
                sum2 += x[t2 + j + 1] * y[j + 1 + t1];
                sum3 += x[t2 + j + 2] * y[j + 2 + t1];
                sum4 += x[t2 + j + 3] * y[j + 3 + t1];
            }
            z[i + k * nrx] = sum1 + sum2 + sum3 + sum4;
        }
    }
}

/* Sample n integers without replacement from {0, ..., N-1}
 *
 * This implements Knuth's Algorithm S (TAOCP vol. 2, 3.4.2)
 */
void sample_int(int n, int N, int *out)
{
    int t, m, i;

    m = 0;
    i = 0;
    for (t = 1; t <= N; t++)
    {
        if ((N - t) * unif_rand() < n - m)
        {                   /* select */
            out[i] = t - 1; /* map numbers from {1, ... N} to {0, ..., N-1} */
            ++i;
            if (++m == n)
            {
                return;
            }
        }
    }
}

/* Calculate a Clarkson Woodruff sketch.
 *
 * Proper SEXP types must be ensured before calling from R.
 *  - data: class matrix, storage mode double
 *  - sketch_rows: scalar integer
 */
void sketch_cw(const Matrix &data, size_t sketch_rows, Matrix &sketch)
{
    BCH_conf bch;
    double *s_elt, *d_elt, sgn;
    size_t i, h_i, j;
    size_t s_rows, cols, d_rows;
    uint_fast64_t rnd, a, b, m;

    // GetRNGstate(); /* init for ruint() */
    // dim = getAttrib(data, R_DimSymbol);
    d_rows = data.rows();
    cols = data.columns();
    s_rows = sketch_rows;
    d_elt = data.data(); // d_elt = REAL(data);
    /* initialise BCH generator */
    bch = bch_configure(4);
    /* initialise LCG generator for fast universal hashing */
    for (m = 1; ((s_rows - 1) >> m) > 0; m++)
    {
        /* s_rows is a power of 2, find the position of the 1-bit ... */
    }
    /* ... and generate seeds for the linear congruential generator*/
    a = lcg_init();
    b = lcg_init();
    /* create empty sketch and fill with zero */
    // sketch = PROTECT(allocMatrix(REALSXP, s_rows, cols));
    sketch.allocate(s_rows, cols);
    //s_elt = REAL(sketch);
    s_elt = sketch.data();
    for (i = 0; i < s_rows * cols; i++)
        s_elt[i] = 0.0;
    /* project randomly */
    for (i = 0; i < d_rows; i++)
    {
        /* calculate target row by fast universal hashing */
        rnd = a * i + b;
        h_i = rnd >> (64 - m);
        sgn = bch4_gen(i, bch);
        for (j = 0; j < cols; j++)
        {
            /* R matrices are in column major storage */
            //s_elt[h_i + j * s_rows] += sgn * d_elt[i + j * d_rows];

            // Other parts of the code assume row major storage
            s_elt[h_i * cols + j] += sgn * d_elt[i * cols + j];
        }
    }
    //PutRNGstate();
    //UNPROTECT(1); /* sketch */
}


/* Calculate a Clarkson Woodruff sketch.
 *
 * Proper SEXP types must be ensured before calling from R.
 *  - data: class matrix, storage mode double
 *  - sketch_rows: scalar integer
 */
void sketch_cw_sparse(const CsrMatrix &data, size_t sketch_rows, std::vector<std::map<size_t, double>> &sketchData)
{
    BCH_conf bch;
    double sgn;
    size_t i, h_i, s_rows, d_rows;
    uint_fast64_t rnd, a, b, m;

    d_rows = data.rows();
    s_rows = sketch_rows;

    /* initialise BCH generator */
    bch = bch_configure(4);

    /* initialise LCG generator for fast universal hashing */
    for (m = 1; ((s_rows - 1) >> m) > 0; m++)
    {
        /* s_rows is a power of 2, find the position of the 1-bit ... */
    }

    /* ... and generate seeds for the linear congruential generator*/
    a = lcg_init();
    b = lcg_init();
    
    StopWatch sw(true);

    /* create empty sketch and fill with zero */
    sketchData.resize(s_rows);
    
    /* project randomly */
    for (i = 0; i < d_rows; i++)
    {
        /* calculate target row by fast universal hashing */
        rnd = a * i + b;
        h_i = rnd >> (64 - m);
        sgn = bch4_gen(i, bch);

        // Efficient sparse row multiplication
        data.rowProductWithSign(i, sgn, sketchData[h_i]);
    }

    std::cout << "Clarkson Woodruff sketch completed in " << sw.elapsedStr() << std::endl;
}

/* Calculate a sketch based on Rademacher transforms
 *
 * Proper SEXP types must be ensured before calling from R.
 *  - data: class matrix, storage mode double
 *  - sketch_rows: scalar integer
 */
void sketch_rad(const Matrix &data, size_t sketch_rows, Matrix &sketch)
{
    BCH_conf bch;
    Matrix r_part, p_part;
    double *s_elt, *d_elt, *r_elt, *p_elt, sqrt_rows;
    size_t block_max, block_rows, i, j, k;
    size_t s_rows, d_rows, cols;

    // GetRNGstate(); // getRNGState: Apply the value of .Random.seed to R's internal RNG state
    d_rows = data.rows();
    cols = data.columns();
    s_rows = sketch_rows;
    d_elt = data.data();
    /* initialise BCH generator */
    bch = bch_configure(4);
    /* create empty sketch and fill with zero */
    // sketch = PROTECT(allocMatrix(REALSXP, s_rows, cols));
    sketch.allocate(s_rows, cols);
    // s_elt = REAL(sketch);
    s_elt = sketch.data();
    for (i = 0; i < s_rows * cols; i++)
        s_elt[i] = 0.0;
    /* matrix to hold result of matrix multiplication */
    //p_part = PROTECT(allocMatrix(REALSXP, s_rows, cols));
    std::cout << "Matrix p_part. ";
    p_part.allocate(s_rows, cols);
    //p_elt = REAL(p_part);
    p_elt = p_part.data();
    /* reserve memory for projection sub-matrix */
    block_max = 256; /* Note: this is hand-tuned for execution speed */
    // r_part = PROTECT(allocMatrix(REALSXP, s_rows, block_max));
    std::cout << "Matrix r_part. ";
    r_part.allocate(s_rows, block_max);
    r_elt = r_part.data();
    for (i = 0; i < d_rows; i += block_max)
    { /* i: rows in data matrix */
        if (i + block_max < d_rows)
        {
            block_rows = block_max;
        }
        else
        {
            /* when in last and incomplete block */
            block_rows = d_rows - i;
            // UNPROTECT(1); /* r_part */
            /* set up a smaller projection matrix */
            // r_part = PROTECT(allocMatrix(REALSXP, s_rows, block_rows));
            std::cout << "Matrix r_part. ";
            r_part.allocate(s_rows, block_rows);
            // r_elt = REAL(r_part);
            r_elt = r_part.data();
        }
        /* fill in random projection matrix (in row-major order)*/
        for (j = 0; j < s_rows; j++)         /* j: rows in projection matrix */
            for (k = 0; k < block_rows; k++) /* k: cols in projection matrix */
                r_elt[j * block_rows + k] = bch4_gen(j + (i + k) * s_rows, bch);

        matprod_block_xrm(r_elt, s_rows, block_rows, d_elt, d_rows, cols, i, p_elt);

        for (j = 0; j < s_rows * cols; j++)
            s_elt[j] += p_elt[j];
    }
    /* normalize */
    sqrt_rows = sqrt((double)s_rows);
    for (j = 0; j < s_rows * cols; j++)
        s_elt[j] /= sqrt_rows;
    //PutRNGstate();
    //UNPROTECT(3); /* sketch, p_part, r_part */
    //return sketch;
}

void sketch_srht(const Matrix &data, size_t sketch_rows, Matrix &sketch)
{
    BCH_conf bch;
    Matrix tmp;
    double *s_elt, *d_elt, *t_elt, sgn, sqrt_rows;
    int s_rows, cols, d_rows, i, j, *p;
    unsigned int q;

    // GetRNGstate(); /* init for ruint() */
    // dim = getAttrib(data, R_DimSymbol);
    d_rows = data.rows();
    cols = data.columns();
    s_rows = sketch_rows;
    // d_elt = REAL(data);
    d_elt = data.data();
    /* initialise BCH generator */
    bch = bch_configure(4);
    /* create empty sketch and fill with zero */
    // sketch = PROTECT(allocMatrix(REALSXP, s_rows, cols));
    sketch.allocate(s_rows, cols);
    // s_elt = REAL(sketch);
    s_elt = sketch.data();
    for (i = 0; i < s_rows * cols; i++)
        s_elt[i] = 0.0;
    /* matrix to store data rows multiplied by +/-1 */
    //tmp = PROTECT(allocMatrix(REALSXP, d_rows, cols));
    tmp.allocate(d_rows, cols);
    // t_elt = REAL(tmp);
    t_elt = tmp.data();
    /* next power of two >= d_rows */
    q = 1;
    do
        q *= 2;
    while (q < d_rows);
    /* select row randomisation (sample without replacement) */
    p = (int *)R_alloc(s_rows, sizeof(int));
    sample_int(s_rows, q, p);
    /* multiply data rows by +/-1 */
    for (i = 0; i < d_rows; i++)
    {
        sgn = bch4_gen(i, bch);
        for (j = 0; j < cols; j++)
            t_elt[i + j * d_rows] = sgn * d_elt[i + j * d_rows];
    }
    /* recursively calculate SRHT */
    srht_rec(p, s_rows, q, t_elt, d_rows, cols, 0, s_elt, s_rows, 0);
    /* normalize */
    sqrt_rows = sqrt((double)s_rows);
    for (j = 0; j < s_rows * cols; j++)
        s_elt[j] /= sqrt_rows;
    //PutRNGstate();
    //UNPROTECT(2); /* sketch, tmp */
    //return sketch;
}

void srht_rec(const int *p, int k, int q,
              double *data, int d_rows, int d_cols, int d_offset,
              double *res, int r_rows, int r_offset)
{
    int cnt, atmp, btmp, k_new, q2;
    unsigned int i, j;
    double hd, *d_sub;

    if (k == 0)
        return;
    if (k == 1)
    {
        cnt = 0;
        for (j = 0; j < d_cols; j++)
        { /* 1st col of hadamard matrix is ones */
            res[r_offset + j * r_rows] += data[0 + j * d_rows];
        }
        for (i = 1; i < d_rows; i++)
        {
            btmp = i;
            atmp = p[0];
            while ((btmp & 1) == 0)
            {
                cnt ^= atmp & 1;
                atmp >>= 1;
                btmp >>= 1;
            }
            cnt ^= atmp & 1;
            hd = cnt ? -1.0 : 1.0;
            for (j = 0; j < d_cols; j++)
                res[r_offset + j * r_rows] += hd * data[i + j * d_rows];
        }
        return;
    }
    q2 = q / 2;
    k_new = bin_search(d_offset + q2, p, k);
    d_sub = (double *)R_alloc(q2 * d_cols, sizeof(double));
    /* first recursion: add lower half to upper half */
    for (j = 0; j < d_cols; j++)
        for (i = 0; i < q2; i++)
            d_sub[i + j * q2] = data[i + j * d_rows] + (i + q2 >= d_rows ? 0 : data[i + q2 + j * d_rows]);
    srht_rec(p, k_new, q2,
             d_sub, q2, d_cols, d_offset,
             res, r_rows, r_offset);
    /* second recursion: subtract lower half from upper half */
    for (j = 0; j < d_cols; j++)
        for (i = 0; i < q2; i++)
            d_sub[i + j * q2] = data[i + j * d_rows] - (i + q2 >= d_rows ? 0 : data[i + q2 + j * d_rows]);
    srht_rec(p + k_new, k - k_new, q2,
             d_sub, q2, d_cols, d_offset + q2,
             res, r_rows, r_offset + k_new);
}

#endif