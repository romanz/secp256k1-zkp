/*****************************************************************************
 * Copyright (c) 2013, 2014, 2017 Pieter Wuille, Andrew Poelstra, Jonas Nick *
 * Distributed under the MIT software license, see the accompanying          *
 * file COPYING or http://www.opensource.org/licenses/mit-license.php.       *
 *****************************************************************************/

#ifndef SECP256K1_ECMULT_IMPL_H
#define SECP256K1_ECMULT_IMPL_H

#include <string.h>
#include <stdint.h>

#include "util.h"
#include "group.h"
#include "scalar.h"
#include "ecmult.h"

#if defined(EXHAUSTIVE_TEST_ORDER)
/* We need to lower these values for exhaustive tests because
 * the tables cannot have infinities in them (this breaks the
 * affine-isomorphism stuff which tracks z-ratios) */
#  if EXHAUSTIVE_TEST_ORDER > 128
#    define WINDOW_A 5
#    define WINDOW_G 8
#  elif EXHAUSTIVE_TEST_ORDER > 8
#    define WINDOW_A 4
#    define WINDOW_G 4
#  else
#    define WINDOW_A 2
#    define WINDOW_G 2
#  endif
#else
/* optimal for 128-bit and 256-bit exponents. */
#define WINDOW_A 5
/** larger numbers may result in slightly better performance, at the cost of
    exponentially larger precomputed tables. */
#ifdef USE_ENDOMORPHISM
/** Two tables for window size 15: 1.375 MiB. */
#define WINDOW_G 15
#else
/** One table for window size 16: 1.375 MiB. */
#define WINDOW_G 16
#endif
#endif

#ifdef USE_ENDOMORPHISM
    #define WNAF_BITS 128
#else
    #define WNAF_BITS 256
#endif
#define WNAF_SIZE_BITS(bits, w) (((bits) + (w) - 1) / (w))
#define WNAF_SIZE(w) WNAF_SIZE_BITS(WNAF_BITS, w)

/** The number of entries a table with precomputed multiples needs to have. */
#define ECMULT_TABLE_SIZE(w) (1 << ((w)-2))

/* The number of objects allocated on the scratch space for ecmult_multi algorithms */
#define PIPPENGER_SCRATCH_OBJECTS 6
#define STRAUSS_SCRATCH_OBJECTS 6

#define PIPPENGER_MAX_BUCKET_WINDOW 12

/* Minimum number of points for which pippenger_wnaf is faster than strauss wnaf */
#ifdef USE_ENDOMORPHISM
    #define ECMULT_PIPPENGER_THRESHOLD 88
#else
    #define ECMULT_PIPPENGER_THRESHOLD 160
#endif

#ifdef USE_ENDOMORPHISM
    #define ECMULT_MAX_POINTS_PER_BATCH 5000000
#else
    #define ECMULT_MAX_POINTS_PER_BATCH 10000000
#endif

/** Fill a table 'prej' with precomputed odd multiples of a. Prej will contain
 *  the values [1*a,3*a,...,(2*n-1)*a], so it space for n values. zr[0] will
 *  contain prej[0].z / a.z. The other zr[i] values = prej[i].z / prej[i-1].z.
 *  Prej's Z values are undefined, except for the last value.
 */
static void secp256k1_ecmult_odd_multiples_table(int n, secp256k1_gej *prej, secp256k1_fe *zr, const secp256k1_gej *a) {
    secp256k1_gej d;
    secp256k1_ge a_ge, d_ge;
    int i;

    VERIFY_CHECK(!a->infinity);

    secp256k1_gej_double_var(&d, a, NULL);

    /*
     * Perform the additions on an isomorphism where 'd' is affine: drop the z coordinate
     * of 'd', and scale the 1P starting value's x/y coordinates without changing its z.
     */
    d_ge.x = d.x;
    d_ge.y = d.y;
    d_ge.infinity = 0;

    secp256k1_ge_set_gej_zinv(&a_ge, a, &d.z);
    prej[0].x = a_ge.x;
    prej[0].y = a_ge.y;
    prej[0].z = a->z;
    prej[0].infinity = 0;

    zr[0] = d.z;
    for (i = 1; i < n; i++) {
        secp256k1_gej_add_ge_var(&prej[i], &prej[i-1], &d_ge, &zr[i]);
    }

    /*
     * Each point in 'prej' has a z coordinate too small by a factor of 'd.z'. Only
     * the final point's z coordinate is actually used though, so just update that.
     */
    secp256k1_fe_mul(&prej[n-1].z, &prej[n-1].z, &d.z);
}

/** Fill a table 'pre' with precomputed odd multiples of a.
 *
 *  There are two versions of this function:
 *  - secp256k1_ecmult_odd_multiples_table_globalz_windowa which brings its
 *    resulting point set to a single constant Z denominator, stores the X and Y
 *    coordinates as ge_storage points in pre, and stores the global Z in rz.
 *    It only operates on tables sized for WINDOW_A wnaf multiples.
 *  - secp256k1_ecmult_odd_multiples_table_storage_var, which converts its
 *    resulting point set to actually affine points, and stores those in pre.
 *    It operates on tables of any size, but uses heap-allocated temporaries.
 *
 *  To compute a*P + b*G, we compute a table for P using the first function,
 *  and for G using the second (which requires an inverse, but it only needs to
 *  happen once).
 */
static void secp256k1_ecmult_odd_multiples_table_globalz_windowa(secp256k1_ge *pre, secp256k1_fe *globalz, const secp256k1_gej *a) {
    secp256k1_gej prej[ECMULT_TABLE_SIZE(WINDOW_A)];
    secp256k1_fe zr[ECMULT_TABLE_SIZE(WINDOW_A)];

    /* Compute the odd multiples in Jacobian form. */
    secp256k1_ecmult_odd_multiples_table(ECMULT_TABLE_SIZE(WINDOW_A), prej, zr, a);
    /* Bring them to the same Z denominator. */
    secp256k1_ge_globalz_set_table_gej(ECMULT_TABLE_SIZE(WINDOW_A), pre, globalz, prej, zr);
}

static void secp256k1_ecmult_odd_multiples_table_storage_var(const int n, secp256k1_ge_storage *pre, const secp256k1_gej *a) {
    secp256k1_gej d;
    secp256k1_ge a_ge, d_ge, p_ge;
    secp256k1_ge last_ge;
    secp256k1_gej pj;
    secp256k1_fe zi;
    secp256k1_fe zr;
    secp256k1_fe dx_over_dz_squared;
    int i;

    VERIFY_CHECK(!a->infinity);

    secp256k1_gej_double_var(&d, a, NULL);

    /* First, we perform all the additions in an isomorphic curve obtained by multiplying
     * all `z` coordinates by 1/`d.z`. In these coordinates `d` is affine so we can use
     * `secp256k1_gej_add_ge_var` to perform the additions. For each addition, we store
     * the resulting y-coordinate and the z-ratio, since we only have enough memory to
     * store two field elements. These are sufficient to efficiently undo the isomorphism
     * and recompute all the `x`s.
     */
    d_ge.x = d.x;
    d_ge.y = d.y;
    d_ge.infinity = 0;

    secp256k1_ge_set_gej_zinv(&a_ge, a, &d.z);
    pj.x = a_ge.x;
    pj.y = a_ge.y;
    pj.z = a->z;
    pj.infinity = 0;

    zr = d.z;
    secp256k1_fe_normalize(&zr);
    secp256k1_fe_to_storage(&pre[0].x, &zr);
    secp256k1_fe_normalize(&pj.y);
    secp256k1_fe_to_storage(&pre[0].y, &pj.y);

    for (i = 1; i < n; i++) {
        secp256k1_gej_add_ge_var(&pj, &pj, &d_ge, &zr);
        secp256k1_fe_normalize(&zr);
        secp256k1_fe_to_storage(&pre[i].x, &zr);
        secp256k1_fe_normalize(&pj.y);
        secp256k1_fe_to_storage(&pre[i].y, &pj.y);
    }

    /* Map `pj` back to our curve by multiplying its z-coordinate by `d.z`. */
    zr = pj.z; /* save pj.z so we can use it to extract (d.z)^-1 from zi */
    secp256k1_fe_mul(&pj.z, &pj.z, &d.z);
    /* Directly set `pre[n - 1]` to `pj`, saving the inverted z-coordinate so
     * that we can combine it with the saved z-ratios to compute the other zs
     * without any more inversions. */
    secp256k1_fe_inv_var(&zi, &pj.z);
    secp256k1_ge_set_gej_zinv(&p_ge, &pj, &zi);
    secp256k1_ge_from_storage(&last_ge, &pre[n - 1]);
    secp256k1_ge_to_storage(&pre[n - 1], &p_ge);

    /* Compute the actual x-coordinate of D, which will be needed below. */
    secp256k1_fe_mul(&d.z, &zi, &zr);  /* d.z = 1/d.z */
    secp256k1_fe_sqr(&dx_over_dz_squared, &d.z);
    secp256k1_fe_mul(&dx_over_dz_squared, &dx_over_dz_squared, &d.x);

    i = n - 1;
    while (i > 0) {
        secp256k1_fe zi2, zi3;
        i--;
        /* For the remaining points, we extract the z-ratio from the stored
         * x-coordinate, compute its z^-1 from that, and compute the full
         * point from that: */
        secp256k1_fe_mul(&zi, &zi, &last_ge.x);
        secp256k1_fe_sqr(&zi2, &zi);
        secp256k1_fe_mul(&zi3, &zi2, &zi);
        /* To compute x, we observe that the z-ratio is simply `h` from
         * `gej_add_ge_var` which is equal to `d_x * z^2 - x`, where
         * `d_x` is the x coordinate of `D` and `x`, `z` are the Jacobian
         * coordinates of our desired point. Rearranging and dividing by
         * `z^2` to convert to affine, we get
         *
         *     x = d_x - rzr / z^2
         *       = d_x - rzr * zi2
         */
        secp256k1_fe_mul(&p_ge.x, &last_ge.x, &zi2);
        secp256k1_fe_negate(&p_ge.x, &p_ge.x, 1);
        secp256k1_fe_add(&p_ge.x, &dx_over_dz_squared);
        /* y is stored_y/z^3, as we expect */
        secp256k1_ge_from_storage(&last_ge, &pre[i]);
        secp256k1_fe_mul(&p_ge.y, &last_ge.y, &zi3);
        /* Store */
        secp256k1_ge_to_storage(&pre[i], &p_ge);
    }
}

/** The following two macro retrieves a particular odd multiple from a table
 *  of precomputed multiples. */
#define ECMULT_TABLE_GET_GE(r,pre,n,w) do { \
    VERIFY_CHECK(((n) & 1) == 1); \
    VERIFY_CHECK((n) >= -((1 << ((w)-1)) - 1)); \
    VERIFY_CHECK((n) <=  ((1 << ((w)-1)) - 1)); \
    if ((n) > 0) { \
        *(r) = (pre)[((n)-1)/2]; \
    } else { \
        secp256k1_ge_neg((r), &(pre)[(-(n)-1)/2]); \
    } \
} while(0)

#define ECMULT_TABLE_GET_GE_STORAGE(r,pre,n,w) do { \
    VERIFY_CHECK(((n) & 1) == 1); \
    VERIFY_CHECK((n) >= -((1 << ((w)-1)) - 1)); \
    VERIFY_CHECK((n) <=  ((1 << ((w)-1)) - 1)); \
    if ((n) > 0) { \
        secp256k1_ge_from_storage((r), &(pre)[((n)-1)/2]); \
    } else { \
        secp256k1_ge_from_storage((r), &(pre)[(-(n)-1)/2]); \
        secp256k1_ge_neg((r), (r)); \
    } \
} while(0)

static size_t secp256k1_ecmult_context_preallocated_size(void) {
    int ret = ROUND_TO_ALIGN(sizeof((*((secp256k1_ecmult_context*) NULL)->pre_g)[0]) * ECMULT_TABLE_SIZE(WINDOW_G));
#ifdef USE_ENDOMORPHISM
    ret += ROUND_TO_ALIGN(sizeof((*((secp256k1_ecmult_context*) NULL)->pre_g_128)[0]) * ECMULT_TABLE_SIZE(WINDOW_G));
#endif
    return ret;
}

static void secp256k1_ecmult_context_init(secp256k1_ecmult_context *ctx) {
    ctx->pre_g = NULL;
#ifdef USE_ENDOMORPHISM
    ctx->pre_g_128 = NULL;
#endif
}

static void secp256k1_ecmult_context_build(secp256k1_ecmult_context *ctx, void **prealloc) {
    secp256k1_gej gj;
    void* const base = *prealloc;
    size_t const prealloc_size = secp256k1_ecmult_context_preallocated_size();

    if (ctx->pre_g != NULL) {
        return;
    }

    /* get the generator */
    secp256k1_gej_set_ge(&gj, &secp256k1_ge_const_g);

    ctx->pre_g = (secp256k1_ge_storage (*)[])manual_malloc(prealloc, sizeof((*ctx->pre_g)[0]) * ECMULT_TABLE_SIZE(WINDOW_G), base, prealloc_size);

    /* precompute the tables with odd multiples */
    secp256k1_ecmult_odd_multiples_table_storage_var(ECMULT_TABLE_SIZE(WINDOW_G), *ctx->pre_g, &gj);

#ifdef USE_ENDOMORPHISM
    {
        secp256k1_gej g_128j;
        int i;

        ctx->pre_g_128 = (secp256k1_ge_storage (*)[])manual_malloc(prealloc, sizeof((*ctx->pre_g_128)[0]) * ECMULT_TABLE_SIZE(WINDOW_G), base, prealloc_size);

        /* calculate 2^128*generator */
        g_128j = gj;
        for (i = 0; i < 128; i++) {
            secp256k1_gej_double_var(&g_128j, &g_128j, NULL);
        }
        secp256k1_ecmult_odd_multiples_table_storage_var(ECMULT_TABLE_SIZE(WINDOW_G), *ctx->pre_g_128, &g_128j);
    }
#endif
}

static void secp256k1_ecmult_context_finalize_memcpy(secp256k1_ecmult_context *dst, const secp256k1_ecmult_context *src) {
    if (src->pre_g != NULL) {
        dst->pre_g = (secp256k1_ge_storage (*)[])((unsigned char*)dst + ((unsigned char*)(src->pre_g) - (unsigned char*)src));
    }
#ifdef USE_ENDOMORPHISM
    if (src->pre_g_128 != NULL) {
        dst->pre_g_128 = (secp256k1_ge_storage (*)[])((unsigned char*)dst + ((unsigned char*)(src->pre_g_128) - (unsigned char*)src));
    }
#endif
}

static int secp256k1_ecmult_context_is_built(const secp256k1_ecmult_context *ctx) {
    return ctx->pre_g != NULL;
}

static void secp256k1_ecmult_context_clear(secp256k1_ecmult_context *ctx) {
    secp256k1_ecmult_context_init(ctx);
}

/** Convert a number to WNAF notation. The number becomes represented by sum(2^i * wnaf[i], i=0..bits),
 *  with the following guarantees:
 *  - each wnaf[i] is either 0, or an odd integer between -(1<<(w-1) - 1) and (1<<(w-1) - 1)
 *  - two non-zero entries in wnaf are separated by at least w-1 zeroes.
 *  - the number of set values in wnaf is returned. This number is at most 256, and at most one more
 *    than the number of bits in the (absolute value) of the input.
 */
static int secp256k1_ecmult_wnaf(int *wnaf, int len, const secp256k1_scalar *a, int w) {
    secp256k1_scalar s = *a;
    int last_set_bit = -1;
    int bit = 0;
    int sign = 1;
    int carry = 0;

    VERIFY_CHECK(wnaf != NULL);
    VERIFY_CHECK(0 <= len && len <= 256);
    VERIFY_CHECK(a != NULL);
    VERIFY_CHECK(2 <= w && w <= 31);

    memset(wnaf, 0, len * sizeof(wnaf[0]));

    if (secp256k1_scalar_get_bits(&s, 255, 1)) {
        secp256k1_scalar_negate(&s, &s);
        sign = -1;
    }

    while (bit < len) {
        int now;
        int word;
        if (secp256k1_scalar_get_bits(&s, bit, 1) == (unsigned int)carry) {
            bit++;
            continue;
        }

        now = w;
        if (now > len - bit) {
            now = len - bit;
        }

        word = secp256k1_scalar_get_bits_var(&s, bit, now) + carry;

        carry = (word >> (w-1)) & 1;
        word -= carry << w;

        wnaf[bit] = sign * word;
        last_set_bit = bit;

        bit += now;
    }
#ifdef VERIFY
    CHECK(carry == 0);
    while (bit < 256) {
        CHECK(secp256k1_scalar_get_bits(&s, bit++, 1) == 0);
    }
#endif
    return last_set_bit + 1;
}

struct secp256k1_strauss_point_state {
#ifdef USE_ENDOMORPHISM
    secp256k1_scalar na_1, na_lam;
    int wnaf_na_1[130];
    int wnaf_na_lam[130];
    int bits_na_1;
    int bits_na_lam;
#else
    int wnaf_na[256];
    int bits_na;
#endif
    size_t input_pos;
};

struct secp256k1_strauss_state {
    secp256k1_gej* prej;
    secp256k1_fe* zr;
    secp256k1_ge* pre_a;
#ifdef USE_ENDOMORPHISM
    secp256k1_ge* pre_a_lam;
#endif
    struct secp256k1_strauss_point_state* ps;
};

static void secp256k1_ecmult_strauss_wnaf(const secp256k1_ecmult_context *ctx, const struct secp256k1_strauss_state *state, secp256k1_gej *r, int num, const secp256k1_gej *a, const secp256k1_scalar *na, const secp256k1_scalar *ng) {
    secp256k1_ge tmpa;
    secp256k1_fe Z;
#ifdef USE_ENDOMORPHISM
    /* Splitted G factors. */
    secp256k1_scalar ng_1, ng_128;
    int wnaf_ng_1[129];
    int bits_ng_1 = 0;
    int wnaf_ng_128[129];
    int bits_ng_128 = 0;
#else
    int wnaf_ng[256];
    int bits_ng = 0;
#endif
    int i;
    int bits = 0;
    int np;
    int no = 0;

    for (np = 0; np < num; ++np) {
        if (secp256k1_scalar_is_zero(&na[np]) || secp256k1_gej_is_infinity(&a[np])) {
            continue;
        }
        state->ps[no].input_pos = np;
#ifdef USE_ENDOMORPHISM
        /* split na into na_1 and na_lam (where na = na_1 + na_lam*lambda, and na_1 and na_lam are ~128 bit) */
        secp256k1_scalar_split_lambda(&state->ps[no].na_1, &state->ps[no].na_lam, &na[np]);

        /* build wnaf representation for na_1 and na_lam. */
        state->ps[no].bits_na_1   = secp256k1_ecmult_wnaf(state->ps[no].wnaf_na_1,   130, &state->ps[no].na_1,   WINDOW_A);
        state->ps[no].bits_na_lam = secp256k1_ecmult_wnaf(state->ps[no].wnaf_na_lam, 130, &state->ps[no].na_lam, WINDOW_A);
        VERIFY_CHECK(state->ps[no].bits_na_1 <= 130);
        VERIFY_CHECK(state->ps[no].bits_na_lam <= 130);
        if (state->ps[no].bits_na_1 > bits) {
            bits = state->ps[no].bits_na_1;
        }
        if (state->ps[no].bits_na_lam > bits) {
            bits = state->ps[no].bits_na_lam;
        }
#else
        /* build wnaf representation for na. */
        state->ps[no].bits_na     = secp256k1_ecmult_wnaf(state->ps[no].wnaf_na,     256, &na[np],      WINDOW_A);
        if (state->ps[no].bits_na > bits) {
            bits = state->ps[no].bits_na;
        }
#endif
        ++no;
    }

    /* Calculate odd multiples of a.
     * All multiples are brought to the same Z 'denominator', which is stored
     * in Z. Due to secp256k1' isomorphism we can do all operations pretending
     * that the Z coordinate was 1, use affine addition formulae, and correct
     * the Z coordinate of the result once at the end.
     * The exception is the precomputed G table points, which are actually
     * affine. Compared to the base used for other points, they have a Z ratio
     * of 1/Z, so we can use secp256k1_gej_add_zinv_var, which uses the same
     * isomorphism to efficiently add with a known Z inverse.
     */
    if (no > 0) {
        /* Compute the odd multiples in Jacobian form. */
        secp256k1_ecmult_odd_multiples_table(ECMULT_TABLE_SIZE(WINDOW_A), state->prej, state->zr, &a[state->ps[0].input_pos]);
        for (np = 1; np < no; ++np) {
            secp256k1_gej tmp = a[state->ps[np].input_pos];
#ifdef VERIFY
            secp256k1_fe_normalize_var(&(state->prej[(np - 1) * ECMULT_TABLE_SIZE(WINDOW_A) + ECMULT_TABLE_SIZE(WINDOW_A) - 1].z));
#endif
            secp256k1_gej_rescale(&tmp, &(state->prej[(np - 1) * ECMULT_TABLE_SIZE(WINDOW_A) + ECMULT_TABLE_SIZE(WINDOW_A) - 1].z));
            secp256k1_ecmult_odd_multiples_table(ECMULT_TABLE_SIZE(WINDOW_A), state->prej + np * ECMULT_TABLE_SIZE(WINDOW_A), state->zr + np * ECMULT_TABLE_SIZE(WINDOW_A), &tmp);
            secp256k1_fe_mul(state->zr + np * ECMULT_TABLE_SIZE(WINDOW_A), state->zr + np * ECMULT_TABLE_SIZE(WINDOW_A), &(a[state->ps[np].input_pos].z));
        }
        /* Bring them to the same Z denominator. */
        secp256k1_ge_globalz_set_table_gej(ECMULT_TABLE_SIZE(WINDOW_A) * no, state->pre_a, &Z, state->prej, state->zr);
    } else {
        secp256k1_fe_set_int(&Z, 1);
    }

#ifdef USE_ENDOMORPHISM
    for (np = 0; np < no; ++np) {
        for (i = 0; i < ECMULT_TABLE_SIZE(WINDOW_A); i++) {
            secp256k1_ge_mul_lambda(&state->pre_a_lam[np * ECMULT_TABLE_SIZE(WINDOW_A) + i], &state->pre_a[np * ECMULT_TABLE_SIZE(WINDOW_A) + i]);
        }
    }

    if (ng) {
        /* split ng into ng_1 and ng_128 (where gn = gn_1 + gn_128*2^128, and gn_1 and gn_128 are ~128 bit) */
        secp256k1_scalar_split_128(&ng_1, &ng_128, ng);

        /* Build wnaf representation for ng_1 and ng_128 */
        bits_ng_1   = secp256k1_ecmult_wnaf(wnaf_ng_1,   129, &ng_1,   WINDOW_G);
        bits_ng_128 = secp256k1_ecmult_wnaf(wnaf_ng_128, 129, &ng_128, WINDOW_G);
        if (bits_ng_1 > bits) {
            bits = bits_ng_1;
        }
        if (bits_ng_128 > bits) {
            bits = bits_ng_128;
        }
    }
#else
    if (ng) {
        bits_ng     = secp256k1_ecmult_wnaf(wnaf_ng,     256, ng,      WINDOW_G);
        if (bits_ng > bits) {
            bits = bits_ng;
        }
    }
#endif

    secp256k1_gej_set_infinity(r);

    for (i = bits - 1; i >= 0; i--) {
        int n;
        secp256k1_gej_double_var(r, r, NULL);
#ifdef USE_ENDOMORPHISM
        for (np = 0; np < no; ++np) {
            if (i < state->ps[np].bits_na_1 && (n = state->ps[np].wnaf_na_1[i])) {
                ECMULT_TABLE_GET_GE(&tmpa, state->pre_a + np * ECMULT_TABLE_SIZE(WINDOW_A), n, WINDOW_A);
                secp256k1_gej_add_ge_var(r, r, &tmpa, NULL);
            }
            if (i < state->ps[np].bits_na_lam && (n = state->ps[np].wnaf_na_lam[i])) {
                ECMULT_TABLE_GET_GE(&tmpa, state->pre_a_lam + np * ECMULT_TABLE_SIZE(WINDOW_A), n, WINDOW_A);
                secp256k1_gej_add_ge_var(r, r, &tmpa, NULL);
            }
        }
        if (i < bits_ng_1 && (n = wnaf_ng_1[i])) {
            ECMULT_TABLE_GET_GE_STORAGE(&tmpa, *ctx->pre_g, n, WINDOW_G);
            secp256k1_gej_add_zinv_var(r, r, &tmpa, &Z);
        }
        if (i < bits_ng_128 && (n = wnaf_ng_128[i])) {
            ECMULT_TABLE_GET_GE_STORAGE(&tmpa, *ctx->pre_g_128, n, WINDOW_G);
            secp256k1_gej_add_zinv_var(r, r, &tmpa, &Z);
        }
#else
        for (np = 0; np < no; ++np) {
            if (i < state->ps[np].bits_na && (n = state->ps[np].wnaf_na[i])) {
                ECMULT_TABLE_GET_GE(&tmpa, state->pre_a + np * ECMULT_TABLE_SIZE(WINDOW_A), n, WINDOW_A);
                secp256k1_gej_add_ge_var(r, r, &tmpa, NULL);
            }
        }
        if (i < bits_ng && (n = wnaf_ng[i])) {
            ECMULT_TABLE_GET_GE_STORAGE(&tmpa, *ctx->pre_g, n, WINDOW_G);
            secp256k1_gej_add_zinv_var(r, r, &tmpa, &Z);
        }
#endif
    }

    if (!r->infinity) {
        secp256k1_fe_mul(&r->z, &r->z, &Z);
    }
}

static void secp256k1_ecmult(const secp256k1_ecmult_context *ctx, secp256k1_gej *r, const secp256k1_gej *a, const secp256k1_scalar *na, const secp256k1_scalar *ng) {
    secp256k1_gej prej[ECMULT_TABLE_SIZE(WINDOW_A)];
    secp256k1_fe zr[ECMULT_TABLE_SIZE(WINDOW_A)];
    secp256k1_ge pre_a[ECMULT_TABLE_SIZE(WINDOW_A)];
    struct secp256k1_strauss_point_state ps[1];
#ifdef USE_ENDOMORPHISM
    secp256k1_ge pre_a_lam[ECMULT_TABLE_SIZE(WINDOW_A)];
#endif
    struct secp256k1_strauss_state state;

    state.prej = prej;
    state.zr = zr;
    state.pre_a = pre_a;
#ifdef USE_ENDOMORPHISM
    state.pre_a_lam = pre_a_lam;
#endif
    state.ps = ps;
    secp256k1_ecmult_strauss_wnaf(ctx, &state, r, 1, a, na, ng);
}

struct secp256k1_pippenger_point_state {
    int skew_na;
    size_t input_pos;
};

struct secp256k1_pippenger_state {
    int *wnaf_na;
    struct secp256k1_pippenger_point_state* ps;
};

#ifdef USE_ENDOMORPHISM
SECP256K1_INLINE static void secp256k1_ecmult_endo_split(secp256k1_scalar *s1, secp256k1_scalar *s2, secp256k1_ge *p1, secp256k1_ge *p2) {
    secp256k1_scalar tmp = *s1;
    secp256k1_scalar_split_lambda(s1, s2, &tmp);
    secp256k1_ge_mul_lambda(p2, p1);

    if (secp256k1_scalar_is_high(s1)) {
        secp256k1_scalar_negate(s1, s1);
        secp256k1_ge_neg(p1, p1);
    }
    if (secp256k1_scalar_is_high(s2)) {
        secp256k1_scalar_negate(s2, s2);
        secp256k1_ge_neg(p2, p2);
    }
}
#endif

typedef int (*secp256k1_ecmult_multi_func)(const secp256k1_ecmult_context*, secp256k1_scratch*, secp256k1_gej*, const secp256k1_scalar*, secp256k1_ecmult_multi_callback cb, void*, size_t);

#endif /* SECP256K1_ECMULT_IMPL_H */
