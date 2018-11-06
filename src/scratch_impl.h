/**********************************************************************
 * Copyright (c) 2017 Andrew Poelstra                                 *
 * Distributed under the MIT software license, see the accompanying   *
 * file COPYING or http://www.opensource.org/licenses/mit-license.php.*
 **********************************************************************/

#ifndef _SECP256K1_SCRATCH_IMPL_H_
#define _SECP256K1_SCRATCH_IMPL_H_

#include "util.h"
#include "scratch.h"

static secp256k1_scratch* secp256k1_scratch_create(const secp256k1_callback* error_callback, size_t max_size) {
    secp256k1_scratch* ret = (secp256k1_scratch*)checked_malloc(error_callback, sizeof(*ret));
    if (ret != NULL) {
        memset(ret, 0, sizeof(*ret));
        ret->max_size = max_size;
        ret->error_callback = error_callback;
    }
    return ret;
}

static void secp256k1_scratch_destroy(secp256k1_scratch* scratch) {
    if (scratch != NULL) {
        VERIFY_CHECK(scratch->frame == 0);
        free(scratch);
    }
}

#endif
