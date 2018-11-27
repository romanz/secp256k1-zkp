#ifndef SECP256K1_PREALLOC_H
#define SECP256K1_PREALLOC_H

#include "secp256k1.h"

#ifdef __cplusplus
extern "C" {
#endif

/** Determine the memory size of a secp256k1 context object to be created in
 *  caller-provided memory.
 *
 *  The purpose of this function is to determine how much memory must be provided
 *  to secp256k1_context_prealloc_create.
 *
 *  Returns: the required size of the caller-provided memory block
 *  In:      flags:    which parts of the context to initialize.
 */
SECP256K1_API size_t secp256k1_context_prealloc_size(
    unsigned int flags
) SECP256K1_WARN_UNUSED_RESULT;

/** Create a secp256k1 context object in caller-provided memory.
 *
 *  Returns: a newly created context object.
 *  In:      prealloc: a pointer to a rewritable contiguous block of memory of
 *                     size at least secp256k1_context_prealloc_size(flags)
 *                     bytes, suitably aligned to hold an object of any type
 *                     (cannot be NULL)
 *           flags:    which parts of the context to initialize.
 *
 *  See also secp256k1_context_randomize.
 */
SECP256K1_API secp256k1_context* secp256k1_context_prealloc_create(
    void* prealloc,
    unsigned int flags
) SECP256K1_ARG_NONNULL(1) SECP256K1_WARN_UNUSED_RESULT;

/** Determine the memory size of a secp256k1 context object to be copied into
 *  caller-provided memory.
 *
 *  The purpose of this function is to determine how much memory must be provided
 *  to secp256k1_context_prealloc_clone when copying the context ctx.
 *
 *  Returns: the required size of the caller-provided memory block.
 *  In:      ctx: an existing context to copy (cannot be NULL)
 */
SECP256K1_API size_t secp256k1_context_prealloc_clone_size(
    const secp256k1_context* ctx
) SECP256K1_ARG_NONNULL(1) SECP256K1_WARN_UNUSED_RESULT;

/** Copy a secp256k1 context object into caller-provided memory.
 *
 *  Returns: a newly created context object.
 *  Args:    ctx:      an existing context to copy (cannot be NULL)
 *  In:      prealloc: a pointer to a rewritable contiguous block of memory of
 *                     size at least secp256k1_context_prealloc_size(flags)
 *                     bytes, suitably aligned to hold an object of any type
 *                     (cannot be NULL)
 */
SECP256K1_API secp256k1_context* secp256k1_context_prealloc_clone(
    const secp256k1_context* ctx,
    void* prealloc
) SECP256K1_ARG_NONNULL(1) SECP256K1_ARG_NONNULL(2) SECP256K1_WARN_UNUSED_RESULT;

/** Destroy a secp256k1 context object that has been created in
 *  caller-provided memory.
 *
 *  The context pointer may not be used afterwards.
 *
 *  The context to destroy must have been created using
 *  secp256k1_context_prealloc_create or secp256k1_context_prealloc_clone.
 *  If the context has instead been created using secp256k1_context_create or
 *  secp256k1_context_clone, the behaviour is undefined. In that case,
 *  secp256k1_context_destroy must be used instead.
 *
 *  Args:   ctx: an existing context to destroy, constructed using
 *               secp256k1_context_prealloc_create or
 *               secp256k1_context_prealloc_clone (cannot be NULL)
 */
SECP256K1_API void secp256k1_context_prealloc_destroy(
    secp256k1_context* ctx
);

#ifdef __cplusplus
}
#endif

#endif /* SECP256K1_PREALLOC_H */
