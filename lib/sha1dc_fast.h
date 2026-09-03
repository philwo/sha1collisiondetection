/*
 * Internal interface of the hardware-accelerated fast path (see
 * sha1dc_fast_x86.c and sha1dc_fast_arm64.c). Included by sha1.c, the
 * two backends and sha1perf.c; the enable guard below is the only place
 * where the compile condition is spelled out, so the extern symbols
 * exist exactly when sha1.c uses them.
 *
 * SHA1DC_FAST_SHANI: the fast path is compiled in.
 * SHA1DC_FAST_HAVE_TIER1: the x86-64 backend has an extra AVX2 tier
 * (scan8/fused8, level 1) for CPUs without AVX-512; aarch64 has only
 * levels 0 and 2.
 *
 * sha1.c includes this header after its endianness detection, so
 * SHA1DC_BIGENDIAN is honoured there; the backends and sha1perf never
 * define it and are covered by the __BYTE_ORDER__ and
 * SHA1DC_FORCE_BIGENDIAN terms. The compiler floors cover
 * __get_cpuid_count and the AVX-512BW intrinsics (GCC >= 7) and the
 * aarch64 sha2 target attribute (GCC >= 8).
 */
#ifndef SHA1DC_FAST_H
#define SHA1DC_FAST_H

#ifndef SHA1DC_NO_STANDARD_INCLUDES
#include <stdint.h>
#include <stddef.h>
#endif

#if !defined(SHA1DC_NO_FAST_SHANI) && !defined(SHA1DC_FORCE_BIGENDIAN) && \
    !defined(SHA1DC_BIGENDIAN) && defined(__GNUC__) && \
    (!defined(__BYTE_ORDER__) || __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__)
#if defined(__x86_64__) && (defined(__clang__) || __GNUC__ >= 7)
#define SHA1DC_FAST_SHANI 1
#define SHA1DC_FAST_HAVE_TIER1 1
#elif defined(__aarch64__) && \
      (defined(__ARM_FEATURE_SHA2) || defined(__clang__) || __GNUC__ >= 8)
#define SHA1DC_FAST_SHANI 1
#endif
#endif

#ifdef SHA1DC_FAST_SHANI

/* Dispatch level: 0 = off, 1 = SHA-NI+AVX2 (x86-64 only), 2 = SHA-NI+AVX-512
 * or FEAT_SHA1+NEON. Set once at startup by sha1.c from
 * sha1dc_fast_detect_level() and the SHA1DC_NO_FAST / SHA1DC_FAST_LEVEL
 * environment variables. */
extern int sha1dc_fast_level;

/* Backend CPU detection: the highest level this CPU supports. */
int sha1dc_fast_detect_level(void);

/* Compress a whole number of blocks with the hardware instructions. */
void sha1dc_fast_compress(uint32_t ihv[5], const unsigned char *data, size_t len);

/* Compress nblocks (<= 16), storing the chaining value entering block i
 * into ckpt[i]. */
void sha1dc_fast_compress_ckpt(uint32_t ihv[5], const unsigned char *p,
			       unsigned nblocks, uint32_t ckpt[][5]);

/* Scan a group of consecutive blocks: dvout[i] receives block i's ubc_check
 * dvmask, the return value is the bitmask of flagged (non-zero) blocks. */
typedef uint32_t (*sha1dc_fast_scan_fn)(const unsigned char *p, uint32_t *dvout);

/* Compress the group at 'cur' (checkpointing like sha1dc_fast_compress_ckpt)
 * while scanning the group at 'next' as sha1dc_fast_scan_fn does. */
typedef uint32_t (*sha1dc_fast_fused_fn)(uint32_t ihv[5], const unsigned char *cur,
					 const unsigned char *next, uint32_t *dvout,
					 uint32_t ckpt[][5]);

uint32_t sha1dc_fast_scan16(const unsigned char *p, uint32_t *dvout);
uint32_t sha1dc_fast_fused16(uint32_t ihv[5], const unsigned char *cur,
			     const unsigned char *next, uint32_t *dvout,
			     uint32_t ckpt[][5]);
#ifdef SHA1DC_FAST_HAVE_TIER1
uint32_t sha1dc_fast_scan8(const unsigned char *p, uint32_t *dvout);
uint32_t sha1dc_fast_fused8(uint32_t ihv[5], const unsigned char *cur,
			    const unsigned char *next, uint32_t *dvout,
			    uint32_t ckpt[][5]);
#endif

/* For a flagged block: the expanded message W[80] and the compression
 * states entering steps 58 and 65, in the layout sha1_recompression_step
 * expects (see sha1dc_fast_states_tail in sha1dc_fast_impl.h). */
void sha1dc_fast_states(const unsigned char *block, uint32_t W[80],
			const uint32_t ihvin[5],
			uint32_t state58[5], uint32_t state65[5]);

#endif /* SHA1DC_FAST_SHANI */
#endif /* SHA1DC_FAST_H */
