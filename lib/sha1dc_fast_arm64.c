/*
 * Hardware-accelerated fast path for sha1dc on aarch64 (used by sha1.c).
 *
 * Same design as sha1dc_fast_x86.c: ubc_check() depends only on the
 * expanded message words of a block. A block whose dvmask is zero needs
 * no recompression checks, and its compression output equals plain
 * SHA-1, so it can be computed with the ARMv8 SHA-1 crypto extensions
 * (FEAT_SHA1: sha1c/sha1p/sha1m/sha1h and sha1su0/sha1su1). The scan
 * runs on NEON with 4 blocks per pass; sha1dc_fast_scan16 and
 * sha1dc_fast_fused16 process 16-block groups by running four 4-lane
 * scans, so sha1.c can use the same group loop on both architectures.
 * There is no tier-1 (scan8/fused8) variant here: sha1dc_fast_level is
 * either 0 (off) or 2 (fast).
 *
 * The SHA-1 block step is based on code written and placed in the public
 * domain by Jeffrey Walton (sha1-arm.c, github.com/noloader/SHA-Intrinsics).
 */
#ifndef SHA1DC_NO_STANDARD_INCLUDES
#include <stdint.h>
#include <stdlib.h>
#include <stddef.h>
#endif

#ifdef SHA1DC_CUSTOM_INCLUDE_SHA1DC_FAST_ARM64_C
#include SHA1DC_CUSTOM_INCLUDE_SHA1DC_FAST_ARM64_C
#endif

#if defined(__aarch64__) && defined(__GNUC__) && !defined(SHA1DC_NO_FAST_SHANI) && \
    (!defined(__BYTE_ORDER__) || __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__)

/*
 * The vsha1* intrinsics need the sha2 target feature. Apple compilers
 * predefine __ARM_FEATURE_SHA2, so no attribute is needed there; other
 * compilers get a per-function target attribute. If neither works, the
 * file compiles to nothing and sha1.c must not define SHA1DC_FAST_SHANI
 * (its guard mirrors this condition).
 */
#if defined(__ARM_FEATURE_SHA2)
#define SHA1DC_TGT_SHA1
#define SHA1DC_FAST_ARM64 1
#elif defined(__clang__)
#define SHA1DC_TGT_SHA1 __attribute__((target("sha2")))
#define SHA1DC_FAST_ARM64 1
#elif __GNUC__ >= 8
#define SHA1DC_TGT_SHA1 __attribute__((target("+sha2")))
#define SHA1DC_FAST_ARM64 1
#endif

#ifdef SHA1DC_FAST_ARM64

#include <arm_neon.h>

#include "ubc_check_simd.h"

int sha1dc_fast_level;

#define SHA1DC_INLINE static inline __attribute__((always_inline))

/*
 * One SHA-1 block step with the ARMv8 SHA-1 instructions. ABCD holds
 * a,b,c,d with a in lane 0; E is scalar (vsha1h computes the rotate,
 * vsha1{c,p,m}q add it back in).
 *
 * With collect == 0 this compresses one block (W and pE56 are unused).
 * With collect == 1 it instead prepares the recompression-check inputs
 * for a flagged block: the message schedule still runs in full and each
 * completed 4-word group is stored to W[80] (the lanes are already in
 * logical order), but the rounds stop after round 55; *pABCD returns
 * the working state entering round 56, *pE56 the e entering round 56,
 * and *pE is not updated. collect must be a literal so the branches
 * constant-fold.
 */
SHA1DC_TGT_SHA1 SHA1DC_INLINE
void sha1neon_block1_impl(uint32x4_t *pABCD, uint32_t *pE,
			  const unsigned char *data, const int collect,
			  uint32_t *W, uint32_t *pE56)
{
	uint32x4_t ABCD = *pABCD, ABCD_SAVE = *pABCD;
	uint32_t E0 = *pE, E0_SAVE = *pE, E1;
	uint32x4_t MSG0, MSG1, MSG2, MSG3, TMP0, TMP1;
	const uint32x4_t K0 = vdupq_n_u32(0x5A827999);
	const uint32x4_t K1 = vdupq_n_u32(0x6ED9EBA1);
	const uint32x4_t K2 = vdupq_n_u32(0x8F1BBCDC);
	const uint32x4_t K3 = vdupq_n_u32(0xCA62C1D6);

#define STORE_W(k, MSG) do { \
	if (collect) \
		vst1q_u32(W + (k), MSG); \
} while (0)

	MSG0 = vreinterpretq_u32_u8(vrev32q_u8(vld1q_u8(data + 0)));
	MSG1 = vreinterpretq_u32_u8(vrev32q_u8(vld1q_u8(data + 16)));
	MSG2 = vreinterpretq_u32_u8(vrev32q_u8(vld1q_u8(data + 32)));
	MSG3 = vreinterpretq_u32_u8(vrev32q_u8(vld1q_u8(data + 48)));
	STORE_W(0, MSG0);
	STORE_W(4, MSG1);
	STORE_W(8, MSG2);
	STORE_W(12, MSG3);

	TMP0 = vaddq_u32(MSG0, K0);
	TMP1 = vaddq_u32(MSG1, K0);

	/* Rounds 0-3 */
	E1 = vsha1h_u32(vgetq_lane_u32(ABCD, 0));
	ABCD = vsha1cq_u32(ABCD, E0, TMP0);
	TMP0 = vaddq_u32(MSG2, K0);
	MSG0 = vsha1su0q_u32(MSG0, MSG1, MSG2);

	/* Rounds 4-7 */
	E0 = vsha1h_u32(vgetq_lane_u32(ABCD, 0));
	ABCD = vsha1cq_u32(ABCD, E1, TMP1);
	TMP1 = vaddq_u32(MSG3, K0);
	MSG0 = vsha1su1q_u32(MSG0, MSG3);
	STORE_W(16, MSG0);
	MSG1 = vsha1su0q_u32(MSG1, MSG2, MSG3);

	/* Rounds 8-11 */
	E1 = vsha1h_u32(vgetq_lane_u32(ABCD, 0));
	ABCD = vsha1cq_u32(ABCD, E0, TMP0);
	TMP0 = vaddq_u32(MSG0, K0);
	MSG1 = vsha1su1q_u32(MSG1, MSG0);
	STORE_W(20, MSG1);
	MSG2 = vsha1su0q_u32(MSG2, MSG3, MSG0);

	/* Rounds 12-15 */
	E0 = vsha1h_u32(vgetq_lane_u32(ABCD, 0));
	ABCD = vsha1cq_u32(ABCD, E1, TMP1);
	TMP1 = vaddq_u32(MSG1, K1);
	MSG2 = vsha1su1q_u32(MSG2, MSG1);
	STORE_W(24, MSG2);
	MSG3 = vsha1su0q_u32(MSG3, MSG0, MSG1);

	/* Rounds 16-19 */
	E1 = vsha1h_u32(vgetq_lane_u32(ABCD, 0));
	ABCD = vsha1cq_u32(ABCD, E0, TMP0);
	TMP0 = vaddq_u32(MSG2, K1);
	MSG3 = vsha1su1q_u32(MSG3, MSG2);
	STORE_W(28, MSG3);
	MSG0 = vsha1su0q_u32(MSG0, MSG1, MSG2);

	/* Rounds 20-23 */
	E0 = vsha1h_u32(vgetq_lane_u32(ABCD, 0));
	ABCD = vsha1pq_u32(ABCD, E1, TMP1);
	TMP1 = vaddq_u32(MSG3, K1);
	MSG0 = vsha1su1q_u32(MSG0, MSG3);
	STORE_W(32, MSG0);
	MSG1 = vsha1su0q_u32(MSG1, MSG2, MSG3);

	/* Rounds 24-27 */
	E1 = vsha1h_u32(vgetq_lane_u32(ABCD, 0));
	ABCD = vsha1pq_u32(ABCD, E0, TMP0);
	TMP0 = vaddq_u32(MSG0, K1);
	MSG1 = vsha1su1q_u32(MSG1, MSG0);
	STORE_W(36, MSG1);
	MSG2 = vsha1su0q_u32(MSG2, MSG3, MSG0);

	/* Rounds 28-31 */
	E0 = vsha1h_u32(vgetq_lane_u32(ABCD, 0));
	ABCD = vsha1pq_u32(ABCD, E1, TMP1);
	TMP1 = vaddq_u32(MSG1, K1);
	MSG2 = vsha1su1q_u32(MSG2, MSG1);
	STORE_W(40, MSG2);
	MSG3 = vsha1su0q_u32(MSG3, MSG0, MSG1);

	/* Rounds 32-35 */
	E1 = vsha1h_u32(vgetq_lane_u32(ABCD, 0));
	ABCD = vsha1pq_u32(ABCD, E0, TMP0);
	TMP0 = vaddq_u32(MSG2, K2);
	MSG3 = vsha1su1q_u32(MSG3, MSG2);
	STORE_W(44, MSG3);
	MSG0 = vsha1su0q_u32(MSG0, MSG1, MSG2);

	/* Rounds 36-39 */
	E0 = vsha1h_u32(vgetq_lane_u32(ABCD, 0));
	ABCD = vsha1pq_u32(ABCD, E1, TMP1);
	TMP1 = vaddq_u32(MSG3, K2);
	MSG0 = vsha1su1q_u32(MSG0, MSG3);
	STORE_W(48, MSG0);
	MSG1 = vsha1su0q_u32(MSG1, MSG2, MSG3);

	/* Rounds 40-43 */
	E1 = vsha1h_u32(vgetq_lane_u32(ABCD, 0));
	ABCD = vsha1mq_u32(ABCD, E0, TMP0);
	TMP0 = vaddq_u32(MSG0, K2);
	MSG1 = vsha1su1q_u32(MSG1, MSG0);
	STORE_W(52, MSG1);
	MSG2 = vsha1su0q_u32(MSG2, MSG3, MSG0);

	/* Rounds 44-47 */
	E0 = vsha1h_u32(vgetq_lane_u32(ABCD, 0));
	ABCD = vsha1mq_u32(ABCD, E1, TMP1);
	TMP1 = vaddq_u32(MSG1, K2);
	MSG2 = vsha1su1q_u32(MSG2, MSG1);
	STORE_W(56, MSG2);
	MSG3 = vsha1su0q_u32(MSG3, MSG0, MSG1);

	/* Rounds 48-51 */
	E1 = vsha1h_u32(vgetq_lane_u32(ABCD, 0));
	ABCD = vsha1mq_u32(ABCD, E0, TMP0);
	TMP0 = vaddq_u32(MSG2, K2);
	MSG3 = vsha1su1q_u32(MSG3, MSG2);
	STORE_W(60, MSG3);
	MSG0 = vsha1su0q_u32(MSG0, MSG1, MSG2);

	/* Rounds 52-55; the vsha1h result is the e entering round 56 */
	E0 = vsha1h_u32(vgetq_lane_u32(ABCD, 0));
	if (collect)
		*pE56 = E0;
	ABCD = vsha1mq_u32(ABCD, E1, TMP1);
	TMP1 = vaddq_u32(MSG3, K3);
	MSG0 = vsha1su1q_u32(MSG0, MSG3);
	STORE_W(64, MSG0);
	MSG1 = vsha1su0q_u32(MSG1, MSG2, MSG3);

	/* Rounds 56-59; with collect, only the message schedule remains */
	if (!collect) {
		E1 = vsha1h_u32(vgetq_lane_u32(ABCD, 0));
		ABCD = vsha1mq_u32(ABCD, E0, TMP0);
		TMP0 = vaddq_u32(MSG0, K3);
	}
	MSG1 = vsha1su1q_u32(MSG1, MSG0);
	STORE_W(68, MSG1);
	MSG2 = vsha1su0q_u32(MSG2, MSG3, MSG0);

	/* Rounds 60-63 */
	if (!collect) {
		E0 = vsha1h_u32(vgetq_lane_u32(ABCD, 0));
		ABCD = vsha1pq_u32(ABCD, E1, TMP1);
		TMP1 = vaddq_u32(MSG1, K3);
	}
	MSG2 = vsha1su1q_u32(MSG2, MSG1);
	STORE_W(72, MSG2);
	MSG3 = vsha1su0q_u32(MSG3, MSG0, MSG1);

	/* Rounds 64-67 */
	if (!collect) {
		E1 = vsha1h_u32(vgetq_lane_u32(ABCD, 0));
		ABCD = vsha1pq_u32(ABCD, E0, TMP0);
		TMP0 = vaddq_u32(MSG2, K3);
	}
	MSG3 = vsha1su1q_u32(MSG3, MSG2);
	STORE_W(76, MSG3);

	if (collect) {
		*pABCD = ABCD;	/* working state entering round 56 */
		return;
	}

	/* Rounds 68-71 */
	E0 = vsha1h_u32(vgetq_lane_u32(ABCD, 0));
	ABCD = vsha1pq_u32(ABCD, E1, TMP1);
	TMP1 = vaddq_u32(MSG3, K3);

	/* Rounds 72-75 */
	E1 = vsha1h_u32(vgetq_lane_u32(ABCD, 0));
	ABCD = vsha1pq_u32(ABCD, E0, TMP0);

	/* Rounds 76-79 */
	E0 = vsha1h_u32(vgetq_lane_u32(ABCD, 0));
	ABCD = vsha1pq_u32(ABCD, E1, TMP1);

	/* Combine state */
	*pABCD = vaddq_u32(ABCD, ABCD_SAVE);
	*pE = E0 + E0_SAVE;

#undef STORE_W
}

SHA1DC_TGT_SHA1 SHA1DC_INLINE
void sha1neon_block1(uint32x4_t *pABCD, uint32_t *pE, const unsigned char *data)
{
	sha1neon_block1_impl(pABCD, pE, data, 0, NULL, NULL);
}

SHA1DC_INLINE
void sha1neon_load_state(const uint32_t ihv[5], uint32x4_t *pABCD, uint32_t *pE)
{
	*pABCD = vld1q_u32(ihv);
	*pE = ihv[4];
}

SHA1DC_INLINE
void sha1neon_store_state(uint32x4_t ABCD, uint32_t E, uint32_t ihv[5])
{
	vst1q_u32(ihv, ABCD);
	ihv[4] = E;
}

/* Compress a whole number of blocks. */
SHA1DC_TGT_SHA1
void sha1dc_fast_compress(uint32_t ihv[5], const unsigned char *data, size_t len)
{
	uint32x4_t ABCD;
	uint32_t E;

	sha1neon_load_state(ihv, &ABCD, &E);
	while (len >= 64) {
		sha1neon_block1(&ABCD, &E, data);
		data += 64;
		len -= 64;
	}
	sha1neon_store_state(ABCD, E, ihv);
}

/*
 * Compress nblocks (<= 16), storing the chaining value that enters
 * block i into ckpt[i] so flagged blocks can be verified out of band.
 */
SHA1DC_TGT_SHA1
void sha1dc_fast_compress_ckpt(uint32_t ihv[5], const unsigned char *p,
			       unsigned nblocks, uint32_t ckpt[][5])
{
	uint32x4_t ABCD;
	uint32_t E;
	unsigned i;

	sha1neon_load_state(ihv, &ABCD, &E);
	for (i = 0; i < nblocks; i++) {
		sha1neon_store_state(ABCD, E, ckpt[i]);
		sha1neon_block1(&ABCD, &E, p + i * 64);
	}
	sha1neon_store_state(ABCD, E, ihv);
}

#define rol32(x, n) (((x) << (n)) | ((x) >> (32 - (n))))

/*
 * Prepare the data a flagged block's recompression checks need: the
 * decoded and expanded message W[80] and the sha1_compression_states()
 * snapshots at steps 58 and 65. sha1neon_block1_impl in collect mode
 * runs the full message schedule with sha1su0/sha1su1 (storing each
 * group to W) and the rounds up to 55 with the sha1 round instructions;
 * only rounds 56-64 run as scalar code here. The snapshots use the
 * rotating-variable order of SHA1_STORE_STATE in sha1.c: at step 58 the
 * stored words are {D,E,A,B,C} of the logical state, at step 65 they
 * are {A,B,C,D,E}.
 */
SHA1DC_TGT_SHA1
void sha1dc_fast_states(const unsigned char *block, uint32_t W[80],
			const uint32_t ihvin[5],
			uint32_t state58[5], uint32_t state65[5])
{
	uint32x4_t ABCD;
	uint32_t E, a, b, c, d, e;

	sha1neon_load_state(ihvin, &ABCD, &E);
	sha1neon_block1_impl(&ABCD, &E, block, 1, W, &e);

	a = vgetq_lane_u32(ABCD, 0);
	b = vgetq_lane_u32(ABCD, 1);
	c = vgetq_lane_u32(ABCD, 2);
	d = vgetq_lane_u32(ABCD, 3);

#define FAST_ROUND(f, k, t) do { \
	uint32_t tmp = rol32(a, 5) + (f) + e + (k) + W[t]; \
	e = d; d = c; c = rol32(b, 30); b = a; a = tmp; \
} while (0)
#define FAST_ROUND3(t) FAST_ROUND(((b & c) + (d & (b ^ c))), 0x8F1BBCDC, t)
#define FAST_ROUND4(t) FAST_ROUND((b ^ c ^ d), 0xCA62C1D6, t)

	FAST_ROUND3(56);
	FAST_ROUND3(57);
	state58[0] = d; state58[1] = e; state58[2] = a;
	state58[3] = b; state58[4] = c;
	FAST_ROUND3(58);
	FAST_ROUND3(59);
	FAST_ROUND4(60);
	FAST_ROUND4(61);
	FAST_ROUND4(62);
	FAST_ROUND4(63);
	FAST_ROUND4(64);
	state65[0] = a; state65[1] = b; state65[2] = c;
	state65[3] = d; state65[4] = e;

#undef FAST_ROUND
#undef FAST_ROUND3
#undef FAST_ROUND4
}

/*
 * Load + byteswap + transpose + message-expand 4 consecutive blocks, one
 * per NEON lane. The whole 16-word expansion window lives in registers
 * (the W[t-3] recurrence through a memory array would serialize on
 * store-to-load forwarding); only W[35..64] are stored, the only indices
 * ubc_check reads.
 */
SHA1DC_INLINE
void scan4_prepare(const unsigned char *p, u32v4 W[65])
{
	uint32x4_t w0, w1, w2, w3, w4, w5, w6, w7;
	uint32x4_t w8, w9, w10, w11, w12, w13, w14, w15;

	/* one 4x4 dword transpose per 16-byte column of the 4 blocks */
#define SCAN4_LOAD(g, c0, c1, c2, c3) do { \
	uint32x4_t r0 = vreinterpretq_u32_u8(vrev32q_u8(vld1q_u8(p + 0 * 64 + (g) * 16))); \
	uint32x4_t r1 = vreinterpretq_u32_u8(vrev32q_u8(vld1q_u8(p + 1 * 64 + (g) * 16))); \
	uint32x4_t r2 = vreinterpretq_u32_u8(vrev32q_u8(vld1q_u8(p + 2 * 64 + (g) * 16))); \
	uint32x4_t r3 = vreinterpretq_u32_u8(vrev32q_u8(vld1q_u8(p + 3 * 64 + (g) * 16))); \
	uint32x4_t t0 = vtrn1q_u32(r0, r1); \
	uint32x4_t t1 = vtrn2q_u32(r0, r1); \
	uint32x4_t t2 = vtrn1q_u32(r2, r3); \
	uint32x4_t t3 = vtrn2q_u32(r2, r3); \
	c0 = vreinterpretq_u32_u64(vtrn1q_u64(vreinterpretq_u64_u32(t0), vreinterpretq_u64_u32(t2))); \
	c1 = vreinterpretq_u32_u64(vtrn1q_u64(vreinterpretq_u64_u32(t1), vreinterpretq_u64_u32(t3))); \
	c2 = vreinterpretq_u32_u64(vtrn2q_u64(vreinterpretq_u64_u32(t0), vreinterpretq_u64_u32(t2))); \
	c3 = vreinterpretq_u32_u64(vtrn2q_u64(vreinterpretq_u64_u32(t1), vreinterpretq_u64_u32(t3))); \
} while (0)

	SCAN4_LOAD(0, w0, w1, w2, w3);
	SCAN4_LOAD(1, w4, w5, w6, w7);
	SCAN4_LOAD(2, w8, w9, w10, w11);
	SCAN4_LOAD(3, w12, w13, w14, w15);
#undef SCAN4_LOAD

	/* w[t & 15] is W[t]; each step overwrites the slot it consumes.
	 * d is W[t-3], the serial recurrence input, so it XORs last. */
#define SCAN4_EXP(t, a, b, c, d) do { \
	uint32x4_t x = veorq_u32(veorq_u32(veorq_u32(a, b), c), d); \
	a = vsriq_n_u32(vshlq_n_u32(x, 1), x, 31); \
	if ((t) >= 35) \
		W[t] = (u32v4)a; \
} while (0)
#define SCAN4_EXP16(base) do { \
	SCAN4_EXP((base) + 0, w0, w2, w8, w13); \
	SCAN4_EXP((base) + 1, w1, w3, w9, w14); \
	SCAN4_EXP((base) + 2, w2, w4, w10, w15); \
	SCAN4_EXP((base) + 3, w3, w5, w11, w0); \
	SCAN4_EXP((base) + 4, w4, w6, w12, w1); \
	SCAN4_EXP((base) + 5, w5, w7, w13, w2); \
	SCAN4_EXP((base) + 6, w6, w8, w14, w3); \
	SCAN4_EXP((base) + 7, w7, w9, w15, w4); \
	SCAN4_EXP((base) + 8, w8, w10, w0, w5); \
	SCAN4_EXP((base) + 9, w9, w11, w1, w6); \
	SCAN4_EXP((base) + 10, w10, w12, w2, w7); \
	SCAN4_EXP((base) + 11, w11, w13, w3, w8); \
	SCAN4_EXP((base) + 12, w12, w14, w4, w9); \
	SCAN4_EXP((base) + 13, w13, w15, w5, w10); \
	SCAN4_EXP((base) + 14, w14, w0, w6, w11); \
	SCAN4_EXP((base) + 15, w15, w1, w7, w12); \
} while (0)

	SCAN4_EXP16(16);
	SCAN4_EXP16(32);
	SCAN4_EXP16(48);
	SCAN4_EXP(64, w0, w2, w8, w13);
#undef SCAN4_EXP16
#undef SCAN4_EXP
}

/* Store per-block dvmask values and return the flagged-block bitmask. */
SHA1DC_INLINE
uint32_t scan4_result(u32v4 dv, uint32_t dvout[4])
{
	vst1q_u32(dvout, (uint32x4_t)dv);
	return (uint32_t)((dvout[0] != 0) | (dvout[1] != 0) << 1 |
			  (dvout[2] != 0) << 2 | (dvout[3] != 0) << 3);
}

/* ubc_check thirds, split so the fused kernel can interleave them. */
SHA1DC_INLINE
void scan4_ubc_a(const u32v4 W[65], u32v4 m[4])
{
	m[0] = m[1] = m[2] = m[3] = ~(u32v4){0};
	ubc_check_v4_part0(W, m);
	ubc_check_v4_part1(W, m);
	ubc_check_v4_part2(W, m);
}

SHA1DC_INLINE
void scan4_ubc_b(const u32v4 W[65], u32v4 m[4])
{
	ubc_check_v4_part3(W, m);
	ubc_check_v4_part4(W, m);
	ubc_check_v4_part5(W, m);
}

SHA1DC_INLINE
uint32_t scan4_ubc_c(const u32v4 W[65], u32v4 m[4], uint32_t dvout[4])
{
	ubc_check_v4_part6(W, m);
	ubc_check_v4_part7(W, m);
	return scan4_result(m[0] & m[1] & m[2] & m[3], dvout);
}

/* Scan 4 consecutive blocks; returns a bitmask of flagged blocks. */
SHA1DC_INLINE
uint32_t scan4(const unsigned char *p, uint32_t dvout[4])
{
	u32v4 W[65];
	u32v4 dv;

	scan4_prepare(p, W);
	ubc_check_v4_inline(W, &dv);
	return scan4_result(dv, dvout);
}

/* Scan 16 consecutive blocks as four 4-lane scans. */
uint32_t sha1dc_fast_scan16(const unsigned char *p, uint32_t dvout[16])
{
	return scan4(p, dvout)
	     | scan4(p + 256, dvout + 4) << 4
	     | scan4(p + 512, dvout + 8) << 8
	     | scan4(p + 768, dvout + 12) << 12;
}

/*
 * Fused kernel: compress the 16 blocks of 'cur' with the SHA-1
 * instructions while scanning the 16 blocks of 'next', interleaved at
 * source level. The sha1 round instruction chain is latency-bound and
 * leaves NEON issue slots idle; placing a slice of scan work between
 * consecutive block steps keeps both instruction streams inside the
 * out-of-order window. Each of the four 4-lane scans is cut into four
 * slices (load+transpose+expand, three ubc thirds) so the 16 slices fill
 * the 16 block-step gaps. The chaining value entering block i is stored
 * into ckpt[i].
 */
SHA1DC_TGT_SHA1
uint32_t sha1dc_fast_fused16(uint32_t ihv[5], const unsigned char *cur,
			     const unsigned char *next, uint32_t dvout[16],
			     uint32_t ckpt[16][5])
{
	uint32x4_t ABCD;
	uint32_t E;
	u32v4 W[65];
	u32v4 m[4];
	uint32_t flags;

	sha1neon_load_state(ihv, &ABCD, &E);

#define FUSED_BLOCK(i) do { \
	sha1neon_store_state(ABCD, E, ckpt[i]); \
	sha1neon_block1(&ABCD, &E, cur + (i) * 64); \
} while (0)

	FUSED_BLOCK(0);
	scan4_prepare(next, W);
	FUSED_BLOCK(1);
	scan4_ubc_a(W, m);
	FUSED_BLOCK(2);
	scan4_ubc_b(W, m);
	FUSED_BLOCK(3);
	flags = scan4_ubc_c(W, m, dvout);
	FUSED_BLOCK(4);
	scan4_prepare(next + 256, W);
	FUSED_BLOCK(5);
	scan4_ubc_a(W, m);
	FUSED_BLOCK(6);
	scan4_ubc_b(W, m);
	FUSED_BLOCK(7);
	flags |= scan4_ubc_c(W, m, dvout + 4) << 4;
	FUSED_BLOCK(8);
	scan4_prepare(next + 512, W);
	FUSED_BLOCK(9);
	scan4_ubc_a(W, m);
	FUSED_BLOCK(10);
	scan4_ubc_b(W, m);
	FUSED_BLOCK(11);
	flags |= scan4_ubc_c(W, m, dvout + 8) << 8;
	FUSED_BLOCK(12);
	scan4_prepare(next + 768, W);
	FUSED_BLOCK(13);
	scan4_ubc_a(W, m);
	FUSED_BLOCK(14);
	scan4_ubc_b(W, m);
	FUSED_BLOCK(15);
	flags |= scan4_ubc_c(W, m, dvout + 12) << 12;
#undef FUSED_BLOCK

	sha1neon_store_state(ABCD, E, ihv);
	return flags;
}

#if defined(__APPLE__)
#include <sys/sysctl.h>
static int sha1dc_detect_fast_level(void)
{
	int v = 0;
	size_t sz = sizeof(v);

	if (sysctlbyname("hw.optional.arm.FEAT_SHA1", &v, &sz, NULL, 0) != 0)
		return 0;
	return v ? 2 : 0;
}
#elif defined(__linux__)
#include <sys/auxv.h>
#ifndef HWCAP_SHA1
#define HWCAP_SHA1 (1UL << 5)
#endif
static int sha1dc_detect_fast_level(void)
{
	return (getauxval(AT_HWCAP) & HWCAP_SHA1) ? 2 : 0;
}
#else
static int sha1dc_detect_fast_level(void)
{
	return 0;
}
#endif

static void sha1dc_fast_init(void) __attribute__((constructor));
static void sha1dc_fast_init(void)
{
	const char *s;

	if (getenv("SHA1DC_NO_FAST"))
		return;
	sha1dc_fast_level = sha1dc_detect_fast_level();
	/* SHA1DC_FAST_LEVEL can lower (never raise) the level, for A/B tests. */
	s = getenv("SHA1DC_FAST_LEVEL");
	if (s && s[0] >= '0' && s[0] <= '2' && !s[1] &&
	    s[0] - '0' < sha1dc_fast_level)
		sha1dc_fast_level = s[0] - '0';
}

#endif /* SHA1DC_FAST_ARM64 */
#endif /* __aarch64__ && __GNUC__ && !SHA1DC_NO_FAST_SHANI && little-endian */
