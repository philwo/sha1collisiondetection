/*
 * Hardware-accelerated fast path for sha1dc on x86-64 (used by sha1.c).
 *
 * ubc_check() depends only on the expanded message words of a block. A
 * block whose dvmask is zero needs no recompression checks, and its
 * compression output equals plain SHA-1, so it can be computed with the
 * SHA-NI instructions. The helpers here scan blocks for that condition
 * with AVX2 or AVX-512 (one block per vector lane) and compress with
 * SHA-NI. Everything is compiled with GCC target attributes and selected
 * at run time via sha1dc_fast_level (0 = off, 1 = SHA-NI+AVX2,
 * 2 = SHA-NI+AVX-512).
 *
 * The SHA-NI block step is based on code written and placed in the public
 * domain by Jeffrey Walton (sha1-x86.c, github.com/noloader/SHA-Intrinsics),
 * which is in turn based on code from Intel and by Sean Gulley.
 */
#ifndef SHA1DC_NO_STANDARD_INCLUDES
#include <stdint.h>
#include <stdlib.h>
#include <stddef.h>
#endif

#ifdef SHA1DC_CUSTOM_INCLUDE_SHA1DC_FAST_X86_C
#include SHA1DC_CUSTOM_INCLUDE_SHA1DC_FAST_X86_C
#endif

#if defined(__x86_64__) && defined(__GNUC__) && !defined(SHA1DC_NO_FAST_SHANI)

#include <immintrin.h>
#include <cpuid.h>

#include "ubc_check_simd.h"

int sha1dc_fast_level;

#define SHA1DC_TGT_SHANI __attribute__((target("sha,sse4.1")))
#define SHA1DC_TGT_AVX2 __attribute__((target("sha,sse4.1,avx2")))
#define SHA1DC_TGT_AVX512 __attribute__((target("sha,sse4.1,avx512f,avx512bw")))
#define SHA1DC_INLINE static inline __attribute__((always_inline))

/*
 * One SHA-1 block step with SHA-NI. ABCD holds a,b,c,d (element order
 * reversed with shuffle 0x1B); E0 holds e in the top dword.
 *
 * With collect == 0 this compresses one block (W and a52 are unused).
 * With collect == 1 it instead prepares the recompression-check inputs
 * for a flagged block: the message schedule still runs in full and each
 * completed 4-word group is stored to W[80] in decoded order, but the
 * rounds stop after round 55; *pABCD returns the working state entering
 * round 56, *a52 the 'a' value entering round 52, and *pE0 is not
 * updated. collect must be a literal so the branches constant-fold.
 */
SHA1DC_TGT_SHANI SHA1DC_INLINE
void sha1ni_block1_impl(__m128i *pABCD, __m128i *pE0, const unsigned char *data,
			const int collect, uint32_t *W, uint32_t *a52)
{
	__m128i ABCD = *pABCD, E0 = *pE0, E1;
	__m128i MSG0, MSG1, MSG2, MSG3, ABCD_SAVE, E0_SAVE;
	const __m128i MASK = _mm_set_epi64x(0x0001020304050607ULL, 0x08090a0b0c0d0e0fULL);

#define STORE_W(k, MSG) do { \
	if (collect) \
		_mm_storeu_si128((__m128i *)(W + (k)), \
				 _mm_shuffle_epi32(MSG, 0x1B)); \
} while (0)

	ABCD_SAVE = ABCD;
	E0_SAVE = E0;

	/* Rounds 0-3 */
	MSG0 = _mm_loadu_si128((const __m128i*)(data + 0));
	MSG0 = _mm_shuffle_epi8(MSG0, MASK);
	STORE_W(0, MSG0);
	E0 = _mm_add_epi32(E0, MSG0);
	E1 = ABCD;
	ABCD = _mm_sha1rnds4_epu32(ABCD, E0, 0);

	/* Rounds 4-7 */
	MSG1 = _mm_loadu_si128((const __m128i*)(data + 16));
	MSG1 = _mm_shuffle_epi8(MSG1, MASK);
	STORE_W(4, MSG1);
	E1 = _mm_sha1nexte_epu32(E1, MSG1);
	E0 = ABCD;
	ABCD = _mm_sha1rnds4_epu32(ABCD, E1, 0);
	MSG0 = _mm_sha1msg1_epu32(MSG0, MSG1);

	/* Rounds 8-11 */
	MSG2 = _mm_loadu_si128((const __m128i*)(data + 32));
	MSG2 = _mm_shuffle_epi8(MSG2, MASK);
	STORE_W(8, MSG2);
	E0 = _mm_sha1nexte_epu32(E0, MSG2);
	E1 = ABCD;
	ABCD = _mm_sha1rnds4_epu32(ABCD, E0, 0);
	MSG1 = _mm_sha1msg1_epu32(MSG1, MSG2);
	MSG0 = _mm_xor_si128(MSG0, MSG2);

	/* Rounds 12-15 */
	MSG3 = _mm_loadu_si128((const __m128i*)(data + 48));
	MSG3 = _mm_shuffle_epi8(MSG3, MASK);
	STORE_W(12, MSG3);
	E1 = _mm_sha1nexte_epu32(E1, MSG3);
	E0 = ABCD;
	MSG0 = _mm_sha1msg2_epu32(MSG0, MSG3);
	STORE_W(16, MSG0);
	ABCD = _mm_sha1rnds4_epu32(ABCD, E1, 0);
	MSG2 = _mm_sha1msg1_epu32(MSG2, MSG3);
	MSG1 = _mm_xor_si128(MSG1, MSG3);

	/* Rounds 16-19 */
	E0 = _mm_sha1nexte_epu32(E0, MSG0);
	E1 = ABCD;
	MSG1 = _mm_sha1msg2_epu32(MSG1, MSG0);
	STORE_W(20, MSG1);
	ABCD = _mm_sha1rnds4_epu32(ABCD, E0, 0);
	MSG3 = _mm_sha1msg1_epu32(MSG3, MSG0);
	MSG2 = _mm_xor_si128(MSG2, MSG0);

	/* Rounds 20-23 */
	E1 = _mm_sha1nexte_epu32(E1, MSG1);
	E0 = ABCD;
	MSG2 = _mm_sha1msg2_epu32(MSG2, MSG1);
	STORE_W(24, MSG2);
	ABCD = _mm_sha1rnds4_epu32(ABCD, E1, 1);
	MSG0 = _mm_sha1msg1_epu32(MSG0, MSG1);
	MSG3 = _mm_xor_si128(MSG3, MSG1);

	/* Rounds 24-27 */
	E0 = _mm_sha1nexte_epu32(E0, MSG2);
	E1 = ABCD;
	MSG3 = _mm_sha1msg2_epu32(MSG3, MSG2);
	STORE_W(28, MSG3);
	ABCD = _mm_sha1rnds4_epu32(ABCD, E0, 1);
	MSG1 = _mm_sha1msg1_epu32(MSG1, MSG2);
	MSG0 = _mm_xor_si128(MSG0, MSG2);

	/* Rounds 28-31 */
	E1 = _mm_sha1nexte_epu32(E1, MSG3);
	E0 = ABCD;
	MSG0 = _mm_sha1msg2_epu32(MSG0, MSG3);
	STORE_W(32, MSG0);
	ABCD = _mm_sha1rnds4_epu32(ABCD, E1, 1);
	MSG2 = _mm_sha1msg1_epu32(MSG2, MSG3);
	MSG1 = _mm_xor_si128(MSG1, MSG3);

	/* Rounds 32-35 */
	E0 = _mm_sha1nexte_epu32(E0, MSG0);
	E1 = ABCD;
	MSG1 = _mm_sha1msg2_epu32(MSG1, MSG0);
	STORE_W(36, MSG1);
	ABCD = _mm_sha1rnds4_epu32(ABCD, E0, 1);
	MSG3 = _mm_sha1msg1_epu32(MSG3, MSG0);
	MSG2 = _mm_xor_si128(MSG2, MSG0);

	/* Rounds 36-39 */
	E1 = _mm_sha1nexte_epu32(E1, MSG1);
	E0 = ABCD;
	MSG2 = _mm_sha1msg2_epu32(MSG2, MSG1);
	STORE_W(40, MSG2);
	ABCD = _mm_sha1rnds4_epu32(ABCD, E1, 1);
	MSG0 = _mm_sha1msg1_epu32(MSG0, MSG1);
	MSG3 = _mm_xor_si128(MSG3, MSG1);

	/* Rounds 40-43 */
	E0 = _mm_sha1nexte_epu32(E0, MSG2);
	E1 = ABCD;
	MSG3 = _mm_sha1msg2_epu32(MSG3, MSG2);
	STORE_W(44, MSG3);
	ABCD = _mm_sha1rnds4_epu32(ABCD, E0, 2);
	MSG1 = _mm_sha1msg1_epu32(MSG1, MSG2);
	MSG0 = _mm_xor_si128(MSG0, MSG2);

	/* Rounds 44-47 */
	E1 = _mm_sha1nexte_epu32(E1, MSG3);
	E0 = ABCD;
	MSG0 = _mm_sha1msg2_epu32(MSG0, MSG3);
	STORE_W(48, MSG0);
	ABCD = _mm_sha1rnds4_epu32(ABCD, E1, 2);
	MSG2 = _mm_sha1msg1_epu32(MSG2, MSG3);
	MSG1 = _mm_xor_si128(MSG1, MSG3);

	/* Rounds 48-51 */
	E0 = _mm_sha1nexte_epu32(E0, MSG0);
	E1 = ABCD;
	MSG1 = _mm_sha1msg2_epu32(MSG1, MSG0);
	STORE_W(52, MSG1);
	ABCD = _mm_sha1rnds4_epu32(ABCD, E0, 2);
	MSG3 = _mm_sha1msg1_epu32(MSG3, MSG0);
	MSG2 = _mm_xor_si128(MSG2, MSG0);

	/* Rounds 52-55 */
	E1 = _mm_sha1nexte_epu32(E1, MSG1);
	E0 = ABCD;
	if (collect)	/* 'a' entering round 52 */
		*a52 = (uint32_t)_mm_extract_epi32(ABCD, 3);
	MSG2 = _mm_sha1msg2_epu32(MSG2, MSG1);
	STORE_W(56, MSG2);
	ABCD = _mm_sha1rnds4_epu32(ABCD, E1, 2);
	MSG0 = _mm_sha1msg1_epu32(MSG0, MSG1);
	MSG3 = _mm_xor_si128(MSG3, MSG1);

	/* Rounds 56-59; with collect, only the message schedule remains */
	if (!collect) {
		E0 = _mm_sha1nexte_epu32(E0, MSG2);
		E1 = ABCD;
		ABCD = _mm_sha1rnds4_epu32(ABCD, E0, 2);
	}
	MSG3 = _mm_sha1msg2_epu32(MSG3, MSG2);
	STORE_W(60, MSG3);
	MSG1 = _mm_sha1msg1_epu32(MSG1, MSG2);
	MSG0 = _mm_xor_si128(MSG0, MSG2);

	/* Rounds 60-63 */
	if (!collect) {
		E1 = _mm_sha1nexte_epu32(E1, MSG3);
		E0 = ABCD;
		ABCD = _mm_sha1rnds4_epu32(ABCD, E1, 3);
	}
	MSG0 = _mm_sha1msg2_epu32(MSG0, MSG3);
	STORE_W(64, MSG0);
	MSG2 = _mm_sha1msg1_epu32(MSG2, MSG3);
	MSG1 = _mm_xor_si128(MSG1, MSG3);

	/* Rounds 64-67 */
	if (!collect) {
		E0 = _mm_sha1nexte_epu32(E0, MSG0);
		E1 = ABCD;
		ABCD = _mm_sha1rnds4_epu32(ABCD, E0, 3);
	}
	MSG1 = _mm_sha1msg2_epu32(MSG1, MSG0);
	STORE_W(68, MSG1);
	MSG3 = _mm_sha1msg1_epu32(MSG3, MSG0);
	MSG2 = _mm_xor_si128(MSG2, MSG0);

	/* Rounds 68-71 */
	if (!collect) {
		E1 = _mm_sha1nexte_epu32(E1, MSG1);
		E0 = ABCD;
		ABCD = _mm_sha1rnds4_epu32(ABCD, E1, 3);
	}
	MSG2 = _mm_sha1msg2_epu32(MSG2, MSG1);
	STORE_W(72, MSG2);
	MSG3 = _mm_xor_si128(MSG3, MSG1);

	/* Rounds 72-75 */
	if (!collect) {
		E0 = _mm_sha1nexte_epu32(E0, MSG2);
		E1 = ABCD;
		ABCD = _mm_sha1rnds4_epu32(ABCD, E0, 3);
	}
	MSG3 = _mm_sha1msg2_epu32(MSG3, MSG2);
	STORE_W(76, MSG3);

	if (collect) {
		*pABCD = ABCD;	/* working state entering round 56 */
		return;
	}

	/* Rounds 76-79 */
	E1 = _mm_sha1nexte_epu32(E1, MSG3);
	E0 = ABCD;
	ABCD = _mm_sha1rnds4_epu32(ABCD, E1, 3);

	/* Combine state */
	E0 = _mm_sha1nexte_epu32(E0, E0_SAVE);
	ABCD = _mm_add_epi32(ABCD, ABCD_SAVE);

	*pABCD = ABCD;
	*pE0 = E0;

#undef STORE_W
}

SHA1DC_TGT_SHANI SHA1DC_INLINE
void sha1ni_block1(__m128i *pABCD, __m128i *pE0, const unsigned char *data)
{
	sha1ni_block1_impl(pABCD, pE0, data, 0, NULL, NULL);
}

SHA1DC_TGT_SHANI SHA1DC_INLINE
void sha1ni_load_state(const uint32_t ihv[5], __m128i *pABCD, __m128i *pE0)
{
	*pABCD = _mm_shuffle_epi32(_mm_loadu_si128((const __m128i *)ihv), 0x1B);
	*pE0 = _mm_set_epi32((int)ihv[4], 0, 0, 0);
}

SHA1DC_TGT_SHANI SHA1DC_INLINE
void sha1ni_store_state(__m128i ABCD, __m128i E0, uint32_t ihv[5])
{
	_mm_storeu_si128((__m128i *)ihv, _mm_shuffle_epi32(ABCD, 0x1B));
	ihv[4] = (uint32_t)_mm_extract_epi32(E0, 3);
}

/* Compress a whole number of blocks with SHA-NI. */
SHA1DC_TGT_SHANI
void sha1dc_fast_compress(uint32_t ihv[5], const unsigned char *data, size_t len)
{
	__m128i ABCD, E0;

	sha1ni_load_state(ihv, &ABCD, &E0);
	while (len >= 64) {
		sha1ni_block1(&ABCD, &E0, data);
		data += 64;
		len -= 64;
	}
	sha1ni_store_state(ABCD, E0, ihv);
}

/*
 * Compress nblocks (<= 16) with SHA-NI, storing the chaining value that
 * enters block i into ckpt[i] so flagged blocks can be verified out of
 * band.
 */
SHA1DC_TGT_SHANI
void sha1dc_fast_compress_ckpt(uint32_t ihv[5], const unsigned char *p,
			       unsigned nblocks, uint32_t ckpt[][5])
{
	__m128i ABCD, E0;
	unsigned i;

	sha1ni_load_state(ihv, &ABCD, &E0);
	for (i = 0; i < nblocks; i++) {
		sha1ni_store_state(ABCD, E0, ckpt[i]);
		sha1ni_block1(&ABCD, &E0, p + i * 64);
	}
	sha1ni_store_state(ABCD, E0, ihv);
}

#define rol32(x, n) (((x) << (n)) | ((x) >> (32 - (n))))

/*
 * Prepare the data a flagged block's recompression checks need: the
 * decoded and expanded message W[80] and the sha1_compression_states()
 * snapshots at steps 58 and 65. sha1ni_block1_impl in collect mode runs
 * the full message schedule with sha1msg1/sha1msg2 (storing each group
 * to W) and the rounds up to 55 with sha1rnds4; only rounds 56-64 run
 * as scalar code here. The snapshots use the rotating-variable order of
 * SHA1_STORE_STATE in sha1.c: at step 58 the stored words are
 * {D,E,A,B,C} of the logical state, at step 65 they are {A,B,C,D,E}.
 */
SHA1DC_TGT_SHANI
void sha1dc_fast_states(const unsigned char *block, uint32_t W[80],
			const uint32_t ihvin[5],
			uint32_t state58[5], uint32_t state65[5])
{
	__m128i ABCD, E0;
	uint32_t a, b, c, d, e, a52;

	sha1ni_load_state(ihvin, &ABCD, &E0);
	sha1ni_block1_impl(&ABCD, &E0, block, 1, W, &a52);

	a = (uint32_t)_mm_extract_epi32(ABCD, 3);
	b = (uint32_t)_mm_extract_epi32(ABCD, 2);
	c = (uint32_t)_mm_extract_epi32(ABCD, 1);
	d = (uint32_t)_mm_extract_epi32(ABCD, 0);
	e = rol32(a52, 30);	/* E_56 = rol(A_52, 30) */

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

/* Transpose the 8x8 dword matrix r0..r7 into out[0..7]. */
SHA1DC_TGT_AVX2 SHA1DC_INLINE
void sha1dc_transpose8(__m256i r0, __m256i r1, __m256i r2, __m256i r3,
		       __m256i r4, __m256i r5, __m256i r6, __m256i r7,
		       u32v8 *out)
{
	__m256i t0 = _mm256_unpacklo_epi32(r0, r1);
	__m256i t1 = _mm256_unpackhi_epi32(r0, r1);
	__m256i t2 = _mm256_unpacklo_epi32(r2, r3);
	__m256i t3 = _mm256_unpackhi_epi32(r2, r3);
	__m256i t4 = _mm256_unpacklo_epi32(r4, r5);
	__m256i t5 = _mm256_unpackhi_epi32(r4, r5);
	__m256i t6 = _mm256_unpacklo_epi32(r6, r7);
	__m256i t7 = _mm256_unpackhi_epi32(r6, r7);
	__m256i u0 = _mm256_unpacklo_epi64(t0, t2);
	__m256i u1 = _mm256_unpackhi_epi64(t0, t2);
	__m256i u2 = _mm256_unpacklo_epi64(t1, t3);
	__m256i u3 = _mm256_unpackhi_epi64(t1, t3);
	__m256i u4 = _mm256_unpacklo_epi64(t4, t6);
	__m256i u5 = _mm256_unpackhi_epi64(t4, t6);
	__m256i u6 = _mm256_unpacklo_epi64(t5, t7);
	__m256i u7 = _mm256_unpackhi_epi64(t5, t7);

	out[0] = (u32v8)_mm256_permute2x128_si256(u0, u4, 0x20);
	out[1] = (u32v8)_mm256_permute2x128_si256(u1, u5, 0x20);
	out[2] = (u32v8)_mm256_permute2x128_si256(u2, u6, 0x20);
	out[3] = (u32v8)_mm256_permute2x128_si256(u3, u7, 0x20);
	out[4] = (u32v8)_mm256_permute2x128_si256(u0, u4, 0x31);
	out[5] = (u32v8)_mm256_permute2x128_si256(u1, u5, 0x31);
	out[6] = (u32v8)_mm256_permute2x128_si256(u2, u6, 0x31);
	out[7] = (u32v8)_mm256_permute2x128_si256(u3, u7, 0x31);
}

/* Load + byteswap one 32-byte slice of 8 blocks and transpose it into
 * W[8*half .. 8*half+7]. */
SHA1DC_TGT_AVX2 SHA1DC_INLINE
void scan8_load_half(const unsigned char *p, int half, u32v8 W[65])
{
	const __m256i bswap = _mm256_broadcastsi128_si256(
		_mm_setr_epi8(3, 2, 1, 0, 7, 6, 5, 4, 11, 10, 9, 8, 15, 14, 13, 12));
	const unsigned char *q = p + half * 32;
	__m256i r0 = _mm256_shuffle_epi8(_mm256_loadu_si256((const __m256i *)(q + 0 * 64)), bswap);
	__m256i r1 = _mm256_shuffle_epi8(_mm256_loadu_si256((const __m256i *)(q + 1 * 64)), bswap);
	__m256i r2 = _mm256_shuffle_epi8(_mm256_loadu_si256((const __m256i *)(q + 2 * 64)), bswap);
	__m256i r3 = _mm256_shuffle_epi8(_mm256_loadu_si256((const __m256i *)(q + 3 * 64)), bswap);
	__m256i r4 = _mm256_shuffle_epi8(_mm256_loadu_si256((const __m256i *)(q + 4 * 64)), bswap);
	__m256i r5 = _mm256_shuffle_epi8(_mm256_loadu_si256((const __m256i *)(q + 5 * 64)), bswap);
	__m256i r6 = _mm256_shuffle_epi8(_mm256_loadu_si256((const __m256i *)(q + 6 * 64)), bswap);
	__m256i r7 = _mm256_shuffle_epi8(_mm256_loadu_si256((const __m256i *)(q + 7 * 64)), bswap);

	sha1dc_transpose8(r0, r1, r2, r3, r4, r5, r6, r7, &W[half * 8]);
}

/* Message expansion for t in [from, to); ubc needs W up to index 64 only. */
SHA1DC_TGT_AVX2 SHA1DC_INLINE
void scan8_expand(u32v8 W[65], int from, int to)
{
	int t;

	for (t = from; t < to; t++) {
		u32v8 x = W[t - 3] ^ W[t - 8] ^ W[t - 14] ^ W[t - 16];

		W[t] = (x << 1) | (x >> 31);
	}
}

/* Store per-block dvmask values and return the flagged-block bitmask. */
SHA1DC_TGT_AVX2 SHA1DC_INLINE
uint32_t scan8_result(u32v8 dv, uint32_t dvout[8])
{
	__m256i z;

	_mm256_storeu_si256((__m256i *)dvout, (__m256i)dv);
	z = _mm256_cmpeq_epi32((__m256i)dv, _mm256_setzero_si256());
	return ~(uint32_t)_mm256_movemask_ps((__m256)z) & 0xff;
}

/*
 * Scan 8 consecutive blocks, one per AVX2 lane; returns a bitmask of
 * flagged blocks. ubc_check reads W[35..64] only, so the expansion stops
 * at W[64].
 */
SHA1DC_TGT_AVX2
uint32_t sha1dc_fast_scan8(const unsigned char *p, uint32_t dvout[8])
{
	u32v8 W[65];
	u32v8 dv;

	scan8_load_half(p, 0, W);
	scan8_load_half(p, 1, W);
	scan8_expand(W, 16, 65);
	ubc_check_v8_inline(W, &dv);
	return scan8_result(dv, dvout);
}

/* Load blocks [from, to) as one zmm each and byteswap the dwords. */
SHA1DC_TGT_AVX512 SHA1DC_INLINE
void scan16_load(const unsigned char *p, __m512i r[16], int from, int to)
{
	const __m512i bswap = _mm512_broadcast_i32x4(
		_mm_setr_epi8(3, 2, 1, 0, 7, 6, 5, 4, 11, 10, 9, 8, 15, 14, 13, 12));
	int i;

	for (i = from; i < to; i++)
		r[i] = _mm512_shuffle_epi8(_mm512_loadu_si512(p + i * 64), bswap);
}

/* 16x16 dword transpose, stage 1+2: 4x4 transpose within 128-bit lanes. */
SHA1DC_TGT_AVX512 SHA1DC_INLINE
void scan16_transpose_ab(const __m512i r[16], __m512i b[16])
{
	__m512i a[16];
	int i;

	for (i = 0; i < 8; i++) {
		a[2 * i] = _mm512_unpacklo_epi32(r[2 * i], r[2 * i + 1]);
		a[2 * i + 1] = _mm512_unpackhi_epi32(r[2 * i], r[2 * i + 1]);
	}
	for (i = 0; i < 4; i++) {
		b[4 * i + 0] = _mm512_unpacklo_epi64(a[4 * i + 0], a[4 * i + 2]);
		b[4 * i + 1] = _mm512_unpackhi_epi64(a[4 * i + 0], a[4 * i + 2]);
		b[4 * i + 2] = _mm512_unpacklo_epi64(a[4 * i + 1], a[4 * i + 3]);
		b[4 * i + 3] = _mm512_unpackhi_epi64(a[4 * i + 1], a[4 * i + 3]);
	}
}

/* Stage 3+4: 4x4 transpose of 128-bit lanes across register groups. */
SHA1DC_TGT_AVX512 SHA1DC_INLINE
void scan16_transpose_c(const __m512i b[16], u32v16 W[16])
{
	int k;

	for (k = 0; k < 4; k++) {
		__m512i d0 = _mm512_shuffle_i32x4(b[k], b[4 + k], 0x88);
		__m512i d1 = _mm512_shuffle_i32x4(b[k], b[4 + k], 0xdd);
		__m512i d2 = _mm512_shuffle_i32x4(b[8 + k], b[12 + k], 0x88);
		__m512i d3 = _mm512_shuffle_i32x4(b[8 + k], b[12 + k], 0xdd);

		W[k] = (u32v16)_mm512_shuffle_i32x4(d0, d2, 0x88);
		W[8 + k] = (u32v16)_mm512_shuffle_i32x4(d0, d2, 0xdd);
		W[4 + k] = (u32v16)_mm512_shuffle_i32x4(d1, d3, 0x88);
		W[12 + k] = (u32v16)_mm512_shuffle_i32x4(d1, d3, 0xdd);
	}
}

/* Message expansion for t in [from, to); ubc needs W up to index 64 only. */
SHA1DC_TGT_AVX512 SHA1DC_INLINE
void scan16_expand(u32v16 W[65], int from, int to)
{
	int t;

	for (t = from; t < to; t++) {
		__m512i x = (__m512i)(W[t - 3] ^ W[t - 8] ^ W[t - 14] ^ W[t - 16]);

		W[t] = (u32v16)_mm512_rol_epi32(x, 1);
	}
}

/* Store per-block dvmask values and return the flagged-block bitmask. */
SHA1DC_TGT_AVX512 SHA1DC_INLINE
uint32_t scan16_result(u32v16 dv, uint32_t dvout[16])
{
	_mm512_storeu_si512(dvout, (__m512i)dv);
	return (uint32_t)_mm512_test_epi32_mask((__m512i)dv, (__m512i)dv);
}

/* Scan 16 consecutive blocks, one per AVX-512 lane. */
SHA1DC_TGT_AVX512
uint32_t sha1dc_fast_scan16(const unsigned char *p, uint32_t dvout[16])
{
	__m512i r[16], b[16];
	u32v16 W[65];
	u32v16 dv;

	scan16_load(p, r, 0, 16);
	scan16_transpose_ab(r, b);
	scan16_transpose_c(b, W);
	scan16_expand(W, 16, 65);
	ubc_check_v16_inline(W, &dv);
	return scan16_result(dv, dvout);
}

/*
 * Fused kernel: compress the 16 blocks of 'cur' with SHA-NI while
 * scanning the 16 blocks of 'next', interleaved at source level. The
 * sha1rnds4 dependency chain is latency-bound and leaves the vector ports
 * idle; placing a slice of scan work between consecutive block steps
 * keeps both instruction streams inside the out-of-order window.
 * The chaining value entering block i is stored into ckpt[i].
 */
SHA1DC_TGT_AVX512
uint32_t sha1dc_fast_fused16(uint32_t ihv[5], const unsigned char *cur,
			     const unsigned char *next, uint32_t dvout[16],
			     uint32_t ckpt[16][5])
{
	__m128i ABCD, E0;
	__m512i r[16], b[16];
	u32v16 W[65];
	u32v16 m[4], dv;

	sha1ni_load_state(ihv, &ABCD, &E0);

#define FUSED_BLOCK(i) do { \
	sha1ni_store_state(ABCD, E0, ckpt[i]); \
	sha1ni_block1(&ABCD, &E0, cur + (i) * 64); \
} while (0)

	FUSED_BLOCK(0);
	scan16_load(next, r, 0, 8);
	FUSED_BLOCK(1);
	scan16_load(next, r, 8, 16);
	FUSED_BLOCK(2);
	scan16_transpose_ab(r, b);
	FUSED_BLOCK(3);
	scan16_transpose_c(b, W);
	FUSED_BLOCK(4);
	scan16_expand(W, 16, 29);
	FUSED_BLOCK(5);
	scan16_expand(W, 29, 41);
	FUSED_BLOCK(6);
	scan16_expand(W, 41, 53);
	FUSED_BLOCK(7);
	scan16_expand(W, 53, 65);
	m[0] = m[1] = m[2] = m[3] = ~(u32v16){0};
	FUSED_BLOCK(8);
	ubc_check_v16_part0(W, m);
	FUSED_BLOCK(9);
	ubc_check_v16_part1(W, m);
	FUSED_BLOCK(10);
	ubc_check_v16_part2(W, m);
	FUSED_BLOCK(11);
	ubc_check_v16_part3(W, m);
	FUSED_BLOCK(12);
	ubc_check_v16_part4(W, m);
	FUSED_BLOCK(13);
	ubc_check_v16_part5(W, m);
	FUSED_BLOCK(14);
	ubc_check_v16_part6(W, m);
	FUSED_BLOCK(15);
	ubc_check_v16_part7(W, m);
#undef FUSED_BLOCK

	sha1ni_store_state(ABCD, E0, ihv);
	dv = m[0] & m[1] & m[2] & m[3];
	return scan16_result(dv, dvout);
}

/*
 * AVX2 variant of the fused kernel: compress the 8 blocks of 'cur' with
 * SHA-NI while scanning the 8 blocks of 'next'. The scan has about the
 * same instruction count as the AVX-512 one (same statements at half
 * width for half as many blocks) but only half as many block steps to
 * hide behind, so several scan slices go into each gap.
 */
SHA1DC_TGT_AVX2
uint32_t sha1dc_fast_fused8(uint32_t ihv[5], const unsigned char *cur,
			    const unsigned char *next, uint32_t dvout[8],
			    uint32_t ckpt[8][5])
{
	__m128i ABCD, E0;
	u32v8 W[65];
	u32v8 m[4], dv;

	sha1ni_load_state(ihv, &ABCD, &E0);

#define FUSED_BLOCK(i) do { \
	sha1ni_store_state(ABCD, E0, ckpt[i]); \
	sha1ni_block1(&ABCD, &E0, cur + (i) * 64); \
} while (0)

	FUSED_BLOCK(0);
	scan8_load_half(next, 0, W);
	FUSED_BLOCK(1);
	scan8_load_half(next, 1, W);
	FUSED_BLOCK(2);
	scan8_expand(W, 16, 33);
	FUSED_BLOCK(3);
	scan8_expand(W, 33, 49);
	FUSED_BLOCK(4);
	scan8_expand(W, 49, 65);
	m[0] = m[1] = m[2] = m[3] = ~(u32v8){0};
	FUSED_BLOCK(5);
	ubc_check_v8_part0(W, m);
	ubc_check_v8_part1(W, m);
	ubc_check_v8_part2(W, m);
	FUSED_BLOCK(6);
	ubc_check_v8_part3(W, m);
	ubc_check_v8_part4(W, m);
	ubc_check_v8_part5(W, m);
	FUSED_BLOCK(7);
	ubc_check_v8_part6(W, m);
	ubc_check_v8_part7(W, m);
#undef FUSED_BLOCK

	sha1ni_store_state(ABCD, E0, ihv);
	dv = m[0] & m[1] & m[2] & m[3];
	return scan8_result(dv, dvout);
}

static int sha1dc_detect_fast_level(void)
{
	unsigned int eax, ebx, ecx, edx, ebx7, lo, hi;
	uint64_t xcr0;

	if (!__get_cpuid_count(7, 0, &eax, &ebx7, &ecx, &edx))
		return 0;
	if (!__get_cpuid(1, &eax, &ebx, &ecx, &edx))
		return 0;
	if (!(ecx & (1u << 19)) || !(ecx & (1u << 27)))	/* SSE4.1, OSXSAVE */
		return 0;
	if (!(ebx7 & (1u << 29)) || !(ebx7 & (1u << 5)))	/* SHA, AVX2 */
		return 0;
	__asm__("xgetbv" : "=a"(lo), "=d"(hi) : "c"(0));
	xcr0 = ((uint64_t)hi << 32) | lo;
	if ((xcr0 & 0x06) != 0x06)			/* XMM+YMM state saved */
		return 0;
	if ((ebx7 & (1u << 16)) && (ebx7 & (1u << 30)) &&
	    (xcr0 & 0xe6) == 0xe6)			/* AVX512F+BW, ZMM state */
		return 2;
	return 1;
}

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

#endif /* __x86_64__ && __GNUC__ && !SHA1DC_NO_FAST_SHANI */
