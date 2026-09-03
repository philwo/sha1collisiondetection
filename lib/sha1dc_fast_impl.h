/*
 * Helpers shared by the two fast-path backends (sha1dc_fast_x86.c and
 * sha1dc_fast_arm64.c). Not included by sha1.c: this file uses C99 and
 * GNU extensions.
 */
#ifndef SHA1DC_FAST_IMPL_H
#define SHA1DC_FAST_IMPL_H

#define SHA1DC_INLINE static inline __attribute__((always_inline))

#define rol32(x, n) (((x) << (n)) | ((x) >> (32 - (n))))

/*
 * Finish sha1dc_fast_states: the backend has run the message schedule
 * (W[80]) and rounds 0-55 with the hardware instructions and passes the
 * working state a..e entering round 56. Rounds 56-64 run here as scalar
 * code and the snapshots at steps 58 and 65 are stored in the
 * rotating-variable order of SHA1_STORE_STATE in sha1.c: at step 58 the
 * stored words are {D,E,A,B,C} of the logical state, at step 65 they are
 * {A,B,C,D,E}.
 */
SHA1DC_INLINE
void sha1dc_fast_states_tail(uint32_t a, uint32_t b, uint32_t c, uint32_t d,
			     uint32_t e, const uint32_t W[80],
			     uint32_t state58[5], uint32_t state65[5])
{
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

#endif /* SHA1DC_FAST_IMPL_H */
