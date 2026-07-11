/*
 * Standalone bench + diff-test harness for git's sha1dc fast path.
 *
 * Build (from the sha1collisiondetection checkout root):
 *   gcc -O2 -Ilib -DSHA1DC_INIT_SAFE_HASH_DEFAULT=0 \
 *       lib/sha1.c lib/ubc_check.c lib/sha1dc_fast_x86.c \
 *       lib/sha1dc_fast_arm64.c sha1perf.c -lcrypto -o sha1perf
 *
 * A stock-sha1dc baseline build (no fast path linked) additionally needs
 * -DBASELINE -DSHA1DC_NO_FAST_SHANI and drops the sha1dc_fast_*.c files.
 *
 * Modes:
 *   sha1perf bench <level> <random|zeros|FILE> [chunksize]
 *   sha1perf benchnd <level> <random|zeros|FILE> [chunksize]  detection off
 *   sha1perf benchssl <random|zeros|FILE>        OpenSSL SHA-1 reference
 *   sha1perf digest              cross-level + OpenSSL digest equality
 *   sha1perf vtest <iters>       SIMD scan / fused / states diff-tests
 *   sha1perf micro <iters>       verify-states component timing
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>
#include <openssl/sha.h>

#include "sha1.h"
#include "ubc_check.h"

/* the AVX2 tier (scan8/fused8) exists only in the x86-64 fast path */
#if defined(__x86_64__) && !defined(BASELINE)
#define SHA1PERF_HAVE_TIER1 1
#endif

/* internals of the fast path (see sha1dc/sha1.c) */
#ifdef BASELINE
int sha1dc_fast_level;	/* no fast path linked; writes to it are inert */
#else
extern int sha1dc_fast_level;
#endif
void sha1dc_fast_compress(uint32_t ihv[5], const unsigned char *data, size_t len);
void sha1dc_fast_compress_ckpt(uint32_t ihv[5], const unsigned char *p,
			       unsigned nblocks, uint32_t ckpt[][5]);
uint32_t sha1dc_fast_scan16(const unsigned char *p, uint32_t dvout[16]);
uint32_t sha1dc_fast_fused16(uint32_t ihv[5], const unsigned char *cur,
			     const unsigned char *next, uint32_t dvout[16],
			     uint32_t ckpt[16][5]);
#ifdef SHA1PERF_HAVE_TIER1
uint32_t sha1dc_fast_scan8(const unsigned char *p, uint32_t dvout[8]);
uint32_t sha1dc_fast_fused8(uint32_t ihv[5], const unsigned char *cur,
			    const unsigned char *next, uint32_t dvout[8],
			    uint32_t ckpt[8][5]);
#endif
void sha1dc_fast_states(const unsigned char *block, uint32_t W[80],
			const uint32_t ihvin[5],
			uint32_t state58[5], uint32_t state65[5]);
void sha1_compression_states(uint32_t ihv[5], const uint32_t m[16],
			     uint32_t W[80], uint32_t states[80][5]);

static uint64_t rngstate = 0x9E3779B97F4A7C15ULL;
static uint64_t rng64(void)
{
	uint64_t x = rngstate;
	x ^= x << 13; x ^= x >> 7; x ^= x << 17;
	rngstate = x;
	return x;
}

static void fill_random(unsigned char *p, size_t n)
{
	size_t i;
	for (i = 0; i + 8 <= n; i += 8) {
		uint64_t v = rng64();
		memcpy(p + i, &v, 8);
	}
	for (; i < n; i++)
		p[i] = (unsigned char)rng64();
}

static double now(void)
{
	struct timespec ts;
	clock_gettime(CLOCK_MONOTONIC, &ts);
	return ts.tv_sec + ts.tv_nsec * 1e-9;
}

static int g_nodc;	/* bench with collision detection disabled */
static int g_ssl;	/* bench OpenSSL SHA-1 instead of sha1dc */

static void dc_digest(int level, const unsigned char *p, size_t n,
		      size_t chunk, unsigned char out[20], int *coll)
{
	SHA1_CTX ctx;
	size_t off = 0;

	if (g_ssl) {
		SHA1(p, n, out);
		*coll = 0;
		return;
	}
	sha1dc_fast_level = level;
	SHA1DCInit(&ctx);
	if (g_nodc)
		SHA1DCSetUseDetectColl(&ctx, 0);
	if (!chunk)
		chunk = n ? n : 1;
	while (off < n) {
		size_t c = n - off < chunk ? n - off : chunk;
		SHA1DCUpdate(&ctx, (const char *)p + off, c);
		off += c;
	}
	*coll = SHA1DCFinal(out, &ctx);
}

static int cmd_bench(int argc, char **argv)
{
	int level = atoi(argv[0]);
	const char *kind = argv[1];
	size_t chunk = argc > 2 ? (size_t)atol(argv[2]) : 0;
	size_t n = 64 << 20;
	unsigned char *buf;
	unsigned char md[20];
	int coll, reps = 0;
	double t0, t1, best = 0;
	int r;

	buf = malloc(n);
	if (!strcmp(kind, "random"))
		fill_random(buf, n);
	else if (!strcmp(kind, "zeros"))
		memset(buf, 0, n);
	else {
		FILE *f = fopen(kind, "rb");
		size_t got;
		if (!f) { perror(kind); return 1; }
		got = fread(buf, 1, n, f);
		fclose(f);
		/* tile the file to fill the buffer */
		while (got && got < n) {
			size_t c = n - got < got ? n - got : got;
			memcpy(buf + got, buf, c);
			got += c;
		}
	}

	/* warmup + measure: 5 runs, report best */
	for (r = 0; r < 6; r++) {
		t0 = now();
		dc_digest(level, buf, n, chunk, md, &coll);
		t1 = now();
		if (r && (best == 0 || t1 - t0 < best))
			best = t1 - t0;
		reps++;
	}
	printf("%s%s%d %s chunk=%zu: %.3f GB/s (best of %d)\n",
	       g_ssl ? "openssl " : "", g_nodc ? "nodc level=" : "level=",
	       level, kind, chunk, n / best / 1e9, reps - 1);
	free(buf);
	return 0;
}

static int cmd_digest(void)
{
	/* lengths crossing block/group/padding boundaries */
	size_t lens[400];
	int nlens = 0, i, li;
	size_t chunks[] = { 0, 64, 4096, 8192, 12345 };
	unsigned char *buf;
	int bad = 0;

	for (i = 0; i <= 130; i++) lens[nlens++] = i;
	for (i = 440; i <= 580; i += 7) lens[nlens++] = i;
	for (i = 960; i <= 1100; i += 9) lens[nlens++] = i;
	for (i = 1980; i <= 2120; i += 11) lens[nlens++] = i;
	lens[nlens++] = 100000; lens[nlens++] = 1000000;
	lens[nlens++] = 1048576; lens[nlens++] = 1048577;
	lens[nlens++] = 16777216;

	buf = malloc(16777216);
	fill_random(buf, 16777216);

	for (li = 0; li < nlens; li++) {
		unsigned char ref[20], md[20];
		size_t n = lens[li];
		int coll, ci, lvl;

		SHA1((const unsigned char *)buf, n, ref);
		for (lvl = 0; lvl <= 2; lvl++) {
			for (ci = 0; ci < (int)(sizeof(chunks)/sizeof(chunks[0])); ci++) {
				dc_digest(lvl, buf, n, chunks[ci], md, &coll);
				if (memcmp(ref, md, 20) || coll) {
					printf("MISMATCH len=%zu level=%d chunk=%zu coll=%d\n",
					       n, lvl, chunks[ci], coll);
					bad++;
				}
			}
		}
	}
	/* also zeros and a repeating pattern */
	memset(buf, 0, 1048577);
	for (li = 0; li < 3; li++) {
		size_t n = 1048575 + li;
		unsigned char ref[20], md[20];
		int coll, lvl;
		SHA1(buf, n, ref);
		for (lvl = 0; lvl <= 2; lvl++) {
			dc_digest(lvl, buf, n, 8192, md, &coll);
			if (memcmp(ref, md, 20) || coll) {
				printf("MISMATCH zeros len=%zu level=%d\n", n, lvl);
				bad++;
			}
		}
	}
	printf(bad ? "digest: %d FAILURES\n" : "digest: all OK\n", bad);
	free(buf);
	return bad != 0;
}

#ifdef BASELINE
static int cmd_vtest(int iters)
{
	(void)iters;
	fprintf(stderr, "vtest not available in BASELINE build\n");
	return 2;
}
#else
static void expand_block(const unsigned char *p, uint32_t W[80])
{
	int t;
	for (t = 0; t < 16; t++)
		W[t] = ((uint32_t)p[t*4] << 24) | ((uint32_t)p[t*4+1] << 16)
		     | ((uint32_t)p[t*4+2] << 8) | (uint32_t)p[t*4+3];
	/* see SHA1DC_NOVECTOR in sha1.c: auto-vectorizing this recurrence
	 * causes store-forwarding stalls */
#if defined(__clang__)
#pragma clang loop vectorize(disable)
#elif defined(__GNUC__) && __GNUC__ >= 14
#pragma GCC novector
#endif
	for (t = 16; t < 80; t++) {
		uint32_t x = W[t-3] ^ W[t-8] ^ W[t-14] ^ W[t-16];
		W[t] = (x << 1) | (x >> 31);
	}
}

static int cmd_vtest(int iters)
{
	unsigned char *buf = malloc(4096);	/* 2 groups of 16 blocks + slack */
	int it, bad = 0;

	for (it = 0; it < iters && bad < 20; it++) {
		uint32_t dvs16[16], dvsf[16];
		uint32_t ckpt[16][5], ckref[17][5];
		uint32_t ihv[5], ihv2[5], flags, ref;
#ifdef SHA1PERF_HAVE_TIER1
		uint32_t dvs8[8], dvsf8[8], ckpt8[8][5], flags8;
#endif
		int i;

		/* random data; sometimes copy one block's prefix over others
		 * to raise correlation between lanes */
		fill_random(buf, 4096);
		if (it & 1)
			for (i = 1; i < 8; i++)
				memcpy(buf + i * 64, buf, 48);

		/* reference: scalar ubc_check per block */
		ref = 0;
		flags = sha1dc_fast_scan16(buf, dvs16);
#ifdef SHA1PERF_HAVE_TIER1
		flags8 = sha1dc_fast_scan8(buf, dvs8);
#endif
		for (i = 0; i < 16; i++) {
			uint32_t Wx[80], dv[1] = { 0 };

			expand_block(buf + i * 64, Wx);
			ubc_check(Wx, dv);
			if (dv[0])
				ref |= 1u << i;
			if (dvs16[i] != dv[0]) {
				printf("scan16 dv lane %d it=%d: got %08x want %08x\n", i, it, dvs16[i], dv[0]);
				bad++;
			}
#ifdef SHA1PERF_HAVE_TIER1
			if (i < 8 && dvs8[i] != dv[0]) {
				printf("scan8 dv lane %d it=%d: got %08x want %08x\n", i, it, dvs8[i], dv[0]);
				bad++;
			}
#endif
		}
		if (flags != ref) {
			printf("scan16 flags mismatch it=%d: got %04x want %04x\n", it, flags, ref);
			bad++;
		}
#ifdef SHA1PERF_HAVE_TIER1
		if (flags8 != (ref & 0xff)) {
			printf("scan8 flags mismatch it=%d: got %02x want %02x\n", it, flags8, ref & 0xff);
			bad++;
		}
#endif

		/* fused16: ihv advance, ckpt chain, flags of 'next' group */
		ihv[0] = 0x67452301 ^ (uint32_t)rng64(); ihv[1] = 0xEFCDAB89;
		ihv[2] = 0x98BADCFE; ihv[3] = 0x10325476; ihv[4] = 0xC3D2E1F0 ^ (uint32_t)rng64();
		memcpy(ihv2, ihv, sizeof(ihv));
		memcpy(ckref[0], ihv, sizeof(ihv));
		for (i = 0; i < 16; i++) {
			memcpy(ckref[i + 1], ckref[i], sizeof(ihv));
			sha1dc_fast_compress(ckref[i + 1], buf + i * 64, 64);
		}
		flags = sha1dc_fast_fused16(ihv, buf, buf + 1024, dvsf, ckpt);
		if (memcmp(ihv, ckref[16], sizeof(ihv))) {
			printf("fused16 ihv mismatch it=%d\n", it); bad++;
		}
		for (i = 0; i < 16; i++)
			if (memcmp(ckpt[i], ckref[i], sizeof(ihv))) {
				printf("fused16 ckpt[%d] mismatch it=%d\n", i, it); bad++;
			}
		{
			uint32_t want[16];
			uint32_t wf = sha1dc_fast_scan16(buf + 1024, want);
			if (flags != wf || memcmp(want, dvsf, sizeof(want))) {
				printf("fused16 nextflags mismatch it=%d\n", it); bad++;
			}
		}

#ifdef SHA1PERF_HAVE_TIER1
		/* fused8 (AVX2 tier) */
		memcpy(ihv, ihv2, sizeof(ihv));
		flags8 = sha1dc_fast_fused8(ihv, buf, buf + 512, dvsf8, ckpt8);
		if (memcmp(ihv, ckref[8], sizeof(ihv))) {
			printf("fused8 ihv mismatch it=%d\n", it); bad++;
		}
		for (i = 0; i < 8; i++)
			if (memcmp(ckpt8[i], ckref[i], sizeof(ihv))) {
				printf("fused8 ckpt[%d] mismatch it=%d\n", i, it); bad++;
			}
		{
			uint32_t want8[8];
			uint32_t wf8 = sha1dc_fast_scan8(buf + 512, want8);
			if (flags8 != wf8 || memcmp(want8, dvsf8, sizeof(want8))) {
				printf("fused8 nextflags mismatch it=%d\n", it); bad++;
			}
		}
#else
		(void)ihv2;
#endif /* SHA1PERF_HAVE_TIER1 */

		/* fast states58/65 vs scalar sha1_compression_states */
		for (i = 0; i < 4; i++) {
			uint32_t m16[16], Wref[80], Wfast[80], states[80][5];
			uint32_t ihvx[5], ihvy[5], s58[5], s65[5];
			int t;

			for (t = 0; t < 5; t++)
				ihvx[t] = ihvy[t] = (uint32_t)rng64();
			for (t = 0; t < 16; t++)
				memcpy(&m16[t], buf + i * 64 + t * 4, 4); /* native order for compression_states */
			sha1_compression_states(ihvx, m16, Wref, states);

			sha1dc_fast_states(buf + i * 64, Wfast, ihvy, s58, s65);
			if (memcmp(Wref, Wfast, sizeof(Wref))) {
				printf("W expansion mismatch it=%d blk=%d\n", it, i); bad++;
			}
			if (memcmp(s58, states[58], sizeof(s58))) {
				printf("states[58] mismatch it=%d blk=%d: got %08x.. want %08x..\n",
				       it, i, s58[0], states[58][0]);
				bad++;
			}
			if (memcmp(s65, states[65], sizeof(s65))) {
				printf("states[65] mismatch it=%d blk=%d: got %08x.. want %08x..\n",
				       it, i, s65[0], states[65][0]);
				bad++;
			}
		}
	}
	printf(bad ? "vtest: %d FAILURES\n" : "vtest: all OK (%d iters)\n", bad ? bad : iters);
	free(buf);
	return bad != 0;
}
#endif /* !BASELINE */

#ifndef BASELINE
/* time scalar sha1_compression_states vs expand+sha1dc_fast_states */
static int cmd_micro(int iters)
{
	unsigned char *buf = malloc(64 * 256);
	uint32_t ihv[5] = { 0x67452301, 0xEFCDAB89, 0x98BADCFE, 0x10325476, 0xC3D2E1F0 };
	static uint32_t states[80][5];
	uint32_t W[80], s58[5], s65[5], sink = 0;
	double t0, t1;
	int it;

	fill_random(buf, 64 * 256);

	t0 = now();
	for (it = 0; it < iters; it++) {
		uint32_t m16[16];
		uint32_t ihvx[5];
		memcpy(ihvx, ihv, sizeof(ihv));
		memcpy(m16, buf + (it & 255) * 64, 64);
		sha1_compression_states(ihvx, m16, W, states);
		sink ^= states[58][0] ^ states[65][0] ^ ihvx[0];
	}
	t1 = now();
	printf("scalar compression_states: %.1f ns/block\n", (t1 - t0) / iters * 1e9);

	t0 = now();
	for (it = 0; it < iters; it++) {
		sha1dc_fast_states(buf + (it & 255) * 64, W, ihv, s58, s65);
		sink ^= s58[0] ^ s65[0] ^ W[79];
	}
	t1 = now();
	printf("fast_states (fused expand): %.1f ns/block (sink %08x)\n",
	       (t1 - t0) / iters * 1e9, sink);

	/* group-kernel isolation: compress vs scan vs fused, per 16 blocks */
	{
		uint32_t ckpt[16][5], dvs[16];
		int groups = iters / 16 + 1;

		t0 = now();
		for (it = 0; it < groups; it++)
			sha1dc_fast_compress_ckpt(ihv, buf + (it & 15) * 1024, 16, ckpt);
		t1 = now();
		printf("compress_ckpt16: %7.1f ns/group (%.3f GB/s)\n",
		       (t1 - t0) / groups * 1e9, groups * 1024.0 / (t1 - t0) / 1e9);

		t0 = now();
		for (it = 0; it < groups; it++)
			sink ^= sha1dc_fast_scan16(buf + (it & 15) * 1024, dvs);
		t1 = now();
		printf("scan16:          %7.1f ns/group (%.3f GB/s)\n",
		       (t1 - t0) / groups * 1e9, groups * 1024.0 / (t1 - t0) / 1e9);

		t0 = now();
		for (it = 0; it < groups; it++)
			sink ^= sha1dc_fast_fused16(ihv, buf + (it & 15) * 1024,
						    buf + ((it + 1) & 15) * 1024, dvs, ckpt);
		t1 = now();
		printf("fused16:         %7.1f ns/group (%.3f GB/s, sink %08x)\n",
		       (t1 - t0) / groups * 1e9, groups * 1024.0 / (t1 - t0) / 1e9, sink);
	}
	free(buf);
	return 0;
}
#endif

/* hash a file at every level with detection on; report the coll flag */
static int cmd_coll(const char *path)
{
	unsigned char *buf = malloc(64 << 20);
	unsigned char md[20];
	FILE *f = fopen(path, "rb");
	size_t n;
	int lvl, coll, i;

	if (!f) { perror(path); return 1; }
	n = fread(buf, 1, 64 << 20, f);
	fclose(f);
	for (lvl = 0; lvl <= 2; lvl++) {
		dc_digest(lvl, buf, n, 8192, md, &coll);
		printf("level=%d coll=%d ", lvl, coll);
		for (i = 0; i < 20; i++)
			printf("%02x", md[i]);
		printf(" %s\n", path);
	}
	free(buf);
	return 0;
}

int main(int argc, char **argv)
{
	if (argc >= 3 && !strcmp(argv[1], "coll"))
		return cmd_coll(argv[2]);
#ifndef BASELINE
	if (argc >= 3 && !strcmp(argv[1], "micro"))
		return cmd_micro(atoi(argv[2]));
#endif
	if (argc >= 4 && !strcmp(argv[1], "bench"))
		return cmd_bench(argc - 2, argv + 2);
	if (argc >= 4 && !strcmp(argv[1], "benchnd")) {
		g_nodc = 1;
		return cmd_bench(argc - 2, argv + 2);
	}
	if (argc >= 3 && !strcmp(argv[1], "benchssl")) {
		static char *fake[3];
		g_ssl = 1;
		fake[0] = "0";
		fake[1] = argv[2];
		fake[2] = argc > 3 ? argv[3] : NULL;
		return cmd_bench(argc - 2, fake);
	}
	if (argc >= 2 && !strcmp(argv[1], "digest"))
		return cmd_digest();
	if (argc >= 3 && !strcmp(argv[1], "vtest"))
		return cmd_vtest(atoi(argv[2]));
	fprintf(stderr, "usage: %s bench <level> <random|zeros|FILE> [chunk] | digest | vtest <iters>\n", argv[0]);
	return 2;
}
