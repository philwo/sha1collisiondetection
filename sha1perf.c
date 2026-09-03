/*
 * Benchmark and diff-test harness for sha1dc's hardware fast path.
 *
 * Build: make sha1perf (SHA1PERF_OPENSSL=1 adds the OpenSSL SHA-1
 * reference and benchssl mode, needs -lcrypto). A stock-sha1dc baseline
 * build is the same with -DSHA1DC_NO_FAST_SHANI in CFLAGS.
 *
 * Modes:
 *   sha1perf test [-c] FILE...   digest/collision equality of every level
 *                                and chunk size vs the scalar path, with
 *                                reduced-round detection off and on; -c
 *                                also requires a collision to be detected
 *                                with it on (make test)
 *   sha1perf digest              same on random/zero data across lengths
 *   sha1perf vtest <iters>       SIMD scan / fused / states diff-tests
 *   sha1perf bench <level> <random|zeros|FILE> [chunksize]
 *   sha1perf benchnd <level> <random|zeros|FILE> [chunksize]  detection off
 *   sha1perf benchssl <random|zeros|FILE> [chunksize]  OpenSSL reference
 *   sha1perf coll FILE           digest + coll flag of FILE at every level
 *   sha1perf micro <iters>       verify-states component timing
 *
 * Levels above what the CPU supports (or above SHA1DC_FAST_LEVEL) are
 * reported as unsupported rather than executed.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>
#ifdef SHA1PERF_OPENSSL
#include <openssl/sha.h>
#endif

#include "sha1.h"
#include "ubc_check.h"
#include "sha1dc_fast.h"

#ifndef SHA1DC_FAST_SHANI
int sha1dc_fast_level;	/* no fast path compiled in; only level 0 exists */
#endif

void sha1_compression_states(uint32_t ihv[5], const uint32_t m[16],
			     uint32_t W[80], uint32_t states[80][5]);

static int g_max_level;	/* sha1dc_fast_level as detected at startup */
static int g_nodc;	/* bench with collision detection disabled */
static int g_ssl;	/* bench OpenSSL SHA-1 instead of sha1dc */
static int g_rr;	/* detect reduced-round collisions */

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

static void *xmalloc(size_t n)
{
	void *p = malloc(n);
	if (!p) {
		fprintf(stderr, "out of memory (%zu bytes)\n", n);
		exit(1);
	}
	return p;
}

/* Read up to cap bytes of path into buf; returns the size or 0 on error. */
static size_t read_file(const char *path, unsigned char *buf, size_t cap)
{
	FILE *f = fopen(path, "rb");
	size_t got;

	if (!f) {
		perror(path);
		return 0;
	}
	got = fread(buf, 1, cap, f);
	if (ferror(f)) {
		perror(path);
		got = 0;
	} else if (!got)
		fprintf(stderr, "%s: empty file\n", path);
	fclose(f);
	return got;
}

static int level_ok(int level)
{
	if (level >= 0 && level <= g_max_level)
		return 1;
	fprintf(stderr, "level %d unsupported on this CPU (max %d)\n",
		level, g_max_level);
	return 0;
}

/*
 * Hash n bytes at the given dispatch level in chunks of 'chunk' bytes
 * (0 = one call) with safe_hash off, so SHA1DCUpdate takes the fast
 * path. The caller checks level <= g_max_level.
 */
static void dc_digest(int level, const unsigned char *p, size_t n,
		      size_t chunk, unsigned char out[20], int *coll)
{
	SHA1_CTX ctx;
	size_t off = 0;

#ifdef SHA1PERF_OPENSSL
	if (g_ssl) {
		SHA1(p, n, out);
		*coll = 0;
		return;
	}
#endif
	sha1dc_fast_level = level;
	SHA1DCInit(&ctx);
	SHA1DCSetSafeHash(&ctx, 0);
	if (g_nodc)
		SHA1DCSetUseDetectColl(&ctx, 0);
	if (g_rr)
		SHA1DCSetDetectReducedRoundCollision(&ctx, 1);
	if (!chunk)
		chunk = n ? n : 1;
	while (off < n) {
		size_t c = n - off < chunk ? n - off : chunk;
		SHA1DCUpdate(&ctx, (const char *)p + off, c);
		off += c;
	}
	*coll = SHA1DCFinal(out, &ctx);
	sha1dc_fast_level = g_max_level;
}

static void print_hex(const unsigned char *md)
{
	int i;
	for (i = 0; i < 20; i++)
		printf("%02x", md[i]);
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

	if (!level_ok(level))
		return 2;
	buf = xmalloc(n);
	if (!strcmp(kind, "random"))
		fill_random(buf, n);
	else if (!strcmp(kind, "zeros"))
		memset(buf, 0, n);
	else {
		size_t got = read_file(kind, buf, n);
		if (!got) {
			free(buf);
			return 1;
		}
		/* tile the file to fill the buffer */
		while (got < n) {
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

/*
 * Compare every level and chunk size against the level-0 (scalar path)
 * one-call result. Returns the number of mismatches.
 */
static int check_levels(const unsigned char *p, size_t n, const char *what,
			int expect_coll)
{
	static const size_t chunks[] = { 0, 1, 63, 64, 65, 511, 512, 513, 1000,
					 1023, 1024, 1025, 4096, 8192, 12345 };
	unsigned char ref[20], md[20];
	int refcoll, coll, lvl, ci, bad = 0;

	dc_digest(0, p, n, 0, ref, &refcoll);
	if (expect_coll && g_rr && !refcoll) {
		printf("%s: no collision detected\n", what);
		bad++;
	}
	for (lvl = 0; lvl <= g_max_level; lvl++) {
		for (ci = 0; ci < (int)(sizeof(chunks) / sizeof(chunks[0])); ci++) {
			dc_digest(lvl, p, n, chunks[ci], md, &coll);
			if (memcmp(ref, md, 20) || coll != refcoll) {
				printf("MISMATCH %s len=%zu level=%d chunk=%zu rr=%d: coll=%d want %d, ",
				       what, n, lvl, chunks[ci], g_rr, coll, refcoll);
				print_hex(md);
				printf(" want ");
				print_hex(ref);
				printf("\n");
				bad++;
			}
		}
	}
	return bad;
}

static int cmd_test(int argc, char **argv)
{
	size_t cap = 64 << 20;
	unsigned char *buf = xmalloc(cap);
	int expect_coll = 0, i, bad = 0;

	if (argc > 0 && !strcmp(argv[0], "-c")) {
		expect_coll = 1;
		argc--;
		argv++;
	}
	if (argc == 0) {
		fprintf(stderr, "test: no files given\n");
		free(buf);
		return 2;
	}
	for (i = 0; i < argc; i++) {
		size_t n = read_file(argv[i], buf, cap);
		unsigned char md[20];
		int coll, rrcoll;

		if (!n) {
			bad++;
			continue;
		}
		for (g_rr = 0; g_rr <= 1; g_rr++)
			bad += check_levels(buf, n, argv[i], expect_coll);
		g_rr = 1;
		dc_digest(0, buf, n, 0, md, &rrcoll);
		g_rr = 0;
		dc_digest(0, buf, n, 0, md, &coll);
		printf("coll=%d rrcoll=%d ", coll, rrcoll);
		print_hex(md);
		printf(" %s\n", argv[i]);
	}
	printf(bad ? "test: %d FAILURES (max level %d)\n" : "test: all OK (max level %d)\n",
	       bad ? bad : g_max_level, g_max_level);
	free(buf);
	return bad != 0;
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

	buf = xmalloc(16777216);
	fill_random(buf, 16777216);

	for (li = 0; li < nlens; li++) {
		unsigned char ref[20], md[20];
		size_t n = lens[li];
		int coll, ci, lvl;

		/* reference: the scalar path (and OpenSSL when available) */
		dc_digest(0, buf, n, 0, ref, &coll);
		if (coll) {
			printf("MISMATCH len=%zu level=0: coll=1 on random data\n", n);
			bad++;
		}
#ifdef SHA1PERF_OPENSSL
		SHA1(buf, n, md);
		if (memcmp(ref, md, 20)) {
			printf("MISMATCH len=%zu level=0 vs OpenSSL\n", n);
			bad++;
		}
#endif
		for (lvl = 0; lvl <= g_max_level; lvl++) {
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
		dc_digest(0, buf, n, 0, ref, &coll);
		for (lvl = 0; lvl <= g_max_level; lvl++) {
			dc_digest(lvl, buf, n, 8192, md, &coll);
			if (memcmp(ref, md, 20) || coll) {
				printf("MISMATCH zeros len=%zu level=%d\n", n, lvl);
				bad++;
			}
		}
	}
	printf(bad ? "digest: %d FAILURES (max level %d)\n" : "digest: all OK (max level %d)\n",
	       bad ? bad : g_max_level, g_max_level);
	free(buf);
	return bad != 0;
}

#ifndef SHA1DC_FAST_SHANI
static int cmd_vtest(int iters)
{
	(void)iters;
	printf("vtest: skipped (no fast path compiled in)\n");
	return 0;
}

static int cmd_micro(int iters)
{
	(void)iters;
	fprintf(stderr, "micro: no fast path compiled in\n");
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

/*
 * Diff-test the group kernels against scalar references. scan16/fused16
 * need level 2; scan8/fused8 (x86-64 only) need level 1; the states
 * check only needs the hardware compress (level >= 1).
 */
static int cmd_vtest(int iters)
{
	unsigned char *buf = xmalloc(4096);	/* 2 groups of 16 blocks + slack */
	int it, bad = 0;
	int have16 = g_max_level >= 2;
#ifdef SHA1DC_FAST_HAVE_TIER1
	int have8 = g_max_level >= 1;
#endif

	if (g_max_level < 1) {
		printf("vtest: skipped (fast path not available on this CPU)\n");
		free(buf);
		return 0;
	}
	for (it = 0; it < iters && bad < 20; it++) {
		uint32_t dvs16[16], dvsf[16];
		uint32_t ckpt[16][5], ckref[17][5];
		uint32_t ihv[5], ihv2[5], flags = 0, ref;
#ifdef SHA1DC_FAST_HAVE_TIER1
		uint32_t dvs8[8], dvsf8[8], ckpt8[8][5], flags8 = 0;
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
		if (have16)
			flags = sha1dc_fast_scan16(buf, dvs16);
#ifdef SHA1DC_FAST_HAVE_TIER1
		if (have8)
			flags8 = sha1dc_fast_scan8(buf, dvs8);
#endif
		for (i = 0; i < 16; i++) {
			uint32_t Wx[80], dv[1] = { 0 };

			expand_block(buf + i * 64, Wx);
			ubc_check(Wx, dv);
			if (dv[0])
				ref |= 1u << i;
			if (have16 && dvs16[i] != dv[0]) {
				printf("scan16 dv lane %d it=%d: got %08x want %08x\n", i, it, dvs16[i], dv[0]);
				bad++;
			}
#ifdef SHA1DC_FAST_HAVE_TIER1
			if (have8 && i < 8 && dvs8[i] != dv[0]) {
				printf("scan8 dv lane %d it=%d: got %08x want %08x\n", i, it, dvs8[i], dv[0]);
				bad++;
			}
#endif
		}
		if (have16 && flags != ref) {
			printf("scan16 flags mismatch it=%d: got %04x want %04x\n", it, flags, ref);
			bad++;
		}
#ifdef SHA1DC_FAST_HAVE_TIER1
		if (have8 && flags8 != (ref & 0xff)) {
			printf("scan8 flags mismatch it=%d: got %02x want %02x\n", it, flags8, ref & 0xff);
			bad++;
		}
#endif

		/* checkpoint chain reference via single-block compress */
		ihv[0] = 0x67452301 ^ (uint32_t)rng64(); ihv[1] = 0xEFCDAB89;
		ihv[2] = 0x98BADCFE; ihv[3] = 0x10325476; ihv[4] = 0xC3D2E1F0 ^ (uint32_t)rng64();
		memcpy(ihv2, ihv, sizeof(ihv));
		memcpy(ckref[0], ihv, sizeof(ihv));
		for (i = 0; i < 16; i++) {
			memcpy(ckref[i + 1], ckref[i], sizeof(ihv));
			sha1dc_fast_compress(ckref[i + 1], buf + i * 64, 64);
		}

		/* fused16: ihv advance, ckpt chain, flags of 'next' group */
		if (have16) {
			uint32_t want[16], wf;

			flags = sha1dc_fast_fused16(ihv, buf, buf + 1024, dvsf, ckpt);
			if (memcmp(ihv, ckref[16], sizeof(ihv))) {
				printf("fused16 ihv mismatch it=%d\n", it); bad++;
			}
			for (i = 0; i < 16; i++)
				if (memcmp(ckpt[i], ckref[i], sizeof(ihv))) {
					printf("fused16 ckpt[%d] mismatch it=%d\n", i, it); bad++;
				}
			wf = sha1dc_fast_scan16(buf + 1024, want);
			if (flags != wf || memcmp(want, dvsf, sizeof(want))) {
				printf("fused16 nextflags mismatch it=%d\n", it); bad++;
			}
		}

#ifdef SHA1DC_FAST_HAVE_TIER1
		/* fused8 (AVX2 tier) */
		if (have8) {
			uint32_t want8[8], wf8;

			memcpy(ihv, ihv2, sizeof(ihv));
			flags8 = sha1dc_fast_fused8(ihv, buf, buf + 512, dvsf8, ckpt8);
			if (memcmp(ihv, ckref[8], sizeof(ihv))) {
				printf("fused8 ihv mismatch it=%d\n", it); bad++;
			}
			for (i = 0; i < 8; i++)
				if (memcmp(ckpt8[i], ckref[i], sizeof(ihv))) {
					printf("fused8 ckpt[%d] mismatch it=%d\n", i, it); bad++;
				}
			wf8 = sha1dc_fast_scan8(buf + 512, want8);
			if (flags8 != wf8 || memcmp(want8, dvsf8, sizeof(want8))) {
				printf("fused8 nextflags mismatch it=%d\n", it); bad++;
			}
		}
#else
		(void)ihv2;
#endif /* SHA1DC_FAST_HAVE_TIER1 */

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
	printf(bad ? "vtest: %d FAILURES\n" : "vtest: all OK (%d iters, max level %d)\n",
	       bad ? bad : iters, g_max_level);
	free(buf);
	return bad != 0;
}

/* time scalar sha1_compression_states vs expand+sha1dc_fast_states */
static int cmd_micro(int iters)
{
	unsigned char *buf;
	uint32_t ihv[5] = { 0x67452301, 0xEFCDAB89, 0x98BADCFE, 0x10325476, 0xC3D2E1F0 };
	static uint32_t states[80][5];
	uint32_t W[80], s58[5], s65[5], sink = 0;
	double t0, t1;
	int it;

	if (!level_ok(2))
		return 2;
	buf = xmalloc(64 * 256);
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
#endif /* SHA1DC_FAST_SHANI */

/* hash a file at every level with detection on; report the coll flag */
static int cmd_coll(const char *path)
{
	size_t cap = 64 << 20;
	unsigned char *buf = xmalloc(cap);
	unsigned char md[20];
	size_t n = read_file(path, buf, cap);
	int lvl, coll;

	if (!n) {
		free(buf);
		return 1;
	}
	for (lvl = 0; lvl <= g_max_level; lvl++) {
		dc_digest(lvl, buf, n, 8192, md, &coll);
		printf("level=%d coll=%d ", lvl, coll);
		print_hex(md);
		printf(" %s\n", path);
	}
	free(buf);
	return 0;
}

int main(int argc, char **argv)
{
	g_max_level = sha1dc_fast_level;

	if (argc >= 3 && !strcmp(argv[1], "test"))
		return cmd_test(argc - 2, argv + 2);
	if (argc >= 3 && !strcmp(argv[1], "coll"))
		return cmd_coll(argv[2]);
	if (argc >= 3 && !strcmp(argv[1], "micro"))
		return cmd_micro(atoi(argv[2]));
	if (argc >= 4 && !strcmp(argv[1], "bench"))
		return cmd_bench(argc - 2, argv + 2);
	if (argc >= 4 && !strcmp(argv[1], "benchnd")) {
		g_nodc = 1;
		return cmd_bench(argc - 2, argv + 2);
	}
	if (argc >= 3 && !strcmp(argv[1], "benchssl")) {
#ifdef SHA1PERF_OPENSSL
		char *fake[3];
		int nfake = 2;

		g_ssl = 1;
		fake[0] = "0";
		fake[1] = argv[2];
		if (argc > 3)
			fake[nfake++] = argv[3];
		return cmd_bench(nfake, fake);
#else
		fprintf(stderr, "benchssl: built without OpenSSL (make SHA1PERF_OPENSSL=1)\n");
		return 2;
#endif
	}
	if (argc >= 2 && !strcmp(argv[1], "digest"))
		return cmd_digest();
	if (argc >= 3 && !strcmp(argv[1], "vtest"))
		return cmd_vtest(atoi(argv[2]));
	fprintf(stderr,
		"usage: %s test [-c] FILE... | digest | vtest <iters> |\n"
		"       bench|benchnd <level> <random|zeros|FILE> [chunk] |\n"
		"       benchssl <random|zeros|FILE> [chunk] | coll FILE | micro <iters>\n",
		argv[0]);
	return 2;
}
