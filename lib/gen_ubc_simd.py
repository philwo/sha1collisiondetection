#!/usr/bin/env python3
"""Generate an N-lane SIMD version of sha1dc's ubc_check using GCC vector
extensions. Each vector lane checks one 64-byte block.

Usage:
  gen_ubc_simd.py ubc_check.c LANES            -> monolithic .c on stdout
  gen_ubc_simd.py ubc_check.c LANES --header K -> .h with K inline part
                                                  functions plus an inline
                                                  monolithic wrapper
  Append --tst for the NEON-friendly statement style (see below).

Translation rules:
- 'mask &= (bitwise expr);'  -> unchanged (elementwise on vectors).
- 'if (mask & (...))' guards -> dropped (running the &= is always correct).
- 'if (mask) {' wrapper      -> early-out when all lanes are already zero.
- Tail conditions 'if (!C1 || !C2 ...) mask &= ~DV;' where each Ci is
  '((X) & (1<<K))' or '!((X) & (1<<K))' ->
      keep = product of per-lane 0/1 condition values
      mask &= -keep | ~DV;
Anything else fails loudly.

With --tst, statements of the kill-when-set shapes
  ((X>>K)&1)-1 | ~DVS        (kills DVS when bit K of X is set)
  (X&(1<<K))-(1<<K) | ~DVS   (valid because no DVS bit is below K; asserted)
  C-(X&C) | ~DVS             (valid because DVS is a subset of C; asserted)
are emitted as  mask &= ~((VT)(((X) & bit) != 0) & DVS)  instead, and tail
conditions as vector comparisons. Semantically identical (the bit-position
preconditions are asserted); on NEON each test compiles to cmtst+and+bic
with no shifts. The few bit-alignment statements (<<, ~>>) stay verbatim.

The mask is spread over 4 round-robin accumulators to break the serial
dependency chain; the final result is their AND.
"""
import re
import sys

TST = '--tst' in sys.argv
argv = [a for a in sys.argv if a != '--tst']
SRC = argv[1]
LANES = int(argv[2])
HEADER_PARTS = 0
if len(argv) > 4 and argv[3] == '--header':
    HEADER_PARTS = int(argv[4])

VT = f'u32v{LANES}'
src = open(SRC).read()

consts = re.findall(r'static const uint32_t DV_\w+\s*=\s*\(uint32_t\)\(1\) << \d+;', src)
assert len(consts) == 32, len(consts)
DV_BIT = {m.group(1): int(m.group(2)) for m in
          re.finditer(r'static const uint32_t (DV_\w+)\s*=\s*\(uint32_t\)\(1\) << (\d+);', src)}


def balanced(s):
    return s.count('(') == s.count(')')


def strip_shift(y):
    """Split '(X>>K)' into (X, K); otherwise (y, 0)."""
    my = re.fullmatch(r'\((.*)>>(\d+)\)', y)
    if my and balanced(my.group(1)):
        return my.group(1), int(my.group(2))
    return y, 0


def tst_plain(rhs):
    """Rewrite one 'mask &= (COND | ~DVS);' rhs into tst style, or return
    None for the bit-alignment shapes (kept verbatim)."""
    m = re.fullmatch(r'\((.*) \| ~(?:\((DV_[\w|]+)\)|(DV_\w+))\);', rhs.strip())
    assert m, rhs
    cond = m.group(1)
    dvs = m.group(2) or m.group(3)
    dvbits = [DV_BIT[n] for n in dvs.split('|')]

    # bit-alignment shapes rely on the DV bit position; keep them verbatim
    if re.fullmatch(r'\(\(\(.*\)<<\d+\)\)|\(~\(.*\)\)', cond):
        return None

    # (C-(X&C)): kill when set; original keeps only C's bits when clear
    mc = re.fullmatch(r'\(((?:\(1<<\d+\))|1)-\((.*)&((?:\(1<<\d+\))|1)\)\)', cond)
    if mc and mc.group(1) == mc.group(3) and balanced(mc.group(2)):
        c = eval(mc.group(1))
        assert all((1 << b) == c for b in dvbits), (rhs, c)
        x, k = mc.group(2), 0
        if c == 1:
            x, k = strip_shift(x)
            c = 1 << k
        return f'~(({VT})((({x}) & (uint32_t){c}u) != 0) & ({dvs}));'

    # (0-(X&C)): kill when CLEAR; original kills bits below C's bit when set
    md = re.fullmatch(r'\(0-\((.*)&((?:\(1<<\d+\))|1)\)\)', cond)
    if md and balanced(md.group(1)):
        c = eval(md.group(2))
        assert all((1 << b) & (c - 1) == 0 for b in dvbits), (rhs, c)
        x, k = md.group(1), 0
        if c == 1:
            x, k = strip_shift(x)
            c = 1 << k
        return f'~(({VT})((({x}) & (uint32_t){c}u) == 0) & ({dvs}));'

    # ((X&(1<<K))-(1<<K)): kill when set; original kills bits < K when clear
    mb = re.fullmatch(r'\(\((.*)&\(1<<(\d+)\)\)-\(1<<(\d+)\)\)', cond)
    if mb and mb.group(2) == mb.group(3) and balanced(mb.group(1)):
        k = int(mb.group(2))
        assert all(b >= k for b in dvbits), (rhs, k)
        return f'~(({VT})((({mb.group(1)}) & (1u<<{k})) != 0) & ({dvs}));'

    # (((X>>K)&1)-1) or ((X&1)-1): kill when bit K (or 0) of X is set
    ma = re.fullmatch(r'\(\((.*)&1\)-1\)', cond)
    assert ma, rhs
    x, k = strip_shift(ma.group(1))
    return f'~(({VT})((({x}) & (1u<<{k})) != 0) & ({dvs}));'

m = re.search(r'void ubc_check\(const uint32_t W\[80\], uint32_t dvmask\[1\]\)\n\{\n(.*)\n\}', src, re.S)
body = m.group(1)
assert 'dvmask[0]=mask;' in body, 'body truncated: final store missing'

# ---- parse into a list of statement chunks + early-out marker position ----
stmts = []          # each entry: list of source lines (using m[0..3] accs)
marker_at = None    # index into stmts where the original 'if (mask) {' sat
acc = 0


def next_acc():
    global acc
    a = f'm[{acc}]'
    acc = (acc + 1) % 4
    return a


lines = body.split('\n')
i = 0
n_plain = n_guard = n_cond = 0
while i < len(lines):
    line = lines[i].strip()
    if not line or line == '}' or line == 'uint32_t mask = ~((uint32_t)(0));' \
       or line == 'dvmask[0]=mask;':
        i += 1
        continue
    if line == 'if (mask) {':
        marker_at = len(stmts)
        i += 1
        continue
    if line.startswith('mask &='):
        rhs = line[len('mask &='):].strip()
        t = tst_plain(rhs) if TST else None
        if t is not None:
            stmts.append(['\t' + next_acc() + ' &= ' + t])
        else:
            stmts.append(['\t' + next_acc() + ' &=' + line[len('mask &='):]])
        n_plain += 1
        i += 1
        continue
    gm = re.fullmatch(r'if \(mask & (?:\()?DV_\w+(?:\|DV_\w+)*(?:\))?\)', line)
    if gm:
        n_guard += 1
        i += 1  # drop the guard, next statement runs unconditionally
        continue
    if line == 'if (':
        cond_lines = []
        i += 1
        while True:
            l = lines[i].strip()
            i += 1
            em = re.fullmatch(r'\)\s*mask &= ~(DV_\w+);', l)
            if em:
                dv = em.group(1)
                break
            cond_lines.append(l)
        cond = ' '.join(cond_lines)
        terms = [t.strip() for t in re.split(r'\|\|', cond)]
        keeps = []
        for t in terms:
            mm2 = re.fullmatch(r'!\(!\((.*) & \(1<<(\d+)\)\)\)', t)
            if mm2:
                x, k = mm2.group(1), mm2.group(2)
                neg = True
            else:
                mm = re.fullmatch(r'!\((.*) & \(1<<(\d+)\)\)', t)
                assert mm, t
                x, k = mm.group(1), mm.group(2)
                neg = False
            if TST:
                # keep is a full-lane mask (-1 keeps) instead of 0/1;
                # each source term is a kill condition, so keep negates it
                expr = f'(({VT})((({x}) & (1u<<{k})) {"==" if neg else "!="} 0))'
            else:
                expr = f'((({x}) >> {k}) & 1)'
                if neg:
                    expr = f'({expr} ^ 1)'
            keeps.append(expr)
        n_cond += 1
        chunk = ['\t{', f'\t\t{VT} keep = {keeps[0]};']
        for k in keeps[1:]:
            chunk.append(f'\t\tkeep &= {k};')
        if TST:
            chunk.append(f'\t\t{next_acc()} &= keep | ~{dv};')
        else:
            chunk.append(f'\t\t{next_acc()} &= -keep | ~{dv};')
        chunk.append('\t}')
        stmts.append(chunk)
        continue
    raise SystemExit(f'unhandled line: {line!r}')

assert marker_at is not None

# ---- emit ----
out = []
out.append(f'/* Generated by gen_ubc_simd.py from ubc_check.c ({LANES} lanes). Do not edit. */')
guard = f'UBC_CHECK_V{LANES}_H'
if HEADER_PARTS:
    out.append(f'#ifndef {guard}\n#define {guard}')
out.append('#include <stdint.h>\n')
out.append(f'typedef uint32_t {VT} __attribute__((vector_size({LANES * 4})));\n')
out.append(f'''static inline __attribute__((always_inline)) int ubc_v{LANES}_all_zero(const {VT} *v)
{{
	uint32_t a = 0;
	int i;
	for (i = 0; i < {LANES}; i++)
		a |= (*v)[i];
	return a == 0;
}}
''')
out.append('#ifndef SHA1DC_UBC_DV_BITS\n#define SHA1DC_UBC_DV_BITS')
out.extend(consts)
out.append('#endif /* SHA1DC_UBC_DV_BITS */')
out.append('''
#ifndef SHA1DC_UBC_INLINE
#define SHA1DC_UBC_INLINE static inline __attribute__((always_inline))
#endif
''')

if not HEADER_PARTS:
    out.append(f'void ubc_check_v{LANES}(const {VT} W[65], {VT} dvmask[1])')
    out.append('{')
    out.append(f'\t{VT} m[4] = {{ ~({VT}){{0}}, ~({VT}){{0}}, ~({VT}){{0}}, ~({VT}){{0}} }};\n')
    for idx, chunk in enumerate(stmts):
        if idx == marker_at:
            out.append('\tm[0] &= m[1] & m[2] & m[3];')
            out.append(f'\tm[1] = m[2] = m[3] = ~({VT}){{0}};')
            out.append(f'\tif (ubc_v{LANES}_all_zero(&m[0])) {{ dvmask[0] = m[0]; return; }}')
        out.extend(chunk)
    out.append('\tdvmask[0] = m[0] & m[1] & m[2] & m[3];')
    out.append('}')
else:
    K = HEADER_PARTS
    per = (len(stmts) + K - 1) // K
    bounds = [min(k * per, len(stmts)) for k in range(K + 1)]
    # early-out goes at the first part boundary at or after the marker
    eo_part = min(k for k in range(1, K + 1) if bounds[k] >= marker_at)
    for k in range(K):
        out.append(f'SHA1DC_UBC_INLINE void ubc_check_v{LANES}_part{k}(const {VT} W[65], {VT} m[4])')
        out.append('{')
        for chunk in stmts[bounds[k]:bounds[k + 1]]:
            out.extend(chunk)
        out.append('}\n')
    out.append(f'#define UBC_V{LANES}_NPARTS {K}')
    out.append(f'#define UBC_V{LANES}_EARLYOUT_AFTER_PART {eo_part}\n')
    out.append(f'SHA1DC_UBC_INLINE void ubc_check_v{LANES}_inline(const {VT} W[65], {VT} dvmask[1])')
    out.append('{')
    out.append(f'\t{VT} m[4] = {{ ~({VT}){{0}}, ~({VT}){{0}}, ~({VT}){{0}}, ~({VT}){{0}} }};\n')
    for k in range(K):
        out.append(f'\tubc_check_v{LANES}_part{k}(W, m);')
        if k + 1 == eo_part:
            out.append('\tm[0] &= m[1] & m[2] & m[3];')
            out.append(f'\tm[1] = m[2] = m[3] = ~({VT}){{0}};')
            out.append(f'\tif (ubc_v{LANES}_all_zero(&m[0])) {{ dvmask[0] = m[0]; return; }}')
    out.append('\tdvmask[0] = m[0] & m[1] & m[2] & m[3];')
    out.append('}')
    out.append(f'#endif /* {guard} */')

print('\n'.join(out))
sys.stderr.write(f'plain={n_plain} guards={n_guard} cond={n_cond} stmts={len(stmts)} marker={marker_at}\n')
