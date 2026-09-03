##
## Copyright 2017 Marc Stevens <marc@marc-stevens.nl>, Dan Shumow (danshu@microsoft.com)
## Distributed under the MIT Software License.
## See accompanying file LICENSE.txt or copy at
## https://opensource.org/licenses/MIT
##

# dynamic library compatibility
# 1. If the library source code has changed at all since the last update,
#    then increment revision (‘c:r:a’ becomes ‘c:r+1:a’).
# 2. If any interfaces have been added, removed, or changed since the last update,
#    increment current, and set revision to 0.
# 3. If any interfaces have been added since the last public release, then increment age.
# 4. If any interfaces have been removed or changed since the last public release,
#    then set age to 0.
LIBCOMPAT=1:0:0

PREFIX ?= /usr/local
BINDIR=$(PREFIX)/bin
LIBDIR=$(PREFIX)/lib
INCLUDEDIR=$(PREFIX)/include/sha1dc

CC ?= gcc
LD ?= gcc
CC_DEP ?= $(CC)

ifeq ($(shell uname),Darwin)
LIBTOOL ?= glibtool
INSTALL ?= install
else
LIBTOOL ?= libtool
INSTALL ?= install
endif


CFLAGS=-O2 -Wall -Werror -Wextra -pedantic -std=c90 -Ilib
LDFLAGS=

LT_CC:=$(LIBTOOL) --tag=CC --mode=compile $(CC)
LT_CC_DEP:=$(CC)
LT_LD:=$(LIBTOOL) --tag=CC --mode=link $(CC)
LT_INSTALL:=$(LIBTOOL) --tag=CC --mode=install $(INSTALL)

MKDIR=mkdir -p

ifneq (, $(shell which $(LIBTOOL) 2>/dev/null ))
CC:=$(LT_CC)
CC_DEP:=$(LT_CC_DEP)
LD:=$(LT_LD)
LDLIB:=$(LT_LD)
LIB_EXT:=la
else
LIB_EXT:=a
LD:=$(CC)
LT_INSTALL:=$(INSTALL)
endif

CFLAGS+=$(TARGETCFLAGS)
LDFLAGS+=$(TARGETLDFLAGS)

# CFLAGS for the sources that need C99 and GNU extensions (see below)
FAST_CFLAGS=$(filter-out -std=c90 -pedantic,$(CFLAGS)) -std=gnu99


LIB_DIR=lib
LIB_DEP_DIR=dep_lib
LIB_OBJ_DIR=obj_lib
SRC_DIR=src
SRC_DEP_DIR=dep_src
SRC_OBJ_DIR=obj_src

H_DEP:=$(shell find . -type f -name "*.h")
FS_LIB=$(wildcard $(LIB_DIR)/*.c)
FS_SRC=$(wildcard $(SRC_DIR)/*.c)
FS_OBJ_LIB=$(FS_LIB:$(LIB_DIR)/%.c=$(LIB_OBJ_DIR)/%.lo)
FS_OBJ_SRC=$(FS_SRC:$(SRC_DIR)/%.c=$(SRC_OBJ_DIR)/%.lo)
FS_OBJ=$(FS_OBJ_SRC) $(FS_OBJ_LIB)
FS_DEP_LIB=$(FS_LIB:$(LIB_DIR)/%.c=$(LIB_DEP_DIR)/%.d)
FS_DEP_SRC=$(FS_SRC:$(SRC_DIR)/%.c=$(SRC_DEP_DIR)/%.d)
FS_DEP=$(FS_DEP_SRC) $(FS_DEP_LIB)

.SUFFIXES: .c .d

.PHONY: all
all: library tools

.PHONY: install
install: all
	$(LT_INSTALL) -d $(LIBDIR) $(BINDIR) $(INCLUDEDIR)
	$(LT_INSTALL) bin/libsha1detectcoll.$(LIB_EXT) $(LIBDIR)/libsha1detectcoll.$(LIB_EXT)
	$(LT_INSTALL) lib/sha1.h $(INCLUDEDIR)/sha1.h
	$(LT_INSTALL) bin/sha1dcsum $(BINDIR)/sha1dcsum
	$(LT_INSTALL) bin/sha1dcsum_partialcoll $(BINDIR)/sha1dcsum_partialcoll

.PHONY: uninstall
uninstall:
	-$(RM) $(BINDIR)/sha1dcsum
	-$(RM) $(BINDIR)/sha1dcsum_partialcoll
	-$(RM) $(INCLUDEDIR)/sha1.h
	-$(RM) $(LIBDIR)/libsha1detectcoll.$(LIB_EXT)

.PHONY: clean
clean:
	-find . -type f -name '*.a' -print -delete
	-find . -type f -name '*.d' -print -delete
	-find . -type f -name '*.o' -print -delete
	-find . -type f -name '*.la' -print -delete
	-find . -type f -name '*.lo' -print -delete
	-find . -type f -name '*.so' -print -delete
	-find . -type d -name '.libs' -print | xargs rm -rv
	-rm -rf bin

.PHONY: test
test: tools
	test e98a60b463a6868a6ce351ab0166c0af0c8c4721 != `bin/sha1dcsum test/sha1_reducedsha_coll.bin | cut -d' ' -f1` || (echo "\nError: Compiled for incorrect endianness" && false)
	test a56374e1cf4c3746499bc7c0acb39498ad2ee185 = `bin/sha1dcsum test/sha1_reducedsha_coll.bin | cut -d' ' -f1`
	test 16e96b70000dd1e7c85b8368ee197754400e58ec = `bin/sha1dcsum test/shattered-1.pdf | cut -d' ' -f1`
	test e1761773e6a35916d99f891b77663e6405313587 = `bin/sha1dcsum test/shattered-2.pdf | cut -d' ' -f1`
	test dd39885a2a5d8f59030b451e00cb45da9f9d3828 = `bin/sha1dcsum_partialcoll test/sha1_reducedsha_coll.bin | cut -d' ' -f1` 
	test d3a1d09969c3b57113fd17b23e01dd3de74a99bb = `bin/sha1dcsum_partialcoll test/shattered-1.pdf | cut -d' ' -f1`
	test 92246b0b718f4c704d37bb025717cbc66babf102 = `bin/sha1dcsum_partialcoll test/shattered-2.pdf | cut -d' ' -f1`
	bin/sha1dcsum test/*
	bin/sha1dcsum_partialcoll test/*
	bin/sha1perf test -c test/shattered-1.pdf test/shattered-2.pdf test/sha-mbles-1.bin test/sha-mbles-2.bin test/sha1_reducedsha_coll.bin
	bin/sha1perf digest
	bin/sha1perf vtest 200
	$(MAKE) check-generated

# lib/ubc_check_simd.h is generated from lib/ubc_check.c; verify it is in sync.
.PHONY: check-generated
check-generated:
	@if command -v python3 >/dev/null 2>&1; then \
		$(MKDIR) $(LIB_OBJ_DIR) && \
		python3 lib/gen_ubc_simd.py lib/ubc_check.c --all > $(LIB_OBJ_DIR)/ubc_check_simd.h.tmp 2>/dev/null && \
		cmp lib/ubc_check_simd.h $(LIB_OBJ_DIR)/ubc_check_simd.h.tmp && \
		echo "lib/ubc_check_simd.h is up to date"; \
	else \
		echo "check-generated skipped: python3 not found"; \
	fi

.PHONY: check
check: test

.PHONY: tools
tools: sha1dcsum sha1dcsum_partialcoll sha1perf

.PHONY: sha1dcsum
sha1dcsum: bin/sha1dcsum

.PHONY: sha1dcsum_partialcoll
sha1dcsum_partialcoll: bin/sha1dcsum_partialcoll

# Benchmark and diff-test harness for the fast path. SHA1PERF_OPENSSL=1
# adds the OpenSSL SHA-1 reference (needs -lcrypto).
.PHONY: sha1perf
sha1perf: bin/sha1perf

ifdef SHA1PERF_OPENSSL
SHA1PERF_CFLAGS=-DSHA1PERF_OPENSSL
SHA1PERF_LDLIBS=-lcrypto
endif


.PHONY: library
library: bin/libsha1detectcoll.$(LIB_EXT)

bin/libsha1detectcoll.la: $(FS_OBJ_LIB)
	$(MKDIR) $(shell dirname $@) && $(LDLIB) $(LDFLAGS) $(FS_OBJ_LIB) -rpath $(LIBDIR) -version-info $(LIBCOMPAT) -o bin/libsha1detectcoll.la
	
bin/libsha1detectcoll.a: $(FS_OBJ_LIB)
	$(MKDIR) $(shell dirname $@) && $(AR) cru bin/libsha1detectcoll.a $(FS_OBJ_LIB)

bin/sha1dcsum: $(FS_OBJ_SRC) bin/libsha1detectcoll.$(LIB_EXT)
	$(LD) $(LDFLAGS) $(FS_OBJ_SRC) -Lbin -lsha1detectcoll -o bin/sha1dcsum

bin/sha1dcsum_partialcoll: $(FS_OBJ_SRC) bin/libsha1detectcoll.$(LIB_EXT)
	$(LD) $(LDFLAGS) $(FS_OBJ_SRC) -Lbin -lsha1detectcoll -o bin/sha1dcsum_partialcoll

$(SRC_OBJ_DIR)/sha1perf.lo $(SRC_OBJ_DIR)/sha1perf.o: override CFLAGS:=$(FAST_CFLAGS) $(SHA1PERF_CFLAGS)
$(SRC_OBJ_DIR)/sha1perf.lo $(SRC_OBJ_DIR)/sha1perf.o: sha1perf.c $(H_DEP)
	$(MKDIR) $(shell dirname $@) && $(CC) $(CFLAGS) -o $@ -c $<

bin/sha1perf: $(SRC_OBJ_DIR)/sha1perf.lo bin/libsha1detectcoll.$(LIB_EXT)
	$(LD) $(LDFLAGS) $(SRC_OBJ_DIR)/sha1perf.lo -Lbin -lsha1detectcoll $(SHA1PERF_LDLIBS) -o bin/sha1perf


$(SRC_DEP_DIR)/%.d: $(SRC_DIR)/%.c
	$(MKDIR) $(shell dirname $@) && $(CC_DEP) $(CFLAGS) -M -MF $@ $<

$(SRC_OBJ_DIR)/%.lo ${SRC_OBJ_DIR}/%.o: ${SRC_DIR}/%.c ${SRC_DEP_DIR}/%.d $(H_DEP)
	$(MKDIR) $(shell dirname $@) && $(CC) $(CFLAGS) -o $@ -c $<


# The hardware fast paths and sha1perf use intrinsics and GNU extensions
# and cannot be built as pedantic C90 (FAST_CFLAGS, defined next to
# CFLAGS above). The backends only contain code on x86-64 / aarch64
# GCC/Clang and compile to empty objects elsewhere. 'override' keeps this
# in effect when CFLAGS is given on the command line.
$(LIB_DEP_DIR)/sha1dc_fast_x86.d $(LIB_OBJ_DIR)/sha1dc_fast_x86.lo $(LIB_OBJ_DIR)/sha1dc_fast_x86.o: override CFLAGS:=$(FAST_CFLAGS)
$(LIB_DEP_DIR)/sha1dc_fast_arm64.d $(LIB_OBJ_DIR)/sha1dc_fast_arm64.lo $(LIB_OBJ_DIR)/sha1dc_fast_arm64.o: override CFLAGS:=$(FAST_CFLAGS)

$(LIB_DEP_DIR)/%.d: $(LIB_DIR)/%.c
	$(MKDIR) $(shell dirname $@) && $(CC_DEP) $(CFLAGS) -M -MF $@ $<

$(LIB_OBJ_DIR)/%.lo $(LIB_OBJ_DIR)/%.o: $(LIB_DIR)/%.c $(LIB_DEP_DIR)/%.d $(H_DEP)
	$(MKDIR) $(shell dirname $@) && $(CC) $(CFLAGS) -o $@ -c $<

-include $(FS_DEP)
