# gasp -- global address space toolbox
#
# makefile for tests
#

CRAY?=no

CC?=cc

.SUFFIXES: .c .h .o
.PHONY: clean

CFLAGS+=-Wall
CFLAGS+=-std=c11
CFLAGS+=-D_GNU_SOURCE
CFLAGS+=-fpic
CFLAGS+=-I.
CFLAGS+=-I../include
CFLAGS+=-I../src
CFLAGS+=-I$(INCLUDE)

ifeq ($(CRAY),yes)
    LDFLAGS+=-dynamic
    LDFLAGS+=-L..
    LDFLAGS+=-lgasp
    LDFLAGS+=-Wl,--whole-archive,-ldmapp,--no-whole-archive
else
ifeq ($(shell uname), Darwin)
    #LDFLAGS+=-dynamic
    #LDFLAGS+=-L..
    #LDFLAGS+=-lgasp
    LDFLAGS+=../libgasp.dylib
else
    LDFLAGS+=../libgasp.so
endif
endif

ifeq ($(INTEL),yes)
    CC=mpiicc
    CFLAGS+=-mt_mpi
endif


ifneq ($(MKLROOT),)
    LDFLAGS+=-L$(MKLROOT)/lib/intel64
    LDFLAGS+=-lmkl_intel_ilp64
    LDFLAGS+=-lmkl_core
    LDFLAGS+=-lmkl_sequential
endif
LDFLAGS+=-lpthread
LDFLAGS+=-lm
LDFLAGS+=-ldl

SRCS=garraytest.c dtreetest.c
OBJS=$(subst .c,.o, $(SRCS))
BINS=$(subst .c,, $(SRCS))

ifeq ($(DEBUG),yes)
    CFLAGS+=-O0 -g
else
    CFLAGS+=-O3
endif

all: $(BINS)

garraytest: $(SRCS)
	$(CC) $(CFLAGS) garraytest.c -o garraytest $(LDFLAGS)

dtreetest: dtreetest.c
	$(CC) $(CFLAGS) -openmp dtreetest.c -o dtreetest $(LDFLAGS)

clean:
	$(RM) -f $(BINS) $(OBJS)

