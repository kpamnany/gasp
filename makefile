# gasp -- global address space toolbox
#
# makefile
#

CC=mpicc

CRAY?=no
INTEL?=no

ifneq ($(wildcard /opt/cray/.),)
    CRAY=yes
endif

.SUFFIXES: .c .h .o .a
.PHONY: clean test

CFLAGS+=-Wall
CFLAGS+=-std=c11
CFLAGS+=-D_GNU_SOURCE
CFLAGS+=-fpic
CFLAGS+=-I./include
CFLAGS+=-I./src

ifdef TRACE_DTREE
    CFLAGS+=-DTRACE_DTREE=$(TRACE_DTREE)
endif
ifdef SHOW_DTREE
    CFLAGS+=-DSHOW_DTREE=1
endif

ifeq ($(CRAY),yes)
    CC=cc
    #CFLAGS+=-craympich-mt
    CFLAGS+=-DSDE_TRACING=1
    LDFLAGS+=-Wl,--whole-archive,-ldmapp,--no-whole-archive
endif

ifeq ($(INTEL),yes)
    CC=mpiicc
    CFLAGS+=-mt_mpi
    CFLAGS+=-DSDE_TRACING=1
endif

SRCS=src/gasp.c src/garray.c src/dtree.c src/log.c
OBJS=$(subst .c,.o, $(SRCS))

ifeq ($(DEBUG),yes)
    CFLAGS+=-O0 -g
else
    CFLAGS+=-O2
endif

ifeq ($(shell uname), Darwin)
    TARGET=libgasp.dylib
else
    TARGET=libgasp.so
endif


all: $(TARGET)

test: $(TARGET)
	$(MAKE) -C test

$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) -shared -o $(TARGET) $(LDFLAGS) $(OBJS)

.c.o:
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	$(MAKE) -C test clean
	$(RM) -f $(TARGET) $(OBJS)

