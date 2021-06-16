#
# Folder structure
#

# Remember to create the folders needed

BIN     := ./bin/
SRC     := ./src/
DEST    := ./obj/

EXE ?= LBM

#
# Executables
#

CC     = gcc
CP     = g++
HIP   = hipcc
RM     = rm

#
# Macros
#

PREC ?= 64
IN ?= 4
BN ?= 3

#
# C/C++ flags
#

CFLAGS    = -Wall -DPREC=$(PREC)
CPPFLAGS  = -Wall -DPREC=$(PREC)

#
# CUDA flags
#

HIPARCH	= -arch=compute_50 -code=sm_50
HIPFLAGS = $(HIPARCH) -DIN=$(IN) -DBN=$(BN) -DPREC=$(PREC)
# -Xptxas=-v
# --maxrregcount=16 
# -Xptxas -dlcm=ca --ptxas-options=-v --use_fast_math

#
# Files to compile: 
#

MAIN   = main.cu
CODC   = 
CODCPP = files.cpp
CODCU  = LBM.cu setup.cu

#
# Formating the folder structure for compiling/linking/cleaning.
#

FC     = 
FCPP   = cpp/
FCU    = cu/

#
# Preparing variables for automated prerequisites
#

OBJC   = $(patsubst %.c,$(DEST)$(FC)%.o,$(CODC))
OBJCPP = $(patsubst %.cpp,$(DEST)$(FCPP)%.o,$(CODCPP))
OBJCU  = $(patsubst %.cu,$(DEST)$(FCU)%.o,$(CODCU))

SRCMAIN = $(patsubst %,$(SRC)%,$(MAIN))
OBJMAIN = $(patsubst $(SRC)%.cu,$(DEST)%.o,$(SRCMAIN))

#
# The MAGIC
#

all:  $(BIN)$(EXE)

$(BIN)$(EXE): $(OBJC) $(OBJCPP) $(OBJCU) $(OBJMAIN)
	$(HIP) $(HIPFLAGS) $^ -o $@

$(OBJMAIN): $(SRCMAIN)
	$(HIP) $(HIPFLAGS) -dc $? -o $@

$(OBJCPP): $(DEST)%.o : $(SRC)%.cpp
	$(CP) $(CPPFLAGS) -c $? -o $@

$(OBJCU): $(DEST)%.o : $(SRC)%.cu
	$(HIP) $(HIPFLAGS) -dc $? -o $@

$(OBJC): $(DEST)%.o : $(SRC)%.c
	$(CC) $(CFLAGS) -c $? -o $@

#
# Makefile for cleaning
# 

clean:
	$(RM) -rf $(DEST)*.o
	$(RM) -rf $(DEST)$(FC)*.o
	$(RM) -rf $(DEST)$(FCPP)*.o
	$(RM) -rf $(DEST)$(FCU)*.o

fresh:
	$(RM) -rf outputs/*

distclean: clean
	$(RM) -rf $(BIN)*

