# Set compiler according to hostname
MEXCC=/usr/local/MATLAB/R2012b/bin/mex
MATLABINC=/usr/local/MATLAB/R2012b/extern/include/ -fPIC

# Compilation targets
TARGETS=sym_plane.mex

# Make rules
.PHONY: all clean

CFLAGS=-I./ -I${MATLABINC} -O2

OBJS = Wm5DistPoint3Triangle3.o Wm5MathematicsPCH.o Wm5Distance.o Wm5Vector3.o Wm5Math.o Wm5Assert.o Wm5FileIO.o Wm5Endian.o

%.o : %.cpp
	$(CC) $(CFLAGS) -I./ -c $< -o $@ 

%.mex : %.cpp $(OBJS)
	$(MEXCC) $< $(OBJS)

all: $(TARGETS)

clean:
	rm -f *.mexglx *.mexa64 *.o
