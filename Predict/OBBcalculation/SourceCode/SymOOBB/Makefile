# Set compiler according to hostname
MEXCC=/usr/local/MATLAB/R2012b/bin/mex
MATLABINC=/usr/local/MATLAB/R2012b/extern/include/ -fPIC

# Compilation targets
TARGETS=oobb.mex

# Make rules
.PHONY: all clean

CFLAGS=-I./ -I${MATLABINC}

OBJS = Wm5ContBox3.o Wm5MathematicsPCH.o Wm5Vector3.o Wm5Math.o Wm5Assert.o Wm5ApprGaussPointsFit3.o Wm5Quaternion.o Wm5EigenDecomposition.o Wm5Matrix2.o Wm5Matrix3.o Wm5SingularValueDecomposition.o Wm5Memory.o Wm5CorePCH.o Wm5Mutex.o Wm5Matrix4.o Wm5Vector4.o Wm5Vector2.o Wm5Query.o Wm5ContMinBox2.o Wm5ContMinBox3.o Wm5ConvexHull2.o Wm5ConvexHull1.o Wm5ConvexHull.o Wm5ConvexHull3.o Wm5FileIO.o Wm5Endian.o

%.o : %.cpp
	$(CC) $(CFLAGS) -I./ -c $< -o $@ 

%.mex : %.cpp $(OBJS)
	$(MEXCC) $< $(OBJS)

all: $(TARGETS)

clean:
	rm -f *.mexglx *.mexa64 *.o
