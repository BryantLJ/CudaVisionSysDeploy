BUILDDIR :=bin/object/			# Directory to store object files
CC :=nvcc				# Compiler
NAMEBIN :=visionSys			# Name of the binary file
#ARCH :=52				# Device target architecture

# @@@@@@@@@@@@@@@@@@@@@@@@ Compiler flags @@@@@@@@@@@@@@@@@@@@@@@@ 
FLAGS := -lineinfo -O3 --use_fast_math -std=c++11 -gencode arch=compute_52,code=compute_52 -gencode arch=compute_52,code=sm_52 --relocatable-device-code=true
CCFLAGS := -c

# @@@@@@@@@@@@@@@@@@@@@@@@ Library paths @@@@@@@@@@@@@@@@@@@@@@@@ 
LIBPATH := -L/usr/local/cuda/targets/x86_64-linux/lib -L/usr/local/lib

# @@@@@@@@@@@@@@@@@@@@@@@@ Libraries @@@@@@@@@@@@@@@@@@@@@@@@ 
LIBS := -lnvToolsExt -lopencv_objdetect -lopencv_imgcodecs -lopencv_videoio -lopencv_calib3d -lopencv_features2d -lopencv_video -lopencv_ml -lopencv_highgui -lopencv_imgproc -lopencv_core -lopencv_flann

# @@@@@@@@@@@@@@@@@@@@@@@@ Inlcude paths @@@@@@@@@@@@@@@@@@@@@@@@ 
INCL := -I/usr/local/include/ -I/usr/local/cuda/targets/x86_64-linux/include

# @@@@@@@@@@@@@@@@@@@@@@@@ Object files @@@@@@@@@@@@@@@@@@@@@@@@ 
OBJS := bin/object/Utils.o bin/object/IniFileIO.o bin/object/cParameters.o bin/object/AccumulativeRefinement.o bin/object/Maths.o bin/object/Refinement.o bin/object/RoiCluster.o bin/object/visionSys.o


all: ${OBJS}
	@echo "Building target..."
	${CC} ${FLAGS} ${LIBPATH} --cudart shared -link -o bin/${NAMEBIN} ${OBJS} ${LIBS}

bin/object/visionSys.o: src/visionSys.cu
	${CC} ${FLAGS} ${CCFLAGS} ${LIBPATH} ${INCL} $? -o $@ ${LIBS}

bin/object/Utils.o: src/init/fileInOut/Utils.cpp
	${CC} ${FLAGS} ${CCFLAGS} ${LIBPATH} ${INCL} $? -o $@ ${LIBS}

bin/object/IniFileIO.o: src/init/fileInOut/IniFileIO.cpp
	${CC} ${FLAGS} ${CCFLAGS} ${LIBPATH} ${INCL} $? -o $@ ${LIBS}

bin/object/cParameters.o: src/init/cParameters.cu
	${CC} ${FLAGS} ${CCFLAGS} ${LIBPATH} ${INCL} $? -o $@ ${LIBS}

bin/object/AccumulativeRefinement.o: src/common/AccumulativeRefinement.cpp
	${CC} ${FLAGS} ${CCFLAGS} ${LIBPATH} ${INCL} $? -o $@ ${LIBS}

bin/object/Maths.o: src/common/Maths.cpp
	${CC} ${FLAGS} ${CCFLAGS} ${LIBPATH} ${INCL} $? -o $@ ${LIBS}

bin/object/Refinement.o: src/common/Refinement.cpp
	${CC} ${FLAGS} ${CCFLAGS} ${LIBPATH} ${INCL} $? -o $@ ${LIBS}

bin/object/RoiCluster.o: src/common/RoiCluster.cpp
	${CC} ${FLAGS} ${CCFLAGS} ${LIBPATH} ${INCL} $? -o $@ ${LIBS}

clean:
	@echo "Cleaning up..."
	rm bin/object/*
	rm bin/${NAMEBIN}


