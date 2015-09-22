################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CPP_SRCS += \
../src/common/AccumulativeRefinement.cpp \
../src/common/Maths.cpp \
../src/common/Refinement.cpp \
../src/common/RoiCluster.cpp 

OBJS += \
./src/common/AccumulativeRefinement.o \
./src/common/Maths.o \
./src/common/Refinement.o \
./src/common/RoiCluster.o 

CPP_DEPS += \
./src/common/AccumulativeRefinement.d \
./src/common/Maths.d \
./src/common/Refinement.d \
./src/common/RoiCluster.d 


# Each subdirectory must supply rules for building sources it contributes
src/common/%.o: ../src/common/%.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: NVCC Compiler'
	/usr/local/cuda-6.5/bin/nvcc -I/usr/local/include/opencv -G -g -O0 -gencode arch=compute_35,code=sm_35  -odir "src/common" -M -o "$(@:%.o=%.d)" "$<"
	/usr/local/cuda-6.5/bin/nvcc -I/usr/local/include/opencv -G -g -O0 --compile  -x c++ -o  "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


