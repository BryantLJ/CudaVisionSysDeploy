################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CU_SRCS += \
../src/visionSys.cu 

CU_DEPS += \
./src/visionSys.d 

OBJS += \
./src/visionSys.o 


# Each subdirectory must supply rules for building sources it contributes
src/%.o: ../src/%.cu
	@echo 'Building file: $<'
	@echo 'Invoking: NVCC Compiler'
	/usr/local/cuda-7.5/bin/nvcc -I/usr/local/include/opencv -I/usr/local/include/cuda-7.5/targets/x86_64-linux/include -lineinfo -pg -O3 --use_fast_math -std=c++11 -gencode arch=compute_52,code=sm_52  -odir "src" -M -o "$(@:%.o=%.d)" "$<"
	/usr/local/cuda-7.5/bin/nvcc -I/usr/local/include/opencv -I/usr/local/include/cuda-7.5/targets/x86_64-linux/include -lineinfo -pg -O3 --use_fast_math -std=c++11 --compile --relocatable-device-code=true -gencode arch=compute_52,code=compute_52 -gencode arch=compute_52,code=sm_52  -x cu -o  "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


