################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CU_SRCS += \
../src/init/cParameters.cu 

CU_DEPS += \
./src/init/cParameters.d 

OBJS += \
./src/init/cParameters.o 


# Each subdirectory must supply rules for building sources it contributes
src/init/%.o: ../src/init/%.cu
	@echo 'Building file: $<'
	@echo 'Invoking: NVCC Compiler'
	/usr/local/cuda-7.5/bin/nvcc -I/usr/local/include/ -I/usr/local/cuda-7.5/targets/x86_64-linux/include -G -g -lineinfo -pg -O0 -std=c++11 -gencode arch=compute_52,code=sm_52  -odir "src/init" -M -o "$(@:%.o=%.d)" "$<"
	/usr/local/cuda-7.5/bin/nvcc -I/usr/local/include/ -I/usr/local/cuda-7.5/targets/x86_64-linux/include -G -g -lineinfo -pg -O0 -std=c++11 --compile --relocatable-device-code=true -gencode arch=compute_52,code=compute_52 -gencode arch=compute_52,code=sm_52  -x cu -o  "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


