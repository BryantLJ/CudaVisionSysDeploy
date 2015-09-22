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
	/usr/local/cuda-6.5/bin/nvcc -I/usr/local/include/opencv -G -g -O0 -gencode arch=compute_35,code=sm_35  -odir "src/init" -M -o "$(@:%.o=%.d)" "$<"
	/usr/local/cuda-6.5/bin/nvcc -I/usr/local/include/opencv -G -g -O0 --compile --relocatable-device-code=true -gencode arch=compute_35,code=compute_35 -gencode arch=compute_35,code=sm_35  -x cu -o  "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


