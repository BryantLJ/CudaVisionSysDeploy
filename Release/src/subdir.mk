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
	/usr/local/cuda-6.5/bin/nvcc -I/usr/local/include/opencv -O3 -gencode arch=compute_35,code=sm_35  -odir "src" -M -o "$(@:%.o=%.d)" "$<"
	/usr/local/cuda-6.5/bin/nvcc -I/usr/local/include/opencv -O3 --compile --relocatable-device-code=true -gencode arch=compute_35,code=compute_35 -gencode arch=compute_35,code=sm_35  -x cu -o  "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


