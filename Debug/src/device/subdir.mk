################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CU_SRCS += \
../src/device/resize.cu 

CU_DEPS += \
./src/device/resize.d 

OBJS += \
./src/device/resize.o 


# Each subdirectory must supply rules for building sources it contributes
src/device/%.o: ../src/device/%.cu
	@echo 'Building file: $<'
	@echo 'Invoking: NVCC Compiler'
	/usr/local/cuda-6.5/bin/nvcc -I/usr/local/include/opencv -G -g -O0 -gencode arch=compute_35,code=sm_35  -odir "src/device" -M -o "$(@:%.o=%.d)" "$<"
	/usr/local/cuda-6.5/bin/nvcc -I/usr/local/include/opencv -G -g -O0 --compile --relocatable-device-code=true -gencode arch=compute_35,code=compute_35 -gencode arch=compute_35,code=sm_35  -x cu -o  "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


