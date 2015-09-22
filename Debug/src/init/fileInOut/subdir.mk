################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CPP_SRCS += \
../src/init/fileInOut/IniFileIO.cpp \
../src/init/fileInOut/Utils.cpp 

OBJS += \
./src/init/fileInOut/IniFileIO.o \
./src/init/fileInOut/Utils.o 

CPP_DEPS += \
./src/init/fileInOut/IniFileIO.d \
./src/init/fileInOut/Utils.d 


# Each subdirectory must supply rules for building sources it contributes
src/init/fileInOut/%.o: ../src/init/fileInOut/%.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: NVCC Compiler'
	/usr/local/cuda-6.5/bin/nvcc -I/usr/local/include/opencv -G -g -O0 -gencode arch=compute_35,code=sm_35  -odir "src/init/fileInOut" -M -o "$(@:%.o=%.d)" "$<"
	/usr/local/cuda-6.5/bin/nvcc -I/usr/local/include/opencv -G -g -O0 --compile  -x c++ -o  "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


