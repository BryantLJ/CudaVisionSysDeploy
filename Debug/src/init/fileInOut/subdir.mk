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
	/usr/local/cuda-7.5/bin/nvcc -I/usr/local/include/opencv -I/usr/local/include/cuda-7.5/targets/x86_64-linux/include -G -g -O0 -std=c++11 -gencode arch=compute_52,code=sm_52  -odir "src/init/fileInOut" -M -o "$(@:%.o=%.d)" "$<"
	/usr/local/cuda-7.5/bin/nvcc -I/usr/local/include/opencv -I/usr/local/include/cuda-7.5/targets/x86_64-linux/include -G -g -O0 -std=c++11 --compile  -x c++ -o  "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


