# CudaVisionSysDeploy
GPU accelerated pedestrian detector using a Pyramidal-Sliding Window approach with HOGLBP as feature descriptor and SVM as the classifier

Software developed by Víctor Campmany (vcampmany@cvc.uab.es)
Computer Vision Center and Computer Architecture Department, Universitat Autonoma de Barcelona

### How to compile it
Add your own library paths to the Makefile (instruction in the Makefile)
```
cd into_repository_path
make all
```

### How to run it
```
./bin/visionSys
```
Edit the parameters.ini file in order to modify the parameters as desired

### Requirements
* OpenCV >= 2.4
* CUDA toolkit
* NVIDIA GPU >= Kepler 3.0

### Related publications
[GPU-based Pedestrian Detection for Autonomous Driving](http://www.sciencedirect.com/science/article/pii/S1877050916309395). V. Campmany, S. Silva, A. Espinosa, JC. Moure, D. Vazquez, A. Lopez.
ICCS 2016, Procedia Computer Science 80, 2377-2381
