CXXFLAGS= -O3 -Wall #-msse -msse3 -m3dnow -msse2 -march=native -O3
#CXX=icpc
CXX=g++
AUX=-DTIMING -DCUDA
PROF=
#OBJECTS= MieAB.o MiePT.o ReferenceDistance.o GeneratePattern.o auxillary.o Calculate_m.o RunningRadius.o main.o cudaGenerate.o cudaReferenceDistance.o ccuda.o mainFunction.o sorting.o

OBJECTS= MieAB.o MiePT.o ReferenceDistance.o GeneratePattern.o auxillary.o Calculate_m.o RunningRadius.o main.o cudaGenerate.o cudaReferenceDistance.o ccuda.o mainFunction.o
OBJECTSCLIENT = client.o
CUDAC=nvcc
CUDA_FLG= --ptxas-options=-v -arch=sm_21 -O3 -m64 --compiler-options '-Wall -Wconversion -O3'# --maxrregcount 48 -Xptxas -dlcm=cg 
CCUDA_FLG=-m64 -Xcompiler "-fopenmp -Wno-unused  -march=native -msse2 -msse -msse3 -m3dnow -O3"

mrfin: $(OBJECTS)
	$(CUDAC) -Xcompiler "-fopenmp"  $(OBJECTS) -o $@ 

client: $(OBJECTSCLIENT)
	$(CUDAC) -Xcompiler "-fopenmp" $(OBJECTSCLIENT) -o $@

%.o: %.cpp %.h
	$(CUDAC) -c $(CCUDA_FLG) $(AUX) $< -o $@
	
%.o: %.cu %.h
	$(CUDAC) -c $(CUDA_FLG) $(AUX) $< -o $@
clean:
	rm -f *.o
	rm -f mrfin

