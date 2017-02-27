#######################################################################
#                              Makefile                               #
#######################################################################

#######################################################################
#                              COMPILERS                              #
#######################################################################
CXX=g++
CUDAC=nvcc
#######################################################################
#                      Handling input arguments                       #
#######################################################################


ifneq (,$(findstring debug, $(MAKETARGET)))
	AUX+=-DDEBUG
endif
ifneq (,$(findstring timing, $(MAKETARGET)))
	AUX+=-DTIMING
endif
ifneq (,$(findstring gf580, $(MAKETARGET)))
	AUX+=-DCUDA -DGF580
	CUDAARCH=sm_21
endif
ifneq (,$(findstring gf680, $(MAKETARGET)))
	AUX+=-DCUDA -DGF680
	CUDAARCH=sm_30
endif
ifneq (,$(findstring DCUDA, $(AUX)))
	CUDAOBJECTS=cudaGenerate.o cudaReferenceDistance.o ccuda.o cudaFindMin.o
endif


CXXFLAGS= -O3 -Wall #-msse -msse3 -m3dnow -msse2 -march=native -O3
AUX+=
PROF=

OBJECTS= MieAB.o MiePT.o ReferenceDistance.o GeneratePattern.o auxillary.o Calculate_m.o RunningRadius.o main.o mainFunction.o
OBJECTSCLIENT = client.o
CUDA_FLG= -arch=$(CUDAARCH) --ptxas-options=-v -O3 -m64 --compiler-options '-Wall -Wconversion -O3'# --maxrregcount 48 -Xptxas -dlcm=cg 
C_FLG=-fopenmp -Wno-unused  -march=native -msse2 -msse -msse3 -m3dnow -O3


.DEFAULT:
ifneq (,$(findstring mrfin,$(MAKECMDGOALS)))
	${MAKE} MAKETARGET=$(MAKECMDGOALS) mrfin
endif
ifneq (,$(findstring client, $(MAKECMDGOALS)))
	${MAKE} client
endif


mrfin: $(OBJECTS) $(CUDAOBJECTS)
	$(CUDAC) -Xcompiler "-fopenmp"  $(OBJECTS) $(CUDAOBJECTS) -o $@ 

client: $(OBJECTSCLIENT)
	$(CXX) -fopenmp $(OBJECTSCLIENT) -o $@

%.o: %.cpp %.h
	$(CXX) -c $(C_FLG) $(AUX) $< -o $@
	
%.o: %.cu %.h
	$(CUDAC) -c $(CUDA_FLG) $(AUX) $< -o $@
clean:
	rm -f *.o
	rm -f mrfin
	rm -f client
	rm -f .socket

#######################################################################
#                               LEGACY                                #
#######################################################################
#CXX=icpc
#OBJECTS= MieAB.o MiePT.o ReferenceDistance.o GeneratePattern.o auxillary.o Calculate_m.o RunningRadius.o main.o cudaGenerate.o cudaReferenceDistance.o ccuda.o mainFunction.o sorting.o
#CCUDA_FLG=-m64 -Xcompiler "-fopenmp -Wno-unused  -march=native -msse2 -msse -msse3 -m3dnow -O3"

