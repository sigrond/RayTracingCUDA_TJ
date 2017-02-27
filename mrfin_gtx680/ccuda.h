void freeCudaPointer(void ** pointer);
void allocCudaPointer(void ** pointer, size_t size);
void freeCudaMemory();
void freeCudaRefMemory();
//void freeCudaPointerInt(int ** pointer);
//void allocCudaPointerInt(int ** pointer, size_t size);

void cudaFinalize();
void cuda1stPolarizationSync();
void freeCudaMemoryMin();
void mallocCudaReferences(int i, int const mPatterns, int const nPatterns, int const mReferences, int const nReferences ) ;
