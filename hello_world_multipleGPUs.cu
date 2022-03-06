// compile your program with
//    nvcc -O3 -arch=sm_70 --ptxas-options=-v -o galaxy filename -lm
//
// run your program with
//    srun -p gpu -c 1 --mem=10G ./galaxy RealGalaxies_100k_arcmin.dat SyntheticGalaxies_100k_arcmin.dat omega.out


// import libraries
#include <cuda.h>
#include <mpi.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>


int main(int argc, char *argv[])
{
   int id, ntasks;
   /*Initialize MPI */
        if ( MPI_Init(&argc, &argv) != MPI_SUCCESS )
             {printf("MPI_init failed!\n"); exit(1); }
        /* Get number of tasks */
        if ( MPI_Comm_size(MPI_COMM_WORLD, &ntasks) != MPI_SUCCESS)
            { printf("MPI_Comm_size failed!\n"); exit(1);}
        /* Get id of this process */
        if ( MPI_Comm_rank(MPI_COMM_WORLD, &id) != MPI_SUCCESS)
            { printf("MPI_Comm_rank failed!\n");exit(1);}
   int getDevice(int id);
   printf("My ID is %d \n", id);

   getDevice(id);
   MPI_Finalize();
   return(0);
}




int getDevice(int id)
{
	
  int deviceCount;
  cudaGetDeviceCount(&deviceCount);
  printf("   Found %d CUDA devices\n",deviceCount);
  if ( deviceCount < 0 || deviceCount > 128 ) return(-1);
  int device;
  for (device = 0; device < deviceCount; ++device) {
       cudaDeviceProp deviceProp;
       cudaGetDeviceProperties(&deviceProp, device);
       printf("      Device %s                  device %d\n", deviceProp.name,device);
       printf("         compute capability           =         %d.%d\n", deviceProp.major, deviceProp.minor);
       printf("         totalGlobalMemory            =        %.2lf GB\n", deviceProp.totalGlobalMem/1000000000.0);
       printf("         l2CacheSize                  =    %8d B\n", deviceProp.l2CacheSize);
       printf("         regsPerBlock                 =    %8d\n", deviceProp.regsPerBlock);
       printf("         multiProcessorCount          =    %8d\n", deviceProp.multiProcessorCount);
       printf("         maxThreadsPerMultiprocessor  =    %8d\n", deviceProp.maxThreadsPerMultiProcessor);
       printf("         sharedMemPerBlock            =    %8d B\n", (int)deviceProp.sharedMemPerBlock);
       printf("         warpSize                     =    %8d\n", deviceProp.warpSize);
       printf("         clockRate                    =    %8.2lf MHz\n", deviceProp.clockRate/1000.0);
       printf("         maxThreadsPerBlock           =    %8d\n", deviceProp.maxThreadsPerBlock);
       printf("         asyncEngineCount             =    %8d\n", deviceProp.asyncEngineCount);
       printf("         f to lf performance ratio    =    %8d\n", deviceProp.singleToDoublePrecisionPerfRatio);
       printf("         maxGridSize                  =    %d x %d x %d\n",
                          deviceProp.maxGridSize[0], deviceProp.maxGridSize[1], deviceProp.maxGridSize[2]);
       printf("         maxThreadsDim                =    %d x %d x %d\n",
                          deviceProp.maxThreadsDim[0], deviceProp.maxThreadsDim[1], deviceProp.maxThreadsDim[2]);
       printf("         concurrentKernels            =    ");
       if(deviceProp.concurrentKernels==1) printf("     yes\n"); else printf("    no\n");
       printf("         deviceOverlap                =    %8d\n", deviceProp.deviceOverlap);
       if(deviceProp.deviceOverlap == 1)
       printf("            Concurrently copy memory/execute kernel\n");
       }

    cudaSetDevice(id);
    cudaGetDevice(&device);
    if ( device != id ) printf("   Unable to set device 0, using %d instead",device);
    else printf("   Using CUDA device %d\n\n", device);

return(0);
}

