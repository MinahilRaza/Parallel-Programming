//
// compile your program with
//    nvcc -O3 -arch=sm_70 --ptxas-options=-v -o galaxy filename -lm
//
// run your program with
//    srun -p gpu -c 1 --mem=10G ./galaxy RealGalaxies_100k_arcmin.dat SyntheticGalaxies_100k_arcmin.dat omega.out


// import libraries
#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

int    NoofReal;
int    NoofRand;
float *real_rasc, *real_decl;
float *rand_rasc, *rand_decl;

unsigned long long int *histogramDR, *histogramDD, *histogramRR;

long int CPUMemory = 0L;
long int GPUMemory = 0L;

int totaldegrees = 360;
int binsperdegree = 4;

// GPU kernel to calculate the histograms
__global__ void  fillHistogram(unsigned long long int *hist, float* rasc1, float* decl1, float* rasc2, float* decl2, int N, bool Skipcalc)
{

    // using 2 dimensions for 2 FOR loops
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int j = blockDim.y * blockIdx.y + threadIdx.y;

    if (i<N && j<N){
      float pif = acosf(-1.0f);
      if (Skipcalc){ // histogram DD and RR
	if (j>i){ //equivalent to starting histogram from j+1
	    float theta = sin(decl1[i])*sin(decl2[j])+ cos(decl1[i])*cos(decl2[j])*cos((rasc1[i]-rasc2[j]));
	    // acosf can only take values between -1 and 1
	    if (theta>1.0) theta = 1.0;
	    if (theta<-1.0) theta = -1.0;

            theta = acosf(theta)*180/pif;        
	    int ind = (int)(theta*4.0);
            atomicAdd(&hist[ind], 2);
	  
	}
        else if (j==i) atomicAdd(&hist[0], 1);
      
      }
     else // if the histogram is DR
     {
         float theta = sin(decl1[i])*sin(decl2[j])+ cos(decl1[i])*cos(decl2[j])*cos((rasc1[i]-rasc2[j]));
	 // acosf can only take values between -1 and 1
	 if (theta>1.0) theta = 1.0;
	 if (theta<-1.0) theta = -1.0;
         theta = acosf(theta)*180/pif;        
         int ind = (int)(theta*4.0);
         atomicAdd(&hist[ind], 1);
     }
        
	
    }
}


int main(int argc, char *argv[])
{
   int    readdata(char *argv1, char *argv2);
   unsigned long long int histogramDRsum, histogramDDsum, histogramRRsum;
   
   clock_t  start, end, walltime;
   int getDevice(void);

   FILE *outfil;

   if ( argc != 4 ) {printf("Usage: a.out real_data random_data output_data\n");return(-1);}
   
   start = clock();
   if ( readdata(argv[1], argv[2]) != 0 ) return(-1);

// For your entertainment: some performance parameters of the GPU you are running your programs on!
   if ( getDevice() != 0 ) return(-1);
   
   size_t arraybytes_h = totaldegrees * sizeof(unsigned long long int);
   histogramDR = (unsigned long long int *)malloc(arraybytes_h);
   histogramDD = (unsigned long long int *)malloc(arraybytes_h);
   histogramRR = (unsigned long long int *)malloc(arraybytes_h);

   for(int i =0; i < totaldegrees; i++){
	histogramDR[i] = 0LLU;
	histogramDD[i] = 0LLU;
	histogramRR[i] = 0LLU;
   }

   CPUMemory += 3L*(totaldegrees+1L)*sizeof(unsigned long long int);

   // input data is available in the arrays float real_rasc[], real_decl[], rand_rasc[], rand_decl[];
   // allocate memory on the GPU for input data and histograms

   unsigned long long int *histogramDR_gpu,*histogramDD_gpu, *histogramRR_gpu ;
   cudaMalloc(&histogramDR_gpu,arraybytes_h);
   cudaMalloc(&histogramDD_gpu,arraybytes_h);
   cudaMalloc(&histogramRR_gpu,arraybytes_h);
   GPUMemory += 3L*(totaldegrees+1L)*sizeof(unsigned long long int);

   size_t arraybytes_1 = (NoofReal) * sizeof(float);
   size_t arraybytes_2 = (NoofRand) * sizeof(float);
   float * real_rasc_gpu,* real_decl_gpu, * rand_rasc_gpu,* rand_decl_gpu;
   cudaMalloc(&real_rasc_gpu, arraybytes_1);
   cudaMalloc(&real_decl_gpu, arraybytes_1);
   cudaMalloc(&rand_rasc_gpu, arraybytes_2);
   cudaMalloc(&rand_decl_gpu, arraybytes_2);
   GPUMemory += 2L*(NoofReal*sizeof(float));
   GPUMemory += 2L*(NoofRand*sizeof(float));

   // and initialize the data on GPU by copying the real and rand data to the GPU
   cudaMemcpy(histogramDR_gpu, histogramDR, arraybytes_h, cudaMemcpyHostToDevice);
   cudaMemcpy(histogramDD_gpu, histogramDD, arraybytes_h, cudaMemcpyHostToDevice);
   cudaMemcpy(histogramRR_gpu, histogramRR, arraybytes_h, cudaMemcpyHostToDevice);
   cudaMemcpy(real_rasc_gpu, real_rasc, arraybytes_1, cudaMemcpyHostToDevice);
   cudaMemcpy(real_decl_gpu, real_decl, arraybytes_1, cudaMemcpyHostToDevice);
   cudaMemcpy(rand_rasc_gpu, rand_rasc, arraybytes_2, cudaMemcpyHostToDevice);
   cudaMemcpy(rand_decl_gpu, rand_decl, arraybytes_2, cudaMemcpyHostToDevice);
   
   // call the GPU kernel(s) that fill the three histograms
   // using 2D grids as it is easier to map the problem in 2D
   dim3 blocksInGrid(3125,3125); 
   dim3 threadsInBlock(32,32);

   fillHistogram<<<blocksInGrid, threadsInBlock>>>(histogramDR_gpu, real_rasc_gpu, real_decl_gpu, rand_rasc_gpu, rand_decl_gpu , NoofReal, 0);
   fillHistogram<<<blocksInGrid, threadsInBlock>>>(histogramDD_gpu, real_rasc_gpu, real_decl_gpu, real_rasc_gpu, real_decl_gpu, NoofReal, 1);
   fillHistogram<<<blocksInGrid, threadsInBlock>>>(histogramRR_gpu, rand_rasc_gpu, rand_decl_gpu, rand_rasc_gpu, rand_decl_gpu, NoofRand, 1);
   cudaDeviceSynchronize();
   
   // copy the histograms back to the CPU memory
   cudaMemcpy(histogramDR, histogramDR_gpu, arraybytes_h, cudaMemcpyDeviceToHost);
   cudaMemcpy(histogramDD, histogramDD_gpu, arraybytes_h, cudaMemcpyDeviceToHost);
   cudaMemcpy(histogramRR, histogramRR_gpu, arraybytes_h, cudaMemcpyDeviceToHost);
   
   // Free cuda memory
   cudaFree(histogramDR_gpu);
   cudaFree(histogramRR_gpu);
   cudaFree(histogramDD_gpu);
   
   cudaFree(real_rasc_gpu);
   cudaFree(real_decl_gpu);
   cudaFree(rand_rasc_gpu);
   cudaFree(rand_decl_gpu);

// checking to see if your histograms have the right number of entries
   histogramDRsum = 0LLU;
   for ( int i = 0; i < totaldegrees;++i )
    histogramDRsum += histogramDR[i];
   printf("   DR histogram sum = %lld\n",histogramDRsum);
   if ( histogramDRsum != 10000000000LLU ) {printf("   Incorrect histogram sum, exiting..\n");return(0);}
                   
   histogramDDsum = 0LLU;
   for ( int i = 0; i < totaldegrees;++i )
        histogramDDsum += histogramDD[i];
   printf("   DD histogram sum = %lld\n",histogramDDsum);
   if ( histogramDDsum != 10000000000LLU ) {printf("   Incorrect histogram sum, exiting..\n");return(0);}

   histogramRRsum = 0LLU;
   for ( int i = 0; i < totaldegrees;++i )
        histogramRRsum += histogramRR[i];
   printf("   RR histogram sum = %lld\n",histogramRRsum);
   if ( histogramRRsum != 10000000000LLU ) {printf("   Incorrect histogram sum, exiting..\n");return(0);}


   printf("   Omega values:");

   outfil = fopen(argv[3],"w");
   if ( outfil == NULL ) {printf("Cannot open output file %s\n",argv[3]);return(-1);}
   fprintf(outfil,"bin start\tomega\t        hist_DD\t        hist_DR\t        hist_RR\n");
   for ( int i = 0; i < totaldegrees; ++i )
       {
       if ( histogramRR[i] > 0 )
          {
          double omega =  (histogramDD[i]-2*histogramDR[i]+histogramRR[i])/((double)(histogramRR[i]));

          fprintf(outfil,"%6.4f\t%15lf\t%15lld\t%15lld\t%15lld\n",((float)i)/binsperdegree, omega,
             histogramDD[i], histogramDR[i], histogramRR[i]);
          if ( i < 5 ) printf("   %6.4lf",omega);
          }
       else
          if ( i < 5 ) printf("         ");
       }

   printf("\n");

   fclose(outfil);

   printf("   Results written to file %s\n",argv[3]);
   printf("   CPU memory allocated  = %.2lf MB\n",CPUMemory/1000000.0);
   printf("   GPU memory allocated  = %.2lf MB\n",GPUMemory/1000000.0);

   end = clock();
   walltime = end - start;
   printf("   Total wall clock time = %.2lf s\n", ((float)walltime) / CLOCKS_PER_SEC);

   return(0);
}

int readdata(char *argv1, char *argv2)
{
  int    i,linecount;
  char   inbuf[80];
  double ra, dec, dpi;
  FILE  *infil;

  printf("   Assuming data is in arc minutes!\n");
                          // phi   = ra/60.0 * dpi/180.0;
                          // theta = (90.0-dec/60.0)*dpi/180.0;
                          // otherwise use
                          // phi   = ra * dpi/180.0;
                          // theta = (90.0-dec)*dpi/180.0;

  dpi = acos(-1.0);
  infil = fopen(argv1,"r");
  if ( infil == NULL ) {printf("Cannot open input file %s\n",argv1);return(-1);}

  linecount =0;
  while ( fgets(inbuf,80,infil) != NULL ) ++linecount;
  rewind(infil);

  printf("   %s contains %d galaxies\n",argv1, linecount-1);

  NoofReal = linecount-1;

  if ( NoofReal != 100000 ) {printf("Incorrect number of galaxies\n");return(1);}

  real_rasc = (float *)calloc(NoofReal,sizeof(float));
  real_decl = (float *)calloc(NoofReal,sizeof(float));
  CPUMemory += 2L*NoofReal*sizeof(float);

  fgets(inbuf,80,infil);
  sscanf(inbuf,"%d",&linecount);
  if ( linecount != 100000 ) {printf("Incorrect number of galaxies\n");return(1);}

  i = 0;
  while ( fgets(inbuf,80,infil) != NULL )
      {
      if ( sscanf(inbuf,"%lf %lf",&ra,&dec) != 2 )
         {
         printf("   Cannot read line %d in %s\n",i+1,argv1);
         fclose(infil);
         return(-1);
         }
      real_rasc[i] = (float)( ra/60.0*dpi/180.0);
      real_decl[i] = (float)(dec/60.0*dpi/180.0);
      ++i;
      }

  fclose(infil);

  if ( i != NoofReal )
      {
      printf("   Cannot read %s correctly\n",argv1);
      return(-1);
      }

  infil = fopen(argv2,"r");
  if ( infil == NULL ) {printf("Cannot open input file %s\n",argv2);return(-1);}

  linecount =0;
  while ( fgets(inbuf,80,infil) != NULL ) ++linecount;
  rewind(infil);

  printf("   %s contains %d galaxies\n",argv2, linecount-1);

  NoofRand = linecount-1;
  if ( NoofRand != 100000 ) {printf("Incorrect number of random galaxies\n");return(1);}

  rand_rasc = (float *)calloc(NoofRand,sizeof(float));
  rand_decl = (float *)calloc(NoofRand,sizeof(float));
  CPUMemory += 2L*NoofRand*sizeof(float);

  fgets(inbuf,80,infil);
  sscanf(inbuf,"%d",&linecount);
  if ( linecount != 100000 ) {printf("Incorrect number of random galaxies\n");return(1);}

  i =0;
  while ( fgets(inbuf,80,infil) != NULL )
      {
      if ( sscanf(inbuf,"%lf %lf",&ra,&dec) != 2 )
         {
         printf("   Cannot read line %d in %s\n",i+1,argv2);
         fclose(infil);
         return(-1);
         }
      rand_rasc[i] = (float)( ra/60.0*dpi/180.0);
      rand_decl[i] = (float)(dec/60.0*dpi/180.0);
      ++i;
      }

  fclose(infil);

  if ( i != NoofReal )
      {
      printf("   Cannot read %s correctly\n",argv2);
      return(-1);
      }

  return(0);
}




int getDevice(void)
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

    cudaSetDevice(0);
    cudaGetDevice(&device);
    if ( device != 0 ) printf("   Unable to set device 0, using %d instead",device);
    else printf("   Using CUDA device %d\n\n", device);

return(0);
}

