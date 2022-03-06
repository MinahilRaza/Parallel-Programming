#include <mpi.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>

float *real_rasc, *real_decl, *rand_rasc, *rand_decl;
float  pif;
long int CPUMemory = 0L;
long int GPUMemory = 0L;

int totaldegrees = 360;
int binsperdegree = 4;
int    NoofReal= 100000;
int    NoofRand= 100000;

unsigned long long int *histogramDR, *histogramDD, *histogramRR;

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

int main(int argc, char* argv[])
    {
        int parseargs_readinput(int argc, char *argv[]);
        int id, ntasks;
        
        real_rasc        = (float *)calloc(100000L, sizeof(float));
        real_decl        = (float *)calloc(100000L, sizeof(float));
        rand_rasc        = (float *)calloc(100000L, sizeof(float));
        rand_decl        = (float *)calloc(100000L, sizeof(float));
        pif = acosf(-1.0f); //value of pi is needed by all processes

        CPUMemory += 10L*100000L*sizeof(float);

        /*Initialize MPI */
        if ( MPI_Init(&argc, &argv) != MPI_SUCCESS )
             {printf("MPI_init failed!\n"); exit(1); }
        /* Get number of tasks */
        if ( MPI_Comm_size(MPI_COMM_WORLD, &ntasks) != MPI_SUCCESS)
            { printf("MPI_Comm_size failed!\n"); exit(1);}
        /* Get id of this process */
        if ( MPI_Comm_rank(MPI_COMM_WORLD, &id) != MPI_SUCCESS)
            { printf("MPI_Comm_rank failed!\n");exit(1);}

        //int r = ceil((float) NumberofGalaxies / ntasks);
        double start_time, time_taken;
	
        if(id == 0 ){
            start_time = MPI_Wtime();

            // read input data from files given on the command line
            if ( parseargs_readinput(argc, argv) != 0 ) {printf("   Program stopped.\n");return(0);}
            printf("   ID %d: Input data read, now broadcasting it to other processes\n", id);
        }
            if (MPI_Bcast(real_rasc, 100000, MPI_FLOAT, 0, MPI_COMM_WORLD)!= MPI_SUCCESS)
            {
                printf("Error in MPI_Bcast\n");
                exit(1);
            }
            if (MPI_Bcast(real_decl, 100000, MPI_FLOAT, 0, MPI_COMM_WORLD)!= MPI_SUCCESS)
            {
                printf("Error in MPI_Bcast\n");
                exit(1);
            }
            if (MPI_Bcast(rand_rasc, 100000, MPI_FLOAT, 0, MPI_COMM_WORLD)!= MPI_SUCCESS)
            {
                printf("Error in MPI_Bcast\n");
                exit(1);
            }
            if (MPI_Bcast(rand_decl, 100000, MPI_FLOAT, 0, MPI_COMM_WORLD)!= MPI_SUCCESS)
            {
                printf("Error in MPI_Bcast\n");
                exit(1);
            }

        //int start = id*r;
        //int end = start + r;
        //if (end >100000) end = 100000;

        /*int i,j,k;
        float theta1, theta2, theta3;
        int ind1, ind2, ind3;
	*/
       	size_t arraybytes_h = totaldegrees * sizeof(unsigned long long int);
   	histogramDR = (unsigned long long int *)malloc(arraybytes_h);
   	histogramDD = (unsigned long long int *)malloc(arraybytes_h);
   	histogramRR = (unsigned long long int *)malloc(arraybytes_h);

   	for(int i =0; i < totaldegrees; i++){
		histogramDR[i] = 0LLU;
		histogramDD[i] = 0LLU;
		histogramRR[i] = 0LLU;
   	}

	cudaSetDevice(id);

	size_t arraybytes_1 = (NoofReal) * sizeof(float);
   	size_t arraybytes_2 = (NoofRand) * sizeof(float);
   	float * real_rasc_gpu,* real_decl_gpu, * rand_rasc_gpu,* rand_decl_gpu;
   	cudaMalloc(&real_rasc_gpu, arraybytes_1);
   	cudaMalloc(&real_decl_gpu, arraybytes_1);
   	cudaMalloc(&rand_rasc_gpu, arraybytes_2);
   	cudaMalloc(&rand_decl_gpu, arraybytes_2);
   	GPUMemory += 2L*(NoofReal*sizeof(float));
   	GPUMemory += 2L*(NoofRand*sizeof(float));
   	CPUMemory += 3L*(totaldegrees+1L)*sizeof(unsigned long long int);
	dim3 blocksInGrid(3125,3125); 
   	dim3 threadsInBlock(32,32);
		
        if (id == 0){
		unsigned long long int *histogramDR_gpu;
		cudaMalloc(&histogramDR_gpu,arraybytes_h);
		cudaMemcpy(histogramDR_gpu, histogramDR, arraybytes_h, cudaMemcpyHostToDevice);
		fillHistogram<<<blocksInGrid, threadsInBlock>>>(histogramDR_gpu, real_rasc_gpu, real_decl_gpu, rand_rasc_gpu, rand_decl_gpu , NoofReal, 0);
		cudaMemcpy(histogramDR, histogramDR_gpu, arraybytes_h, cudaMemcpyDeviceToHost);
		cudaFree(histogramDR_gpu);
		printf("process 0: histogram computed");
	}
	if (id == 1){
		unsigned long long int *histogramDD_gpu;
		cudaMalloc(&histogramDD_gpu,arraybytes_h);
		cudaMemcpy(histogramDD_gpu, histogramDD, arraybytes_h, cudaMemcpyHostToDevice);
		fillHistogram<<<blocksInGrid, threadsInBlock>>>(histogramDD_gpu, real_rasc_gpu, real_decl_gpu, real_rasc_gpu, real_decl_gpu , NoofReal, 1);
		cudaMemcpy(histogramDD, histogramDD_gpu, arraybytes_h, cudaMemcpyDeviceToHost);
		cudaFree(histogramDD_gpu);
		 printf("process 1: histogram computed");

	}
	if (id == 2){
		unsigned long long int *histogramRR_gpu;
		cudaMalloc(&histogramRR_gpu,arraybytes_h);
		cudaMemcpy(histogramRR_gpu, histogramRR, arraybytes_h, cudaMemcpyHostToDevice);
		fillHistogram<<<blocksInGrid, threadsInBlock>>>(histogramRR_gpu, rand_rasc_gpu, rand_decl_gpu, rand_rasc_gpu, rand_decl_gpu , NoofRand, 1);
		cudaMemcpy(histogramRR, histogramRR_gpu, arraybytes_h, cudaMemcpyDeviceToHost);
		cudaFree(histogramRR_gpu);
		 printf("process 2: histogram computed");

	}
	cudaFree(real_rasc_gpu);
   	cudaFree(real_decl_gpu);
   	cudaFree(rand_rasc_gpu);
   	cudaFree(rand_decl_gpu);
	printf("Succesfully computed histograms\n");
	

	unsigned long int histogramDD_final[360] = {0L};
        unsigned long int histogramDR_final[360] = {0L};
        unsigned long int histogramRR_final[360] = {0L};
        CPUMemory += 3L*360L*sizeof(unsigned long long int);
	
	// ensuring that all histograms have been computed prior to copying
	MPI_Barrier(MPI_COMM_WORLD);

        if (MPI_Reduce(histogramDD, histogramDD_final, 360, MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD)!= MPI_SUCCESS)
        {
            printf("Error in MPI_Reduce\n");
            exit(1);
        }
        if (MPI_Reduce(histogramRR, histogramRR_final, 360, MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD)!= MPI_SUCCESS)
        {
            printf("Error in MPI_Reduce\n");
            exit(1);
        }
        if (MPI_Reduce(histogramDR, histogramDR_final, 360, MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD)!= MPI_SUCCESS)
        {
            printf("Error in MPI_Reduce\n");
            exit(1);
        }	
        if (id == 0){
            // check point: the sum of all historgram entries should be 10 000 000 000
            //histogramDD[0] +=100000;
            //histogramRR[0] +=100000;
            long int histsum = 0L;
            int      correct_value=1;
            for ( int i = 0; i < 360; ++i ) histsum += histogramDD_final[i];
            printf("   Histogram DD : sum = %ld\n",histsum);
            if ( histsum != 10000000000L ) correct_value = 0;

            histsum = 0L;
            for ( int i = 0; i < 360; ++i ) histsum += histogramDR_final[i];
            printf("   Histogram DR : sum = %ld\n",histsum);
            if ( histsum != 10000000000L ) correct_value = 0;

            histsum = 0L;
            for ( int i = 0; i < 360; ++i ) histsum += histogramRR_final[i];
            printf("   Histogram RR : sum = %ld\n",histsum);
            if ( histsum != 10000000000L ) correct_value = 0;

            if ( correct_value != 1 )
               {printf("   Histogram sums should be 10000000000. Ending program prematurely\n");return(0);}

            printf("   Omega values for the histograms:\n");
            float omega[360];
            for ( int i = 0; i < 10; ++i )
                if ( histogramRR_final[i] != 0L )
                {
                   omega[i] = (histogramDD_final[i] - 2L*histogramDR_final[i] + histogramRR_final[i])/((float)(histogramRR_final[i]));
                   if ( i < 10 ) printf("      angle %.2f deg. -> %.2f deg. : %.4f\n", i*0.25, (i+1)*0.25, omega[i]);
                }

            FILE *out_file = fopen(argv[3],"w");
            if ( out_file == NULL ) printf("   ERROR: Cannot open output file %s\n",argv[3]);
            else
               {
               for ( int i = 0; i < 360; ++i )
                   if ( histogramRR_final[i] != 0L )
                      fprintf(out_file,"%.2f  : %.3f\n", i*0.25, omega[i] );
               fclose(out_file);
               printf("   Omega values written to file %s\n",argv[3]);
            }

            printf("   Total memory allocated = %.1lf MB\n",CPUMemory/1000000.0);
            time_taken = MPI_Wtime() - start_time;
            printf("Execution time of the program is %f seconds\n", time_taken);
            }
            free(real_rasc); free(real_decl);
            free(rand_rasc); free(rand_decl);
            MPI_Finalize();

    }

int parseargs_readinput(int argc, char *argv[])
    {
    // this function reads data from the files
    // converts angle to radian
    // it also creates a pointer to output file
    FILE *real_data_file, *rand_data_file, *out_file;
    float arcmin2rad = 1.0f/60.0f/180.0f*pif;
    int Number_of_Galaxies;

    if ( argc != 4 )
       {
       printf("   Usage: galaxy real_data random_data output_file\n   All MPI processes will be killed\n");
       return(1);
       }
    if ( argc == 4 )
       {
       printf("   Running galaxy_openmp %s %s %s\n",argv[1], argv[2], argv[3]);

       real_data_file = fopen(argv[1],"r");
       if ( real_data_file == NULL )
          {
          printf("   Usage: galaxy  real_data  random_data  output_file\n");
          printf("   ERROR: Cannot open real data file %s\n",argv[1]);
          return(1);
          }
       else
	  {
          fscanf(real_data_file,"%d",&Number_of_Galaxies);
          for ( int i = 0; i < 100000; ++i )
              {
      	      float rasc, decl;
	      if ( fscanf(real_data_file,"%f %f", &rasc, &decl ) != 2 )
	         {
                 printf("   ERROR: Cannot read line %d in real data file %s\n",i+1,argv[1]);
                 fclose(real_data_file);
	         return(1);
	         }
	      real_rasc[i] = rasc*arcmin2rad;
	      real_decl[i] = decl*arcmin2rad;
	      }
           fclose(real_data_file);
	   printf("   Successfully read 100000 lines from %s\n",argv[1]);
	   }

       rand_data_file = fopen(argv[2],"r");
       if ( rand_data_file == NULL )
          {
          printf("   Usage: galaxy  real_data  random_data  output_file\n");
          printf("   ERROR: Cannot open random data file %s\n",argv[2]);
          return(1);
          }
       else
	  {
          fscanf(rand_data_file,"%d",&Number_of_Galaxies);
          for ( int i = 0; i < 100000; ++i )
              {
      	      float rasc, decl;
	      if ( fscanf(rand_data_file,"%f %f", &rasc, &decl ) != 2 )
	         {
                 printf("   ERROR: Cannot read line %d in real data file %s\n",i+1,argv[2]);
                 fclose(rand_data_file);
	         return(1);
	         }
	      rand_rasc[i] = rasc*arcmin2rad;
	      rand_decl[i] = decl*arcmin2rad;
	      }
          fclose(rand_data_file);
	  printf("   Successfully read 100000 lines from %s\n",argv[2]);
	  }
       out_file = fopen(argv[3],"w");
       if ( out_file == NULL )
          {
          printf("   Usage: galaxy  real_data  random_data  output_file\n");
          printf("   ERROR: Cannot open output file %s\n",argv[3]);
          return(1);
          }
       else fclose(out_file);
       }

    return(0);
    }



