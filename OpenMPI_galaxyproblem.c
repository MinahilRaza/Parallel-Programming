#include <mpi.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>

float *real_rasc, *real_decl, *rand_rasc, *rand_decl;
float  pif;
long int MemoryAllocatedCPU = 0L;

int main(int argc, char* argv[])
    {
        int NumberofGalaxies = 100000;
        int id, ntasks;
        int parseargs_readinput(int argc, char *argv[]);
        real_rasc        = (float *)calloc(100000L, sizeof(float));
        real_decl        = (float *)calloc(100000L, sizeof(float));
        rand_rasc        = (float *)calloc(100000L, sizeof(float));
        rand_decl        = (float *)calloc(100000L, sizeof(float));
        pif = acosf(-1.0f); //value of pi is needed by all processes

        MemoryAllocatedCPU += 10L*100000L*sizeof(float);

        /*Initialize MPI */
        if ( MPI_Init(&argc, &argv) != MPI_SUCCESS )
             {printf("MPI_init failed!\n"); exit(1); }
        /* Get number of tasks */
        if ( MPI_Comm_size(MPI_COMM_WORLD, &ntasks) != MPI_SUCCESS)
            { printf("MPI_Comm_size failed!\n"); exit(1);}
        /* Get id of this process */
        if ( MPI_Comm_rank(MPI_COMM_WORLD, &id) != MPI_SUCCESS)
            { printf("MPI_Comm_rank failed!\n");exit(1);}

        int r = ceil((float) NumberofGalaxies / ntasks);
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

        int start = id*r;
        int end = start + r;
        if (end >100000) end = 100000;

        int i,j,k;
        float theta1, theta2, theta3;
        int ind1, ind2, ind3;

        long int histogram_DD[360] = {0L};
        long int histogram_DR[360] = {0L};
        long int histogram_RR[360] = {0L};
        MemoryAllocatedCPU += 3L*360L*sizeof(long int);

        long int histogram_DD_final[360] = {0L};
        long int histogram_DR_final[360] = {0L};
        long int histogram_RR_final[360] = {0L};
        MemoryAllocatedCPU += 3L*360L*sizeof(long int);

        for (i = start; i < end; ++i )
        {
            for (j = i+1; j < 100000; ++j ){
                theta1 = acosf(sin(real_decl[i])*sin(real_decl[j])+ cos(real_decl[i])*cos(real_decl[j])*cos((real_rasc[i]-real_rasc[j])))*180/pif;
                theta2 = acosf(sin(rand_decl[i])*sin(rand_decl[j])+ cos(rand_decl[i])*cos(rand_decl[j])*cos((rand_rasc[i]-rand_rasc[j])))*180/pif;
                // calculate the histogram indices
                ind1 = (int)(theta1*4.0);
                ind2 = (int)(theta2*4.0);
                histogram_DD[ind1]+=2;
                histogram_RR[ind2]+=2;
            }
            for (k = 0; k <100000; ++k){
                theta3 = acosf(sin(rand_decl[i])*sin(real_decl[k])+ cos(rand_decl[i])*cos(real_decl[k])*cos((rand_rasc[i]-real_rasc[k])))*180/pif;
                ind3 = (int)(theta3*4.0);
                histogram_DR[ind3]++;
            }
        }

        if (MPI_Reduce(histogram_DD, histogram_DD_final, 360, MPI_LONG, MPI_SUM, 0, MPI_COMM_WORLD)!= MPI_SUCCESS)
        {
            printf("Error in MPI_Reduce\n");
            exit(1);
        }
        if (MPI_Reduce(histogram_RR, histogram_RR_final, 360, MPI_LONG, MPI_SUM, 0, MPI_COMM_WORLD)!= MPI_SUCCESS)
        {
            printf("Error in MPI_Reduce\n");
            exit(1);
        }
        if (MPI_Reduce(histogram_DR, histogram_DR_final, 360, MPI_LONG, MPI_SUM, 0, MPI_COMM_WORLD)!= MPI_SUCCESS)
        {
            printf("Error in MPI_Reduce\n");
            exit(1);
        }	
        if (id == 0){
            // check point: the sum of all historgram entries should be 10 000 000 000
            histogram_DD_final[0] +=100000;
            histogram_RR_final[0] +=100000;
            long int histsum = 0L;
            int      correct_value=1;
            for ( int i = 0; i < 360; ++i ) histsum += histogram_DD_final[i];
            printf("   Histogram DD : sum = %ld\n",histsum);
            if ( histsum != 10000000000L ) correct_value = 0;

            histsum = 0L;
            for ( int i = 0; i < 360; ++i ) histsum += histogram_DR_final[i];
            printf("   Histogram DR : sum = %ld\n",histsum);
            if ( histsum != 10000000000L ) correct_value = 0;

            histsum = 0L;
            for ( int i = 0; i < 360; ++i ) histsum += histogram_RR_final[i];
            printf("   Histogram RR : sum = %ld\n",histsum);
            if ( histsum != 10000000000L ) correct_value = 0;

            if ( correct_value != 1 )
               {printf("   Histogram sums should be 10000000000. Ending program prematurely\n");return(0);}

            printf("   Omega values for the histograms:\n");
            float omega[360];
            for ( int i = 0; i < 10; ++i )
                if ( histogram_RR_final[i] != 0L )
                {
                   omega[i] = (histogram_DD_final[i] - 2L*histogram_DR_final[i] + histogram_RR_final[i])/((float)(histogram_RR_final[i]));
                   if ( i < 10 ) printf("      angle %.2f deg. -> %.2f deg. : %.4f\n", i*0.25, (i+1)*0.25, omega[i]);
                }

            FILE *out_file = fopen(argv[3],"w");
            if ( out_file == NULL ) printf("   ERROR: Cannot open output file %s\n",argv[3]);
            else
               {
               for ( int i = 0; i < 360; ++i )
                   if ( histogram_RR_final[i] != 0L )
                      fprintf(out_file,"%.2f  : %.3f\n", i*0.25, omega[i] );
               fclose(out_file);
               printf("   Omega values written to file %s\n",argv[3]);
            }

            printf("   Total memory allocated = %.1lf MB\n",MemoryAllocatedCPU/1000000.0);
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



