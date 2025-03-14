#include <mpi.h>
#include <cstdio>
#include <cmath>
#include <unistd.h>

#define CUDA_AWARE
// #define VERY_VERBOSE
// #define DEBUG_MODE
#define OFFSET 1

#define DATATYPE int

static void HandleError(cudaError_t err, const char *file, int line){
  if(err != cudaSuccess){
    printf("%s in %s at line %d\n", cudaGetErrorString(err), file, line);
    exit(EXIT_FAILURE);
  }
}
#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__))

__global__ void set_data(DATATYPE* q, int rank, int N){
   int i = blockDim.x * blockIdx.x + threadIdx.x;
   if(i < N){
      // q[i] = 1.0*rank + 1.0e-6*i;
      q[i] = 100000*rank + i;
#ifdef VERY_VERBOSE
      if(i<5){
         printf("rank %2d i=%4d q[i]=%d\n", rank, i, q[i]);
      }
#endif
   }
}

void go(int N) {

   int mpi_rank, mpi_size;
   cudaDeviceProp prop;
   int device,ngpu;
   DATATYPE *d_sdata, *d_rdata; // device pointers (gpu)
   DATATYPE *h_sdata, *h_rdata; // host pointers   (cpu)
   char hostname[32];

   gethostname(hostname, 32);

   MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
   MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);

   // useful message
   HANDLE_ERROR( cudaGetDeviceCount(&ngpu) );  
   HANDLE_ERROR( cudaSetDevice(mpi_rank%ngpu) );
   // HANDLE_ERROR( cudaSetDevice(0) );
   HANDLE_ERROR( cudaGetDevice(&device) );
   HANDLE_ERROR( cudaGetDeviceProperties(&prop, device) );

#ifdef DEBUG_MODE
   // printf("[%s] Rank %2d/%2d using gpu %1d/%1d : %s_%d\n",hostname, mpi_rank,mpi_size,device,ngpu,prop.name,prop.pciBusID);
   printf("[%s] Rank %2d/%2d using gpu %1d/%1d : %s_%08x\n",hostname, mpi_rank,mpi_size,device,ngpu,prop.name,prop.uuid);
#endif

   // initialize data
   HANDLE_ERROR( cudaMalloc((void**)&d_sdata, N*sizeof(DATATYPE)) );
   HANDLE_ERROR( cudaMalloc((void**)&d_rdata, N*sizeof(DATATYPE)) );
   h_sdata = new DATATYPE[N];
   h_rdata = new DATATYPE[N];

   dim3 threads(256,1,1);
   dim3 blocks(1,1,1);
   blocks.x = (N-1)/threads.x+1;

   set_data<<<blocks,threads>>>(d_sdata, mpi_rank, N);

   MPI_Request *reqs  = new MPI_Request[2];
   MPI_Status  *stats = new MPI_Status[2];

   int to   = (mpi_rank+OFFSET)%mpi_size;
   int from = (mpi_rank-OFFSET+mpi_size)%mpi_size;

   cudaDeviceSynchronize();

   //
   // Send/recv the data
   //
#ifdef CUDA_AWARE
   MPI_Isend(d_sdata, N*sizeof(DATATYPE), MPI_BYTE,   to, 111, MPI_COMM_WORLD, &reqs[0]);
   MPI_Irecv(d_rdata, N*sizeof(DATATYPE), MPI_BYTE, from, 111, MPI_COMM_WORLD, &reqs[1]);
#else
   HANDLE_ERROR( cudaMemcpy(h_sdata, d_sdata, N*sizeof(DATATYPE), cudaMemcpyDeviceToHost) );

#ifdef VERY_VERBOSE
   for(int rank=0; rank<mpi_size; rank++){
      if(rank==mpi_rank){
         for(int i=0; i<N; i++){
            printf("send %2d | %4d %d\n", rank, i, h_sdata[i]);
         }
      }
      MPI_Barrier(MPI_COMM_WORLD);
   }
#endif
   
   MPI_Isend(h_sdata, N*sizeof(DATATYPE), MPI_BYTE,   to, 111, MPI_COMM_WORLD, &reqs[0]);
   MPI_Irecv(h_rdata, N*sizeof(DATATYPE), MPI_BYTE, from, 111, MPI_COMM_WORLD, &reqs[1]);
#endif
   MPI_Waitall(2, reqs, stats);

#ifndef CUDA_AWARE

#ifdef VERY_VERBOSE
   for(int rank=0; rank<mpi_size; rank++){
      if(rank==mpi_rank){
         for(int i=0; i<N; i++){
            printf("recv %2d | %4d %d\n", rank, i, h_rdata[i]);
         }
      }
      MPI_Barrier(MPI_COMM_WORLD);
   }
#endif

   
   HANDLE_ERROR( cudaMemcpy(d_rdata, h_rdata, N*sizeof(DATATYPE), cudaMemcpyHostToDevice) );   
#endif

   //
   // Now check the data
   //
#ifdef CUDA_AWARE
   HANDLE_ERROR( cudaMemcpy(h_rdata, d_rdata, N*sizeof(DATATYPE), cudaMemcpyDeviceToHost) );
#endif
   int nerr=0;
   DATATYPE ans;
   for(int i=0; i<N; i++){
      // ans = 1.0*from+1.0e-6*i;
      ans = 100000*from + i;
      if(h_rdata[i] != ans){ 
         nerr++;
      }
   }

   int allerr;
   MPI_Reduce(&nerr,&allerr,1,MPI_INT,MPI_SUM,0,MPI_COMM_WORLD);
   if(mpi_rank==0){
      if(allerr==0) printf("Using N=%9d ...passed\n",N);
      else          printf("Using N=%9d ...failed\n",N);
   }

   MPI_Barrier(MPI_COMM_WORLD);

#ifdef DEBUG_MODE
   printf("Proc %2d has %6d errors\n",mpi_rank, nerr);
#endif

   HANDLE_ERROR( cudaFree(d_sdata) );
   HANDLE_ERROR( cudaFree(d_rdata) );
   delete h_sdata;
   delete h_rdata;

}


int main(int argc, char** argv) {

  MPI_Init(&argc, &argv);

  int mpi_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);

  if(mpi_rank == 0){
#ifdef CUDA_AWARE
     printf("Attempting CUDA AWARE data transfers...\n");
#else
     printf("Not CUDA aware...\n");
#endif
  }

#ifdef DEBUG_MODE
  // go(1<<20);
  // go(8);
  go(1<<6);
#else
  for(int i=5; i<20; i++){
     go(1<<i);
  }
#endif

  MPI_Barrier(MPI_COMM_WORLD);
  if(mpi_rank == 0){
     printf("--------------- Finalizing ----------------\n");
  }

  MPI_Finalize();

  return 0;
}
