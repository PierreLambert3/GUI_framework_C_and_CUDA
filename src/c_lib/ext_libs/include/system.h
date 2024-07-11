#ifndef SYSTEM_H
#define SYSTEM_H

#include <unistd.h>
#include <stdbool.h>
#include <stdint.h>
#include <time.h>
#include <pthread.h>
#include <omp.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "lib_constants.h"

#include "probabilities.h"

// -------------------  ERROR HANDLING  -------------------
void die(const char* message);    // print message and exit

// -------------------  CUDA: INIT AND INFO  -------------------
void initialise_cuda(bool verbose);
void print_cuda_device_info(struct cudaDeviceProp prop);

// -------------------  TIME AND SLEEP  -------------------
void     sleep_ms(uint32_t n);    // sleep for n_ms milliseconds
double   timestamp_seconds_dbl(); // never use clock() for timing
uint64_t timestamp_seconds();

// -------------------  CONSOLE COLOURS  -------------------
void set_console_colour(uint8_t r, uint8_t g, uint8_t b); // sets the console colour to the specified RGB values
void set_console_colour_error();
void set_console_colour_success();
void set_console_colour_info();
void reset_console_colour();

// -------------------  MEMORY MANAGEMENT : CPU  -------------------
// mutex handlers
pthread_mutex_t* mutexes_allocate_and_init(uint32_t n_elements); // array of mutexes
pthread_mutex_t* mutex_allocate_and_init(); // single mutex
// malloc handlers for 1d arrays
bool*      malloc_bool_1d(uint32_t n_elements, bool init_val);
float*     malloc_float_1d(uint32_t n_elements, float init_val);
double*    malloc_double_1d(uint32_t n_elements, double init_val);
uint32_t*  malloc_uint32_t_1d(uint32_t n_elements, uint32_t init_val);
// malloc handlers for 2d matrices
// matrix: 2 malloc calls! -->  always free the matrix using free_matrix()
//  | matrix[0] = row 0, ..., matrix[n-1] = row n-1 |  
//  | matrix[n] = ptr to data |
//  if given to cuda, need to be flattened to 1d array using matrix_as_1d_float()
bool**     malloc_bool_matrix(uint32_t n, uint32_t m, bool init_val);
float**    malloc_float_matrix(uint32_t n, uint32_t m, float init_val);
double**   malloc_double_matrix(uint32_t n, uint32_t m, double init_val);
uint32_t** malloc_uint32_t_matrix(uint32_t n, uint32_t m, uint32_t init_val);
// matrix shape handlers
float*     matrix_as_1d_float(float** matrix, uint32_t n, uint32_t m);
// matrix copy functions
void       memcpy_float_matrix(float** recipient, float** original, uint32_t n, uint32_t m);
// free handlers for mallocs
void       free_matrix(void** matrix, uint32_t n);
void       free_1d(void* array);


// -------------------  MEMORY MANAGEMENT : CUDA -------------------
// malloc handlers for 1d arrays on GPU
void malloc_1d_float_cuda(float** ptr_array_GPU, uint32_t n_elements);
void malloc_1d_double_cuda(double** ptr_array_GPU, uint32_t n_elements);
void malloc_1d_uint32_cuda(uint32_t** ptr_array_GPU, uint32_t n_elements);
// copying data between CPU and GPU
void memcpy_CPU_to_CUDA_float(float* ptr_array_GPU, float* ptr_array_CPU, uint32_t n_elements);
void memcpy_CPU_to_CUDA_uint32(uint32_t* ptr_array_GPU, uint32_t* ptr_array_CPU, uint32_t n_elements);
void memcpy_CPU_to_CUDA_double(double* ptr_array_GPU, double* ptr_array_CPU, uint32_t n_elements);
void memcpy_CUDA_to_CPU_float(float* ptr_array_CPU, float* ptr_array_GPU, uint32_t n_elements);
void memcpy_CUDA_to_CPU_uint32(uint32_t* ptr_array_CPU, uint32_t* ptr_array_GPU, uint32_t n_elements);
void memcpy_CUDA_to_CPU_double(double* ptr_array_CPU, double* ptr_array_GPU, uint32_t n_elements);


// -------------------  CPU/GPU safe communication -------------------
/*
!  need to represent matrices as 1d array

Example uses:

CPU ------>   BUFFER
GPU_CPU_sync* sync_neighsHD = &thing->GPU_CPU_comms_neighsHD->sync;
if(is_requesting_now(sync_neighsHD) && !is_ready_now(sync_neighsHD)){
    // wait for the subthreads to finish
    wait_full_path_finished(thing);
    // copy the neighsHD to the buffer, safely
    pthread_mutex_lock(thing->GPU_CPU_comms_neighsHD->sync.mutex_buffer);
    memcpy(thing->GPU_CPU_comms_neighsHD->buffer, as_uint32_1d(thing->neighsHD, thing->N, thing->Khd), thing->N*thing->Khd*sizeof(uint32_t));
    pthread_mutex_unlock(thing->GPU_CPU_comms_neighsHD->sync.mutex_buffer);
    // notify the GPU that the data is ready
    notify_ready(sync_neighsHD);
}

BUFFER ------> GPU
GPU_CPU_sync* sync = &thing->GPU_CPU_comms_neighsLD->sync;
if(is_ready_now(sync)){
    pthread_mutex_lock(sync->mutex_buffer);
    cudaMemcpy(thing->neighsLD_cuda, thing->GPU_CPU_comms_neighsLD->buffer, thing->N*thing->Kld*sizeof(uint32_t), cudaMemcpyHostToDevice);
    pthread_mutex_unlock(sync->mutex_buffer);
    set_ready(sync, false);
}
if(!is_requesting_now(sync)){
    notify_request(sync); // request for the next sync
}
*/

// a struct that contains flags and mutexes for synchronisation
typedef struct {
    // request a buffer update
    pthread_mutex_t* mutex_request;
    bool             flag_request;
    // notify that the buffer has been updated 
    pthread_mutex_t* mutex_ready;
    bool             flag_ready;
    // mutex for the buffer itself
    pthread_mutex_t* mutex_buffer;
} GPU_CPU_sync;

// a struct that contains a float buffer
typedef struct {
    GPU_CPU_sync     sync;
    float*           buffer;
} GPU_CPU_float_comms;

// a struct that contains a uint32_t buffer
typedef struct {
    GPU_CPU_sync     sync;
    uint32_t*        buffer;
} GPU_CPU_uint32_comms;

GPU_CPU_float_comms*  create_GPU_CPU_float_comms(uint32_t size);
GPU_CPU_uint32_comms* create_GPU_CPU_uint32_comms(uint32_t size);
bool sync_is_requesting_now(GPU_CPU_sync* sync);
bool sync_is_ready_now(GPU_CPU_sync* sync);
void sync_notify_ready(GPU_CPU_sync* sync);
void sync_notify_request(GPU_CPU_sync* sync);


// -------------------  system speed tests -------------------
void speed_test_ints_vs_floats(uint32_t n_elements, bool without_caching);

#endif // SYSTEM_H
