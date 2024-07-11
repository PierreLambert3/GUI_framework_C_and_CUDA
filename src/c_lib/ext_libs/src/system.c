#include "system.h"

// -------------------  ERROR HANDLING & GRACEFUL DEATH  -------------------
void die(const char* message){
    set_console_colour_error();
    printf("\n %s \n", message);
    reset_console_colour();
    exit(86);
}

// -------------------  CUDA: INIT AND INFO  -------------------
void initialise_cuda(bool verbose){
    // Set the device to use
    cudaError_t cuda_error = cudaSetDevice(0); // 0 is the default device
    if(cuda_error == cudaSuccess){
        set_console_colour_success();
        printf("[ CORRECT ]   CUDA init   ");
    } else {
        die("[  ERROR  ]   CUDA init   ");}
    reset_console_colour();
    printf("CUDA device set to 0\n");
    if(verbose){
        struct cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, 0);
        print_cuda_device_info(prop);
    }
}

void print_cuda_device_info(struct cudaDeviceProp prop){
    printf("  Device name: %s\n", prop.name);
    printf("  Compute capability: %d.%d\n", prop.major, prop.minor);
    printf("  Total global memory: %lu bytes\n", prop.totalGlobalMem);
    printf("  Maximum threads per block: %d\n", prop.maxThreadsPerBlock);
    printf("  Max threads per multiprocessor: %d\n",prop.maxThreadsPerMultiProcessor);
    printf("  Streaming Multiprocessor (SM) count: %d\n",prop.multiProcessorCount);
    printf("  Warp size: %d\n",prop.warpSize);
    printf("  Rough estimate of number of threads %d\n", prop.multiProcessorCount * prop.maxThreadsPerMultiProcessor);
    printf("  Maximum block dimensions: (%d, %d, %d)\n", prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
    printf("  Maximum grid dimensions: (%d, %d, %d)\n", prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
    printf("  Shared memory per block: %lu bytes   (float per shared memory: %lu)\n", prop.sharedMemPerBlock, prop.sharedMemPerBlock/sizeof(float));
    printf("Registers available per block: %d\n", prop.regsPerBlock);
    // printf("  Memory clock rate: %d kHz\n", prop.memoryClockRate);
    // printf("  Memory bus width: %d bits\n", prop.memoryBusWidth);
    printf("  Peak memory bandwidth: %f GB/s\n", 2.0 * prop.memoryClockRate * (prop.memoryBusWidth / 8) / 1.0e6);
}

// -------------------  TIME AND SLEEP  -------------------
void sleep_ms(uint32_t n){
    usleep(n * 1000u);
}

double timestamp_seconds_dbl(){
    return (double)time(NULL);
}

uint64_t timestamp_seconds(){
    return (uint64_t)time(NULL);
}

// -------------------  CONSOLE COLOURS  -------------------
void set_console_colour(uint8_t r, uint8_t g, uint8_t b) {
    printf("\e[38;2;%d;%d;%dm", r, g, b);
}

void set_console_colour_error(){
    printf("\e[38;2;%d;%d;%dm", TERMINAL_ERROR_COLOUR_R, TERMINAL_ERROR_COLOUR_G, TERMINAL_ERROR_COLOUR_B);
}

void set_console_colour_success(){
    printf("\e[38;2;%d;%d;%dm", TERMINAL_SUCCESS_COLOUR_R, TERMINAL_SUCCESS_COLOUR_G, TERMINAL_SUCCESS_COLOUR_B);
}

void set_console_colour_info(){
    printf("\e[38;2;%d;%d;%dm", TERMINAL_INFO_COLOUR_R, TERMINAL_INFO_COLOUR_G, TERMINAL_INFO_COLOUR_B);
    // printf("\e[38;2;%d;%d;%dm", TERMINAL_SUCCESS_COLOUR_R, TERMINAL_SUCCESS_COLOUR_G, TERMINAL_SUCCESS_COLOUR_B);
}

void reset_console_colour(){
    printf("\e[38;2;%d;%d;%dm", TERMINAL_TEXT_COLOUR_R, TERMINAL_TEXT_COLOUR_G, TERMINAL_TEXT_COLOUR_B);
}


// -------------------  MEMORY MANAGEMENT  -------------------
// mutex handlers
pthread_mutex_t* mutexes_allocate_and_init(uint32_t n_elements){
    pthread_mutex_t* mutexes = (pthread_mutex_t*)malloc(n_elements * sizeof(pthread_mutex_t));
    if (mutexes == NULL) {
        die("Failed to allocate memory for mutexes");}
    for (uint32_t i = 0; i < n_elements; i++) {
        if (pthread_mutex_init(&mutexes[i], NULL) != 0) {
            die("Failed to initialise mutex");}
    }
    return mutexes;
}

pthread_mutex_t* mutex_allocate_and_init(){
    pthread_mutex_t* mutex = (pthread_mutex_t*)malloc(sizeof(pthread_mutex_t));
    if (mutex == NULL) {
        die("Failed to allocate memory for mutex");}
    if (pthread_mutex_init(mutex, NULL) != 0) {
        die("Failed to initialise mutex");}
    return mutex;
}

// malloc handlers for 1d arrays
bool* malloc_bool_1d(uint32_t n_elements, bool init_val) {
    bool* array = (bool*)malloc(n_elements * sizeof(bool));
    if (array == NULL) {
        die("Failed to allocate memory for bool array");
    }
    for (uint32_t i = 0; i < n_elements; i++) {
        array[i] = init_val;
    }
    return array;
}

float* malloc_float_1d(uint32_t n_elements, float init_val) {
    float* array = (float*)malloc(n_elements * sizeof(float));
    if (array == NULL) {
        die("Failed to allocate memory for float array");
    }
    for (uint32_t i = 0; i < n_elements; i++) {
        array[i] = init_val;
    }
    return array;
}

double* malloc_double_1d(uint32_t n_elements, double init_val) {
    double* array = (double*)malloc(n_elements * sizeof(double));
    if (array == NULL) {
        die("Failed to allocate memory for double array");
    }
    for (uint32_t i = 0; i < n_elements; i++) {
        array[i] = init_val;
    }
    return array;
}

uint32_t* malloc_uint32_t_1d(uint32_t n_elements, uint32_t init_val) {
    uint32_t* array = (uint32_t*)malloc(n_elements * sizeof(uint32_t));
    if (array == NULL) {
        die("Failed to allocate memory for uint32_t array");
    }
    for (uint32_t i = 0; i < n_elements; i++) {
        array[i] = init_val;
    }
    return array;
}

uint8_t* malloc_uint8_t_1d(uint32_t n_elements, uint8_t init_val) {
    uint8_t* array = (uint8_t*)malloc(n_elements * sizeof(uint8_t));
    if (array == NULL) {
        die("Failed to allocate memory for uint8_t array");
    }
    for (uint32_t i = 0; i < n_elements; i++) {
        array[i] = init_val;
    }
    return array;
}

// malloc handlers for 2d matrices
bool** malloc_bool_matrix(uint32_t n, uint32_t m, bool init_val) {
    die("this in not tested yet");
    bool** matrix = (bool**)malloc((n + 1) * sizeof(bool*));
    if (matrix == NULL) {
        die("Failed to allocate memory for bool matrix");}
    bool* data = (bool*)malloc(n * m * sizeof(bool));
    if (data == NULL) {
        die("Failed to allocate memory for bool matrix data");}
    matrix[n] = data;  // Store a pointer to data in the matrix array
    for (uint32_t i = 0; i < n; i++) {
        matrix[i] = &data[m * i];
        for (uint32_t j = 0; j < m; j++) {
            matrix[i][j] = init_val;}
    }
    return matrix;
}

float** malloc_float_matrix(uint32_t n, uint32_t m, float init_val) {
    float** matrix = (float**)malloc((n + 1) * sizeof(float*));
    if (matrix == NULL) {
        die("Failed to allocate memory for float matrix");
    }
    float* data = (float*)malloc(n * m * sizeof(float));
    if (data == NULL) {
        die("Failed to allocate memory for float matrix data");}
    matrix[n] = data;  // Store a pointer to data in the matrix array
    for (uint32_t i = 0; i < n; i++) {
        matrix[i] = &data[m * i];
        for (uint32_t j = 0; j < m; j++) {
            matrix[i][j] = init_val;}
    }
    return matrix;
}

double** malloc_double_matrix(uint32_t n, uint32_t m, double init_val) {
    die("this in not tested yet");
    double** matrix = (double**)malloc((n + 1) * sizeof(double*));
    if (matrix == NULL) {
        die("Failed to allocate memory for double matrix");
    }
    double* data = (double*)malloc(n * m * sizeof(double));
    if (data == NULL) {
        die("Failed to allocate memory for double matrix data");
    }
    matrix[n] = data;  // Store a pointer to data in the matrix array
    for (uint32_t i = 0; i < n; i++) {
        matrix[i] = &data[m * i];
        for (uint32_t j = 0; j < m; j++) {
            matrix[i][j] = init_val;
        }
    }
    return matrix;
}

uint32_t** malloc_uint32_t_matrix(uint32_t n, uint32_t m, uint32_t init_val){
    uint32_t** matrix = (uint32_t**)malloc((n + 1) * sizeof(uint32_t*));
    if (matrix == NULL) {
        die("Failed to allocate memory for uint32_t matrix");
    }
    uint32_t* data = (uint32_t*)malloc(n * m * sizeof(uint32_t));
    if (data == NULL) {
        die("Failed to allocate memory for uint32_t matrix data");
    }
    matrix[n] = data;  // Store a pointer to data in the matrix array
    for (uint32_t i = 0; i < n; i++) {
        matrix[i] = &data[m * i];
        for (uint32_t j = 0; j < m; j++) {
            matrix[i][j] = init_val;
        }
    }
    return matrix;
}

// matrix shape handlers
inline float* matrix_as_1d_float(float** matrix, uint32_t n, uint32_t m){
    return matrix[n];
}

// matrix copy functions
inline void memcpy_float_matrix(float** recipient, float** original, uint32_t n, uint32_t m){
    memcpy(recipient[n], original[n], n*m*sizeof(float));
}

// free handlers for mallocs
void free_matrix(void** matrix, uint32_t n){
    free(matrix[n]);  // Free the data array
    free(matrix);     // Free the array of pointers (the rows)
}

void free_1d(void* array){
    free(array);
}


// -------------------  MEMORY MANAGEMENT : CUDA -------------------
// malloc handlers for 1d arrays on GPU
void malloc_1d_float_cuda(float** ptr_array, uint32_t n_elements){
    cudaError_t cuda_error = cudaMalloc((void**)ptr_array, n_elements * sizeof(float));
    if (cuda_error != cudaSuccess) {
        printf("error: %s\n", cudaGetErrorString(cuda_error));
        die("Failed to allocate memory on GPU");
    }
}

void malloc_1d_double_cuda(double** ptr_array, uint32_t n_elements){
    cudaError_t cuda_error = cudaMalloc((void**)ptr_array, n_elements * sizeof(double));
    if (cuda_error != cudaSuccess) {
        printf("error: %s\n", cudaGetErrorString(cuda_error));
        die("Failed to allocate memory on GPU");
    }
}

void malloc_1d_uint32_cuda(uint32_t** ptr_array, uint32_t n_elements){
    cudaError_t cuda_error = cudaMalloc((void**)ptr_array, n_elements * sizeof(uint32_t));
    if (cuda_error != cudaSuccess) {
        printf("error: %s\n", cudaGetErrorString(cuda_error));
        die("Failed to allocate memory on GPU");
    }
}

// copying data between CPU and GPU
void memcpy_CPU_to_CUDA_float(float* ptr_array_GPU, float* ptr_array_CPU, uint32_t n_elements){
    cudaError_t cuda_error = cudaMemcpy(ptr_array_GPU, ptr_array_CPU, n_elements * sizeof(float), cudaMemcpyHostToDevice);
    if (cuda_error != cudaSuccess) {
        printf("error: %s\n", cudaGetErrorString(cuda_error));
        die("Failed to copy memory from CPU to GPU");
    }
}

void memcpy_CPU_to_CUDA_double(double* ptr_array_GPU, double* ptr_array_CPU, uint32_t n_elements){
    cudaError_t cuda_error = cudaMemcpy(ptr_array_GPU, ptr_array_CPU, n_elements * sizeof(double), cudaMemcpyHostToDevice);
    if (cuda_error != cudaSuccess) {
        printf("error: %s\n", cudaGetErrorString(cuda_error));
        die("Failed to copy memory from CPU to GPU");
    }
}

void memcpy_CPU_to_CUDA_uint32(uint32_t* ptr_array_GPU, uint32_t* ptr_array_CPU, uint32_t n_elements){
    cudaError_t cuda_error = cudaMemcpy(ptr_array_GPU, ptr_array_CPU, n_elements * sizeof(uint32_t), cudaMemcpyHostToDevice);
    if (cuda_error != cudaSuccess) {
        printf("error: %s\n", cudaGetErrorString(cuda_error));
        die("Failed to copy memory from CPU to GPU");
    }
}

void memcpy_CUDA_to_CPU_float(float* ptr_array_CPU, float* ptr_array_GPU, uint32_t n_elements){
    cudaError_t cuda_error = cudaMemcpy(ptr_array_CPU, ptr_array_GPU, n_elements * sizeof(float), cudaMemcpyDeviceToHost);
    if (cuda_error != cudaSuccess) {
        printf("error: %s\n", cudaGetErrorString(cuda_error));
        die("Failed to copy memory from GPU to CPU");
    }
}

void memcpy_CUDA_to_CPU_uint32(uint32_t* ptr_array_CPU, uint32_t* ptr_array_GPU, uint32_t n_elements){
    cudaError_t cuda_error = cudaMemcpy(ptr_array_CPU, ptr_array_GPU, n_elements * sizeof(uint32_t), cudaMemcpyDeviceToHost);
    if (cuda_error != cudaSuccess) {
        printf("error: %s\n", cudaGetErrorString(cuda_error));
        die("Failed to copy memory from GPU to CPU");
    }
}

void memcpy_CUDA_to_CPU_double(double* ptr_array_CPU, double* ptr_array_GPU, uint32_t n_elements){
    cudaError_t cuda_error = cudaMemcpy(ptr_array_CPU, ptr_array_GPU, n_elements * sizeof(double), cudaMemcpyDeviceToHost);
    if (cuda_error != cudaSuccess) {
        printf("error: %s\n", cudaGetErrorString(cuda_error));
        die("Failed to copy memory from GPU to CPU");
    }
}

// -------------------  CPU/GPU safe communication -------------------
static void init_GPU_CPU_sync(GPU_CPU_sync* sync) {
    sync->mutex_request = mutex_allocate_and_init();
    sync->mutex_ready   = mutex_allocate_and_init();
    sync->mutex_buffer  = mutex_allocate_and_init();
    sync->flag_request  = false;
    sync->flag_ready    = false;
}

static void init_GPU_CPU_float_comms(GPU_CPU_float_comms* thing, uint32_t size){
    thing->buffer = malloc_float_1d(size, 0.f);
    init_GPU_CPU_sync(&thing->sync);
}

static void init_GPU_CPU_uint32_comms(GPU_CPU_uint32_comms* thing, uint32_t size){
    thing->buffer = malloc_uint32_t_1d(size, 0u);
    init_GPU_CPU_sync(&thing->sync);
}

GPU_CPU_float_comms* create_GPU_CPU_float_comms(uint32_t size){
    GPU_CPU_float_comms* thing = (GPU_CPU_float_comms*)malloc(sizeof(GPU_CPU_float_comms));
    if (thing == NULL) {
        die("Failed to allocate memory for GPU_CPU_float_buffer");}
    init_GPU_CPU_float_comms(thing, size);
    return thing;
}

GPU_CPU_uint32_comms* create_GPU_CPU_uint32_comms(uint32_t size){
    GPU_CPU_uint32_comms* thing = (GPU_CPU_uint32_comms*)malloc(sizeof(GPU_CPU_uint32_comms));
    if (thing == NULL) {
        die("Failed to allocate memory for GPU_CPU_uint32_buffer");}
    init_GPU_CPU_uint32_comms(thing, size);
    return thing;
}

bool sync_is_requesting_now(GPU_CPU_sync* sync){
    pthread_mutex_lock(sync->mutex_request);
    bool flag = sync->flag_request;
    pthread_mutex_unlock(sync->mutex_request);
    return flag;
}

bool sync_is_ready_now(GPU_CPU_sync* sync){
    pthread_mutex_lock(sync->mutex_ready);
    bool flag = sync->flag_ready;
    pthread_mutex_unlock(sync->mutex_ready);
    return flag;
}

static void set_ready(GPU_CPU_sync* sync, bool value){
    pthread_mutex_lock(sync->mutex_ready);
    sync->flag_ready = value;
    pthread_mutex_unlock(sync->mutex_ready);
}

static void set_request(GPU_CPU_sync* sync, bool value){
    pthread_mutex_lock(sync->mutex_request);
    sync->flag_request = value;
    pthread_mutex_unlock(sync->mutex_request);
}

void sync_notify_ready(GPU_CPU_sync* sync){
    set_request(sync, false);
    set_ready(sync, true);
}

void sync_notify_request(GPU_CPU_sync* sync){
    set_ready(sync, false);
    set_request(sync, true);
}

// -------------------  system speed tests -------------------

void speed_test_ints_vs_floats(uint32_t n_elements, bool without_caching){
    float* float_array = malloc_float_1d(n_elements, 0.0f);
    uint32_t* int_array = malloc_uint32_t_1d(n_elements, 0u);
    uint8_t* uint8_array = malloc_uint8_t_1d(n_elements, 0u); // Add uint8_t array
    uint32_t seed = generate_random_seed();
    for(uint32_t i = 0; i < n_elements; i++){
        float_array[i] = unsafe_rand_float_between(&seed, -0.98f, 1.0f);
        int_array[i]   = unsafe_rand_uint32_between(&seed, 0u, 255u);
        uint8_array[i] = (uint8_t) unsafe_rand_uint32_between(&seed, 0u, 255u);
    }
    clock_t start_time = clock();
    float sum_float = 0.0f;
    if(without_caching){
        for(uint32_t i = 0; i < n_elements; i++){
            sum_float += float_array[rand() % n_elements];
        }
    }
    else{
        for(uint32_t i = 0; i < n_elements; i++){
            sum_float += float_array[i];
        }
    }
    
    clock_t end_time = clock();
    double time_taken_float = ((double)(end_time - start_time)) / CLOCKS_PER_SEC;
    start_time = clock();
    uint32_t sum_int = 0;
    if(without_caching){
        for(uint32_t i = 0; i < n_elements; i++){
            sum_int += int_array[rand() % n_elements];
        }
    }
    else{
        for(uint32_t i = 0; i < n_elements; i++){
            sum_int += int_array[i];
        }
    }
    end_time = clock();
    float time_taken_uint32_t = ((double)(end_time - start_time)) / CLOCKS_PER_SEC;
    start_time = clock();
    uint32_t sum_uint8 = 0;
    if(without_caching){
        for(uint32_t i = 0; i < n_elements; i++){
            sum_uint8 += uint8_array[rand() % n_elements];
        }
    }
    else{
        for(uint32_t i = 0; i < n_elements; i++){
            sum_uint8 += uint8_array[i];
        }
    }
    end_time = clock();
    float time_taken_uint8_t = ((double)(end_time - start_time)) / CLOCKS_PER_SEC;
    free_1d(float_array);
    free_1d(int_array);
    free_1d(uint8_array);
    if(without_caching){
        printf("without caching\n");
    }
    else{
        printf("with caching\n");
    }
    printf("Time taken for float sum: %.6f seconds   (sum: %f)\n", time_taken_float, sum_float);
    printf("Time taken for uint32_t sum: %.6f seconds   (sum: %u)\n", time_taken_uint32_t, sum_int);
    printf("Time taken for uint8_t sum: %.6f seconds   (sum: %u)\n", time_taken_uint8_t, sum_uint8);
    printf("uint8_t is %.2f times faster than uint32_t\n", time_taken_uint32_t / time_taken_uint8_t);
    printf("uint32_t is %.2f times faster than float\n", time_taken_float / time_taken_uint32_t);
}


