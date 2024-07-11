#include "common_includes.h"
#include "gui_logic.h"

#define PROGRAM_SEED -1


#include "reduction.h"

void example_cuda_interfacing(){
    uint32_t n_elements = (uint32_t)(1024*571.59);
    // make arr on the CPU
    float* arr = malloc_float_1d(n_elements, 0.0f);
    // make arr on the GPU
    float* cu_arr;
    malloc_1d_float_cuda(&cu_arr, n_elements);
    // fill arr with random numbers, and compute sum on CPU
    uint32_t seed = generate_random_seed();
    float sum_cpu = 0.0f;
    for(uint32_t i = 0; i < n_elements; i++){
        arr[i]   = unsafe_rand_float_between(&seed, -0.98f, 1.0f);
        sum_cpu += arr[i];
    }
    // copy contents to GPU
    memcpy_CPU_to_CUDA_float(cu_arr, arr, n_elements);
    // init the dummy struct
    MyDummyStruct* dummy_struct = (MyDummyStruct*)malloc(sizeof(MyDummyStruct));
    dummy_struct->arr = cu_arr;
    dummy_struct->n   = n_elements;
    // call the reduction function
    badReduceWrapper(dummy_struct);
    
    float sum_gpu = 0.0f;
    cudaMemcpy(&sum_gpu, cu_arr, sizeof(float), cudaMemcpyDeviceToHost);
    if((fabs(sum_cpu - sum_gpu) / sum_cpu) < 1e-5){
        set_console_colour_success(); 
        printf("[ CORRECT ]   GPU wrapper test   ");
    } else {
        set_console_colour_error();
        printf("[  ERROR  ]   GPU wrapper test   ");
        die("GPU wrapper test failed!");
    }
    reset_console_colour();
    printf("sum on CPU: %.4f on GPU: %.4f (diff: %.6f,  relative diff: %.6f)\n", sum_cpu, sum_gpu, sum_cpu - sum_gpu, fabs(sum_cpu - sum_gpu) / sum_cpu);
    free_1d(arr);
    cudaFree(cu_arr);
} 

// requirements:
// install SDL2, and TTF and gfx (apt: libsdl2-gfx-dev)
// also need to have cuda runtime installed
int main() {
    printf("\n");
    reset_console_colour();

    // initilaize the random number generator
    if(PROGRAM_SEED > 0){
        srand((unsigned int)PROGRAM_SEED);
    } else {
        unsigned int general_seed = (unsigned int)time(NULL);
        printf("Seeding program with %u\n", general_seed);
        srand(general_seed);
    }
    // initialise the cuda device
    initialise_cuda(0);
    
    // initialise the gui logic (which in turn initialises the drawer, which in turn initialises SDL)
    GuiLogic* gui_logic;
    GuiLogic_init(&gui_logic, generate_random_seed());

    // run the program
    GuiLogic_run(gui_logic);

    example_cuda_interfacing();

    // check how fast CPU operations on floats are compared to uint32_t and uint8_t
    speed_test_ints_vs_floats((uint32_t)1e7, true);
    speed_test_ints_vs_floats((uint32_t)1e7, false);

    return 0;
}




