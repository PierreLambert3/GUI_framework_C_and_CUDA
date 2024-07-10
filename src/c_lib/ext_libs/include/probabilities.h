#ifndef PROBABILITIES_H
#define PROBABILITIES_H

#include "system.h"


// ---- seeding ----
// properties:
//      non-zero seed 
//      good distribution of zeros and ones
uint32_t generate_random_seed();

// ---- unsafe random number generators ----
//          UNSAFE for sensitive applications:
//          if the period is high when doing (r%period), the result will be biased (for instance if the rand did number between 0 and 4, and you do r%3, you'll get 0 twice as often as 1 or 2)
//          2^32 is a bit more than 4 billion, so be careful when getting close to that period (either re-generate or use a 64 bit generator)

/*
Precautions:
    - Do not use for sensitive applications
    - Do not use for Monte Carlo simulations
    - Do not use to generate ML datasets
    - If multiple threads are using the same state, make sure to lock the state

if the period is high when doing (r%period), the result will be biased (for instance if the rand did number between 0 and 4, and you do r%3, you'll get 0 twice as often as 1 or 2)
2^32 is a bit more than 4 billion, so be careful when getting close to that period (either re-generate or use a 64 bit generator)
*/
uint32_t unsafe_rand_uint32(uint32_t* state);
// excluding max
uint32_t unsafe_rand_uint32_between(uint32_t* state, uint32_t min, uint32_t max);

float unsafe_rand_float(uint32_t* state);
// excluding max
float unsafe_rand_float_between(uint32_t* state, float min, float max);

#endif // PROBABILITIES_H