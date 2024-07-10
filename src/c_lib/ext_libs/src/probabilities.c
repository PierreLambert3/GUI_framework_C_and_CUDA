#include "probabilities.h"

// ---- seeding ----
uint32_t generate_random_seed(){
    uint32_t state = (uint32_t) rand();
    while(state == 0u){
        state = rand();}
    return state;
}

// ---- unsafe random number generators ----
//  always call generate_random_seed() to get a seed
inline uint32_t unsafe_rand_uint32(uint32_t* rand_state) {
    *rand_state ^= *rand_state << 13u;
    *rand_state ^= *rand_state >> 17u;
    *rand_state ^= *rand_state << 5u;
    return *rand_state;
}

inline uint32_t unsafe_rand_uint32_between(uint32_t* rand_state, uint32_t min, uint32_t max) {
    return min + unsafe_rand_uint32(rand_state) % (max - min);
}

inline float unsafe_rand_float(uint32_t* rand_state) {
    return (float)unsafe_rand_uint32(rand_state) / (float)UINT32_MAX;
}

inline float unsafe_rand_float_between(uint32_t* rand_state, float min, float max) {
    return min + unsafe_rand_float(rand_state) * (max - min);
}
