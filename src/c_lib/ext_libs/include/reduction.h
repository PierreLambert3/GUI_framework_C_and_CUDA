#ifndef REDUCTION_H
#define REDUCTION_H

#include <stdint.h>

// dummy struct for badReduceWrapper
typedef struct {
    float* arr;
    int n;
    uint32_t someUint32Value;
} MyDummyStruct;

#ifdef __cplusplus
extern "C" {
#endif

// (float *input, int size)
// void badReduceWrapper(float*, int);
void badReduceWrapper(MyDummyStruct*);

#ifdef __cplusplus
}
#endif

#endif // REDUCTION_H