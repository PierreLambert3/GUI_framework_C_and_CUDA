#ifndef VECTOR_H
#define VECTOR_H
#include "system.h"

// for each dimension M, the data is rescaled to have mean 0 and variance 1
void normalise_float_matrix(float** X, uint32_t N, uint32_t M);

// float Euclidean distance between two vectors of size M
float f_euclidean_sq(float* Xi, float* Xj, uint32_t M);

// float Euclidean distance between two vectors of known sizes
float f_euclidean_sq_2d(float* Xi, float* Xj);
float f_euclidean_sq_3d(float* Xi, float* Xj);

#endif // VECTOR_H
