#include "probabilities.h"
#include "dataset_loaders.h"

void load_mnist_train(uint32_t* N, uint32_t* M, float*** X, uint32_t** Y){
    FILE* X_file = fopen("../datasets/MNIST/MNIST_PCA_X.bin", "r");
    FILE* Y_file = fopen("../datasets/MNIST/MNIST_PCA_Y.bin", "r");
    if (X_file == NULL || Y_file == NULL) {
        die("file not found");
    }
    N[0] = 60000u;
    M[0] = 50u;
    Y[0] = malloc_uint32_t_1d(*N, 0);
    X[0] = malloc_float_matrix(*N, *M, -42.0f);

    float* tmp_Y = malloc_float_1d(*N, 0.0f);
    for (uint32_t i = 0; i < *N; i++) {
        if (fread(X[0][i], sizeof(float), *M, X_file) != *M) {
            die("Error reading X value");
        }
        if (fread(&tmp_Y[i], sizeof(float), 1, Y_file) != 1) {
            die("Error reading Y value");
        }
    }
    // convert Y to integer
    for (uint32_t i = 0; i < *N; i++) {
        Y[0][i] = (uint32_t)tmp_Y[i];
    }

    fclose(X_file);
    fclose(Y_file);
    free(tmp_Y);
}

