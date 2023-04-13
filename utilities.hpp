#include <cassert>
#include <complex>
#include <cuda_runtime_api.h> 
#include <cusparse.h>
#include <random>
#include <stdio.h>
#include <stdlib.h>

#define CHECK_CUDA(func)                                                       \
{                                                                              \
    cudaError_t status = (func);                                               \
    if (status != cudaSuccess) {                                               \
        printf("CUDA API failed at line %d with error: %s (%d)\n",             \
               __LINE__, cudaGetErrorString(status), status);                  \
        return EXIT_FAILURE;                                                   \
    }                                                                          \
}

#define CHECK_CUSPARSE(func)                                                   \
{                                                                              \
    cusparseStatus_t status = (func);                                          \
    if (status != CUSPARSE_STATUS_SUCCESS) {                                   \
        printf("CUSPARSE API failed at line %d with error: %s (%d)\n",         \
               __LINE__, cusparseGetErrorString(status), status);              \
        return EXIT_FAILURE;                                                   \
    }                                                                          \
}

template<class T>
T* generate_random_dense_matrix(int rows, int cols) {
    // Generate RNG
    std::random_device rand_dev;
    std::seed_seq sequence{
        rand_dev(), rand_dev(), rand_dev(), rand_dev(), rand_dev(),
        rand_dev(), rand_dev(), rand_dev(), rand_dev(), rand_dev()
    };
    std::mt19937_64 rand_engine(sequence);
    std::normal_distribution<T> distribution(0.0, 1.0);

    // Allocate memory
    T* matrix = new T[rows * cols];

    // Fill matrix
    for (int i = 0; i < rows * cols; i++) {
        matrix[i] = distribution(rand_engine);
    }

    return matrix;
}

template<class T>
int64_t generate_random_banded_matrix(int rows, int cols, int bands, T** data, int** indices, int** indptr) {
    // Assert bands is odd
    assert(bands % 2 == 1);

    // Generate RNG
    std::random_device rand_dev;
    std::seed_seq sequence{
        rand_dev(), rand_dev(), rand_dev(), rand_dev(), rand_dev(),
        rand_dev(), rand_dev(), rand_dev(), rand_dev(), rand_dev()
    };
    std::mt19937_64 rand_engine(sequence);
    std::normal_distribution<T> distribution(0.0, 1.0);

    // Calculate nnz
    // First (b - 1) / 2 rows
    // (b + 1) / 2 + (b + 3) / 2 + ... + (b - 1) = (b^2 - b) / 2 - (b^2 - 1) / 8
    // Last (b - 1) / 2 rows
    // Same as above
    // Middle r - b + 1 rows
    // (r - b + 1) * b
    int64_t nnz = bands * bands - bands - (bands * bands - 1) / 4 + (rows - bands + 1) * bands;
    printf("nnz: %ld\n", nnz);

    // Allocate mmemory
    *data = new T[nnz]{1};
    *indices = new int[nnz]{2};
    *indptr = new int[rows + 1]{3};

    // Fill indptr and indices
    (*indptr)[0] = 0;
    for (int i = 0; i < (bands - 1) / 2; i++) {
        int num_cols = (bands + 1) / 2 + i;
        int first_col = 0;
        (*indptr)[i + 1] = (*indptr)[i] + num_cols;
        for (int j = 0; j < num_cols; j++) {
            (*indices)[(*indptr)[i] + j] = first_col + j;
        }
    }
    for (int i = (bands - 1) / 2; i < rows - (bands - 1) / 2; i++) {
        int num_cols = bands;
        int first_col = i - (bands - 1) / 2;
        (*indptr)[i + 1] = (*indptr)[i] + num_cols;
        for (int j = 0; j < num_cols; j++) {
            (*indices)[(*indptr)[i] + j] = first_col + j;
        }
    }
    for (int i = rows - (bands - 1) / 2; i < rows; i++) {
        int num_cols = (bands - 1) / 2 + (rows - i);
        int first_col = i - (bands - 1) / 2;
        (*indptr)[i + 1] = (*indptr)[i] + (bands - 1) / 2 + (rows - i);
        for (int j = 0; j < num_cols; j++) {
            (*indices)[(*indptr)[i] + j] = first_col + j;
        }
    }
    (*indptr)[rows] = nnz;

    // Fill data
    for (int i = 0; i < nnz; i++) {
        (*data)[i] = distribution(rand_engine);
    }

    // for (int i = 0; i < rows + 1; i++) {
    //     printf("%d ", (*indptr)[i]);
    // }
    // printf("\n");

    return nnz;
}

template<class T>
void cudaMemcpyHostToDevice_banded_matrix(int rows, int cols, int bands, T* data, int* indices, int* indptr, T** d_data, int** d_indices, int** d_indptr) {
    // Assert bands is odd
    assert(bands % 2 == 1);

    // Calculate nnz
    // First (b - 1) / 2 rows
    // (b + 1) / 2 + (b + 3) / 2 + ... + (b - 1) = (b^2 - b) / 2 - (b^2 - 1) / 8
    // Last (b - 1) / 2 rows
    // Same as above
    // Middle r - b + 1 rows
    // (r - b + 1) * b
    int64_t nnz = bands * bands - bands - (bands * bands - 1) / 4 + (rows - bands + 1) * bands;

    // Allocate mmemory
    cudaMalloc((void**)d_data, nnz * sizeof(T));
    cudaMalloc((void**)d_indices, nnz * sizeof(int));
    cudaMalloc((void**)d_indptr, (rows + 1) * sizeof(int));

    // Copy data
    cudaMemcpy(*d_data, data, nnz * sizeof(T), cudaMemcpyHostToDevice);
    cudaMemcpy(*d_indices, indices, nnz * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(*d_indptr, indptr, (rows + 1) * sizeof(int), cudaMemcpyHostToDevice);
}

template<class T>
void cudaMemcpyDeviceToHost_banded_matrix(int rows, int cols, int nnz, T* d_data, int* d_indices, int* d_indptr, T* data, int* indices, int* indptr) {
    // // Assert bands is odd
    // assert(bands % 2 == 1);

    // // Calculate nnz
    // // First (b - 1) / 2 rows
    // // (b + 1) / 2 + (b + 3) / 2 + ... + (b - 1) = (b^2 - b) / 2 - (b^2 - 1) / 8
    // // Last (b - 1) / 2 rows
    // // Same as above
    // // Middle r - b + 1 rows
    // // (r - b + 1) * b
    // int64_t nnz = bands * bands - bands - (bands * bands - 1) / 4 + (rows - bands + 1) * bands;

    // Copy data
    cudaMemcpy(data, d_data, nnz * sizeof(T), cudaMemcpyDeviceToHost);
    cudaMemcpy(indices, d_indices, nnz * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(indptr, d_indptr, (rows + 1) * sizeof(int), cudaMemcpyDeviceToHost);
}

template<class T>
int csr_to_dense(int rows, int cols, int nnz, T* data, int* indices, int* indptr, T* dense) {
    cudaDataType computeType;
    if (typeid(T) == typeid(float)) {
        computeType = CUDA_R_32F;
    } else if (typeid(T) == typeid(double)) {
        computeType = CUDA_R_64F;
    } else if (typeid(T) == typeid(std::complex<float>)) {
        computeType = CUDA_C_32F;
    } else if (typeid(T) == typeid(std::complex<double>)) {
        computeType = CUDA_C_64F;
    } else {
        printf("Unsupported data type");
        return EXIT_FAILURE;
    }
    cusparseHandle_t     handle = NULL;
    cusparseSpMatDescr_t matA;
    cusparseDnMatDescr_t matB;
    void*                dBuffer    = NULL;
    size_t               bufferSize = 0;
    CHECK_CUSPARSE( cusparseCreate(&handle) )
    // Create sparse matrix A in CSR format
    CHECK_CUSPARSE( cusparseCreateCsr(&matA, rows, cols, nnz,
                                      indptr, indices, data,
                                      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                      CUSPARSE_INDEX_BASE_ZERO, computeType) )
    // Create dense matrix B
    CHECK_CUSPARSE( cusparseCreateDnMat(&matB, rows, cols, cols, dense,
                                        computeType, CUSPARSE_ORDER_ROW) )
    // allocate an external buffer if needed
    CHECK_CUSPARSE( cusparseSparseToDense_bufferSize(
                                        handle, matA, matB,
                                        CUSPARSE_SPARSETODENSE_ALG_DEFAULT,
                                        &bufferSize) )
    CHECK_CUDA( cudaMalloc(&dBuffer, bufferSize) )

    // execute Sparse to Dense conversion
    CHECK_CUSPARSE( cusparseSparseToDense(handle, matA, matB,
                                          CUSPARSE_SPARSETODENSE_ALG_DEFAULT,
                                          dBuffer) )
    // destroy matrix/vector descriptors
    CHECK_CUSPARSE( cusparseDestroySpMat(matA) )
    CHECK_CUSPARSE( cusparseDestroyDnMat(matB) )
    CHECK_CUSPARSE( cusparseDestroy(handle) )
}