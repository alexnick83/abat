#include <cassert>
#include <complex>
#include <cuda_runtime_api.h> 
#include "cublas_v2.h"
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

#define CHECK_CUBLAS(func)                                                     \
{                                                                              \
    cublasStatus_t status = (func);                                            \
    if (status != CUBLAS_STATUS_SUCCESS) {                                     \
        printf("CUBLAS API failed at line %d with error: (%d)\n",           \
               __LINE__, status);               \
        return EXIT_FAILURE;                                                   \
    }                                                                          \
}

template<class T>
void generate_random_dense_matrix(int rows, int cols, T **matrix, bool random = false) {
    // Generate RNG
    std::random_device rand_dev;
    std::seed_seq sequence{
        rand_dev(), rand_dev(), rand_dev(), rand_dev(), rand_dev(),
        rand_dev(), rand_dev(), rand_dev(), rand_dev(), rand_dev()
    };
    std::mt19937_64 rand_engine(sequence);
    std::normal_distribution<double> distribution(0.0, 1.0);

    // Allocate memory
    *matrix = new T[rows * cols];

    // Fill data
    if (random) {
        for (int i = 0; i < rows * cols; i++) {
            (*matrix)[i] = i + 1;
        }
    } else if (std::is_same<T, std::complex<float>>::value) {
        for (int i = 0; i < rows * cols; i++) {
            (*matrix)[i] = std::complex<float>(distribution(rand_engine), distribution(rand_engine));
        }
    } else if (std::is_same<T, std::complex<double>>::value) {
        for (int i = 0; i < rows * cols; i++) {
            (*matrix)[i] = std::complex<double>(distribution(rand_engine), distribution(rand_engine));
        }
    } else {
        for (int i = 0; i < rows * cols; i++) {
            (*matrix)[i] = distribution(rand_engine);
        }
    }
}

template<class T>
int64_t generate_random_banded_matrix(int rows, int cols, int bands, T** data, int** indices, int** indptr, bool random = false) {
    // Assert bands is odd
    assert(bands % 2 == 1);

    // Generate RNG
    std::random_device rand_dev;
    std::seed_seq sequence{
        rand_dev(), rand_dev(), rand_dev(), rand_dev(), rand_dev(),
        rand_dev(), rand_dev(), rand_dev(), rand_dev(), rand_dev()
    };
    std::mt19937_64 rand_engine(sequence);
    std::normal_distribution<double> distribution(0.0, 1.0);

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
    if (random) {
        for (int i = 0; i < nnz; i++) {
            (*data)[i] = i + 1;
        }
    } else if (std::is_same<T, std::complex<float>>::value) {
        for (int i = 0; i < nnz; i++) {
            (*data)[i] = std::complex<float>(distribution(rand_engine), distribution(rand_engine));
        }
    } else if (std::is_same<T, std::complex<double>>::value) {
        for (int i = 0; i < nnz; i++) {
            (*data)[i] = std::complex<double>(distribution(rand_engine), distribution(rand_engine));
        }
    } else {
        for (int i = 0; i < nnz; i++) {
            (*data)[i] = distribution(rand_engine);
        }
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
int cudaMemcpyDeviceToHost_banded_matrix(int rows, int cols, int nnz, T* d_data, int* d_indices, int* d_indptr, T* data, int* indices, int* indptr) {
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
    CHECK_CUDA( cudaMemcpy(data, d_data, nnz * sizeof(T), cudaMemcpyDeviceToHost)   )
    CHECK_CUDA( cudaMemcpy(indices, d_indices, nnz * sizeof(int), cudaMemcpyDeviceToHost) )
    CHECK_CUDA( cudaMemcpy(indptr, d_indptr, (rows + 1) * sizeof(int), cudaMemcpyDeviceToHost)  )

    return EXIT_SUCCESS;
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

    return EXIT_SUCCESS;
}

template<class T>
int matmul(char transa, char transb, int m, int n, int k, T *a, T *b, T *c) {

    // T alpha = (T) 1.0;
    // T beta = (T) 0.0;
    cuDoubleComplex alpha = make_cuDoubleComplex(1.0, 0.0);
    cuDoubleComplex beta = make_cuDoubleComplex(0.0, 0.0);
    cublasOperation_t opA = (transa == 'N' || transa == 'n') ? CUBLAS_OP_N : CUBLAS_OP_T;
    cublasOperation_t opB = (transb == 'N' || transb == 'n') ? CUBLAS_OP_N : CUBLAS_OP_T;
    int lda = (opA == CUBLAS_OP_N) ? m : k;
    int ldb = (opB == CUBLAS_OP_N) ? k : n;
    int ldc = m;

    cublasHandle_t handle;
    CHECK_CUBLAS( cublasCreate(&handle) )
    CHECK_CUBLAS( cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST) )

    CHECK_CUBLAS(   cublasZgemm(handle, opA, opB, m, n, k,
                                &alpha, (cuDoubleComplex*)a, lda, (cuDoubleComplex*)b, ldb,
                                &beta, (cuDoubleComplex*)c, ldc)   )
    
    CHECK_CUBLAS( cublasDestroy(handle) )

    return EXIT_SUCCESS;
}
