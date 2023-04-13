#include "utilities.hpp"

template<class T>
int banded_mm(char transa, char transb, int m, int n, int k, int64_t a_nnz, int64_t b_nnz, int64_t& c_nnz,
              T* a_data, int* a_indices, int* a_indptr,
              T* b_data, int* b_indices, int* b_indptr,
              T** c_data_buf, int** c_indices_buf, int** c_indptr_buf,
              int64_t& nnz_bufsz, int& rowsp1_bufsz,
              size_t& bufsz1, void** buf1, size_t& bufsz2, void** buf2) {
    T alpha = (T) 1.0;
    T beta = (T) 0.0;
    cusparseOperation_t opA = transa == 'T' ? CUSPARSE_OPERATION_TRANSPOSE : CUSPARSE_OPERATION_NON_TRANSPOSE;
    cusparseOperation_t opB = transb == 'T' ? CUSPARSE_OPERATION_TRANSPOSE : CUSPARSE_OPERATION_NON_TRANSPOSE;
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
    //--------------------------------------------------------------------------
    // CUSPARSE APIs
    cusparseHandle_t     handle = NULL;
    cusparseSpMatDescr_t matA, matB, matC;
    // void*  dBuffer1    = NULL, *dBuffer2   = NULL;
    size_t bufferSize1 = 0,    bufferSize2 = 0;
    CHECK_CUSPARSE( cusparseCreate(&handle) )
    // Create sparse matrix A in CSR format
    CHECK_CUSPARSE( cusparseCreateCsr(&matA, m, k, a_nnz,
                                      a_indptr, a_indices, a_data,
                                      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                      CUSPARSE_INDEX_BASE_ZERO, computeType) )
    CHECK_CUSPARSE( cusparseCreateCsr(&matB, k, n, b_nnz,
                                      b_indptr, b_indices, b_data,
                                      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                      CUSPARSE_INDEX_BASE_ZERO, computeType) )
    CHECK_CUSPARSE( cusparseCreateCsr(&matC, m, n, 0,
                                      NULL, NULL, NULL,
                                      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                      CUSPARSE_INDEX_BASE_ZERO, computeType) )
    //--------------------------------------------------------------------------
    // SpGEMM Computation
    cusparseSpGEMMDescr_t spgemmDesc;
    CHECK_CUSPARSE( cusparseSpGEMM_createDescr(&spgemmDesc) )

    // ask bufferSize1 bytes for external memory
    CHECK_CUSPARSE(
        cusparseSpGEMM_workEstimation(handle, opA, opB,
                                      &alpha, matA, matB, &beta, matC,
                                      computeType, CUSPARSE_SPGEMM_DEFAULT,
                                      spgemmDesc, &bufferSize1, NULL) )
    printf("bufferSize1 = %zu\n", bufferSize1);
    if (bufsz1 < bufferSize1) {
        if (bufsz1 > 0) {
            CHECK_CUDA( cudaFree(*buf1) )
        }
        CHECK_CUDA( cudaMalloc((void**) buf1, bufferSize1) )
    }

    // CHECK_CUDA( cudaMalloc((void**) &dBuffer1, bufferSize1) )
    // inspect the matrices A and B to understand the memory requirement for
    // the next step
    CHECK_CUSPARSE(
        cusparseSpGEMM_workEstimation(handle, opA, opB,
                                      &alpha, matA, matB, &beta, matC,
                                      computeType, CUSPARSE_SPGEMM_DEFAULT,
                                      spgemmDesc, &bufferSize1, *buf1) )

    // ask bufferSize2 bytes for external memory
    CHECK_CUSPARSE(
        cusparseSpGEMM_compute(handle, opA, opB,
                               &alpha, matA, matB, &beta, matC,
                               computeType, CUSPARSE_SPGEMM_DEFAULT,
                               spgemmDesc, &bufferSize2, NULL) )
    printf("bufferSize2 = %zu\n", bufferSize2);
    // size_t myBufferSize2 = (a_nnz + b_nnz) * sizeof(T) * 30;
    // size_t myBufferSize2 = m * n * sizeof(T);
    size_t myBufferSize2 = 0;
    if (bufferSize2 > 1e6) {
        myBufferSize2 = bufferSize2 / 8;
    } else {
        myBufferSize2 = bufferSize2;
    }
    printf("myBufferSize2 = %zu\n", myBufferSize2);
    if (bufsz2 < myBufferSize2) {
        if (bufsz2 > 0) {
            CHECK_CUDA( cudaFree(*buf2) )
        }
        CHECK_CUDA( cudaMalloc((void**) buf2, myBufferSize2) )
    }
    // CHECK_CUDA( cudaMalloc((void**) &dBuffer2, myBufferSize2) )
    // compute the intermediate product of A * B
    CHECK_CUSPARSE( cusparseSpGEMM_compute(handle, opA, opB,
                                           &alpha, matA, matB, &beta, matC,
                                           computeType, CUSPARSE_SPGEMM_DEFAULT,
                                           spgemmDesc, &myBufferSize2, *buf2) )
    // get matrix C non-zero entries C_nnz1
    int64_t C_num_rows1, C_num_cols1, C_nnz1;
    CHECK_CUSPARSE( cusparseSpMatGetSize(matC, &C_num_rows1, &C_num_cols1,
                                         &C_nnz1) )                               
    // allocate matrix C
    c_nnz = C_nnz1;

    if (rowsp1_bufsz < C_num_rows1 + 1) {
        if (rowsp1_bufsz > 0) {
            CHECK_CUDA( cudaFree(*c_indptr_buf) )
        }
        CHECK_CUDA( cudaMalloc((void**) c_indptr_buf, (C_num_rows1 + 1) * sizeof(int)) )
        rowsp1_bufsz = C_num_rows1 + 1;
    }
    if (nnz_bufsz < C_nnz1) {
        if (nnz_bufsz > 0) {
            CHECK_CUDA( cudaFree(*c_indices_buf) )
            CHECK_CUDA( cudaFree(*c_data_buf) )
        }
        CHECK_CUDA( cudaMalloc((void**) c_indices_buf, C_nnz1 * sizeof(int)) )
        CHECK_CUDA( cudaMalloc((void**) c_data_buf, C_nnz1 * sizeof(T)) )
        nnz_bufsz = C_nnz1;
    }

    // NOTE: if 'beta' != 0, the values of C must be update after the allocation
    //       of dC_values, and before the call of cusparseSpGEMM_copy

    // update matC with the new pointers
    CHECK_CUSPARSE(
        cusparseCsrSetPointers(matC, *c_indptr_buf, *c_indices_buf, *c_data_buf) )

    // if beta != 0, cusparseSpGEMM_copy reuses/updates the values of dC_values

    // copy the final products to the matrix C
    CHECK_CUSPARSE(
        cusparseSpGEMM_copy(handle, opA, opB,
                            &alpha, matA, matB, &beta, matC,
                            computeType, CUSPARSE_SPGEMM_DEFAULT, spgemmDesc) )

    // destroy matrix/vector descriptors
    CHECK_CUSPARSE( cusparseSpGEMM_destroyDescr(spgemmDesc) )
    CHECK_CUSPARSE( cusparseDestroySpMat(matA) )
    CHECK_CUSPARSE( cusparseDestroySpMat(matB) )
    CHECK_CUSPARSE( cusparseDestroySpMat(matC) )
    CHECK_CUSPARSE( cusparseDestroy(handle) )

    return EXIT_SUCCESS;
}