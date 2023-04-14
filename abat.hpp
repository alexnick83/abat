#include "utilities.hpp"

template<class T>
int abat(int64_t n, int64_t a_nnz, T *a_data, int32_t *a_indices, int32_t *a_indptr, T *B, T *buffer, size_t &work_sz, void **work_buf) {

    T alpha = (T) 1.0;
    T beta = (T) 0.0;

    cudaDataType compute_type;
    if (std::is_same<T, float>::value) {
        compute_type = CUDA_R_32F;
    } else if (std::is_same<T, double>::value) {
        compute_type = CUDA_R_64F;
    } else if (std::is_same<T, std::complex<float>>::value) {
        compute_type = CUDA_C_32F;
    } else if (std::is_same<T, std::complex<double>>::value) {
        compute_type = CUDA_C_64F;
    } else {
        printf("Unsupported data type\n");
        return EXIT_FAILURE;
    }

    cusparseHandle_t handle = nullptr;
    CHECK_CUSPARSE( cusparseCreate(&handle) )

    // A_csr x B_dense = AB_dense in row-major format
    cusparseSpMatDescr_t matA;
    cusparseDnMatDescr_t matB, matAB;
    CHECK_CUSPARSE( cusparseCreateCsr(&matA, n, n, a_nnz,
                                      a_indptr, a_indices, a_data,
                                      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                      CUSPARSE_INDEX_BASE_ZERO, compute_type) )
    CHECK_CUSPARSE( cusparseCreateDnMat(&matB, n, n, n, B, compute_type, CUSPARSE_ORDER_COL) )
    CHECK_CUSPARSE( cusparseCreateDnMat(&matAB, n, n, n, buffer, compute_type, CUSPARSE_ORDER_COL) )

    // Allocate an external buffer if needed
    size_t bufferSize = 0;
    CHECK_CUSPARSE( cusparseSpMM_bufferSize(
                                 handle,
                                 CUSPARSE_OPERATION_TRANSPOSE,
                                 CUSPARSE_OPERATION_NON_TRANSPOSE,
                                 &alpha, matA, matB, &beta, matAB, compute_type,
                                 CUSPARSE_SPMM_ALG_DEFAULT, &bufferSize) )
    printf("Requested buffer size: %zu\n", bufferSize);
    if (bufferSize > work_sz) {
        if (work_sz > 0) {
            cudaFree(*work_buf);
        }
        CHECK_CUDA( cudaMalloc(work_buf, bufferSize) )
        work_sz = bufferSize;
    }

    // Execute SpMM
    CHECK_CUSPARSE( cusparseSpMM(handle,
                                 CUSPARSE_OPERATION_TRANSPOSE,
                                 CUSPARSE_OPERATION_NON_TRANSPOSE,
                                 &alpha, matA, matB, &beta, matAB, compute_type,
                                 CUSPARSE_SPMM_ALG_DEFAULT, *work_buf) )

    CHECK_CUSPARSE( cusparseDestroyDnMat(matB) )
    CHECK_CUSPARSE( cusparseDestroyDnMat(matAB) )

    // A_csr x AB^T_dense = C^T_dense but in column-major format
    CHECK_CUSPARSE( cusparseCreateDnMat(&matB, n, n, n, buffer, compute_type, CUSPARSE_ORDER_COL) )
    CHECK_CUSPARSE( cusparseCreateDnMat(&matAB, n, n, n, B, compute_type, CUSPARSE_ORDER_ROW) )

    // Allocate an external buffer if needed
    bufferSize = 0;
    CHECK_CUSPARSE( cusparseSpMM_bufferSize(
                                 handle,
                                 CUSPARSE_OPERATION_TRANSPOSE,
                                 CUSPARSE_OPERATION_TRANSPOSE,
                                 &alpha, matA, matB, &beta, matAB, compute_type,
                                 CUSPARSE_SPMM_ALG_DEFAULT, &bufferSize) )
    printf("Requested buffer size: %zu\n", bufferSize);
    if (bufferSize > work_sz) {
        if (work_sz > 0) {
            cudaFree(*work_buf);
        }
        CHECK_CUDA( cudaMalloc(work_buf, bufferSize) )
        work_sz = bufferSize;
    }

    // Execute SpMM
    CHECK_CUSPARSE( cusparseSpMM(handle,
                                 CUSPARSE_OPERATION_TRANSPOSE,
                                 CUSPARSE_OPERATION_TRANSPOSE,
                                 &alpha, matA, matB, &beta, matAB, compute_type,
                                 CUSPARSE_SPMM_ALG_DEFAULT, *work_buf) )


    // Destroy matrix descriptors
    CHECK_CUSPARSE( cusparseDestroySpMat(matA) )
    CHECK_CUSPARSE( cusparseDestroyDnMat(matB) )
    CHECK_CUSPARSE( cusparseDestroyDnMat(matAB) )

    return EXIT_SUCCESS;
}