#include "abat.hpp"
// #include "spmm.hpp"
#include <iostream>

#define dtype std::complex<double>

int main(void) {

    // int rows = 6336;
    // int cols = 6336;
    // int bands = 111;
    // int rows = 12000;
    // int cols = 12000;
    // int bands = 555;
    int rows = 2;
    int cols = 2;
    int bands = 3;

    // // Generate random banded matrices
    // dtype *a_data, *b_data;
    // int *a_indices, *a_indptr, *b_indices, *b_indptr;
    // int64_t a_nnz = generate_random_banded_matrix<dtype>(rows, cols, bands, &a_data, &a_indices, &a_indptr);
    // int64_t b_nnz = generate_random_banded_matrix<dtype>(rows, cols, bands, &b_data, &b_indices, &b_indptr);

    dtype *a_data, *dev_a_data, *b, *dev_b, *buffer, *host_buffer;
    int32_t *a_indices, *a_indptr, *dev_a_indices, *dev_a_indptr;
    size_t work_sz;
    void *work;
    int64_t a_nnz = generate_random_banded_matrix<dtype>(rows, cols, bands, &a_data, &a_indices, &a_indptr, true);
    cudaMemcpyHostToDevice_banded_matrix<dtype>(rows, cols, bands, a_data, a_indices, a_indptr, &dev_a_data, &dev_a_indices, &dev_a_indptr);
    generate_random_dense_matrix<dtype>(rows, cols, &b, true);
    CHECK_CUDA( cudaMalloc((void**)&dev_b, rows * cols * sizeof(dtype)) )
    CHECK_CUDA( cudaMemcpy(dev_b, b, rows * cols * sizeof(dtype), cudaMemcpyHostToDevice) )
    int64_t n = rows;
    CHECK_CUDA( cudaMalloc((void**)&buffer, rows * cols * sizeof(dtype)) )
    abat<dtype>(n, a_nnz, dev_a_data, dev_a_indices, dev_a_indptr, dev_b, buffer, work_sz, &work);
    host_buffer = new dtype[rows * cols];
    dtype *dev_b2;
    CHECK_CUDA( cudaMemcpy(host_buffer, buffer, rows * cols * sizeof(dtype), cudaMemcpyDeviceToHost) )
    std::cout << "B:" << std::endl;
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            std::cout << b[i + j * rows];
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
    std::cout << "AB:" << std::endl;
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            std::cout << host_buffer[i + j * rows];
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
    std::cout << "ABAT:" << std::endl;
    CHECK_CUDA( cudaMemcpy(host_buffer, dev_b, rows * cols * sizeof(dtype), cudaMemcpyDeviceToHost) )
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            std::cout << host_buffer[i + j * rows];
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
    CHECK_CUDA( cudaMemcpy(host_buffer, dev_b, rows * cols * sizeof(dtype), cudaMemcpyDeviceToHost) )
    CHECK_CUDA( cudaMalloc((void**)&dev_b2, rows * cols * sizeof(dtype)) )
    CHECK_CUDA( cudaMemcpy(dev_b2, b, rows * cols * sizeof(dtype), cudaMemcpyHostToDevice) )

    // Validation
    dtype *dev_a_dense, *a_dense;
    CHECK_CUDA( cudaMalloc((void**)&dev_a_dense, rows * cols * sizeof(dtype)) )
    csr_to_dense<dtype>(rows, cols, a_nnz, dev_a_data, dev_a_indices, dev_a_indptr, dev_a_dense);
    std::cout << "Created dense A" << std::endl;
    a_dense = new dtype[rows * cols];
    CHECK_CUDA( cudaMemcpy(a_dense, dev_a_dense, rows * cols * sizeof(dtype), cudaMemcpyDeviceToHost) )
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            std::cout << a_dense[i * rows + j];
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
    std::cout << "B:" << std::endl;
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            std::cout << b[i + j * rows];
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
    dtype *dev_ab_dense;
    dtype *host_buffer2 = new dtype[rows * cols];
    CHECK_CUDA( cudaMalloc((void**)&dev_ab_dense, rows * cols * sizeof(dtype)) )
    matmul<dtype>('N', 'N', rows, cols, cols, dev_b2, dev_a_dense, dev_ab_dense);
    std::cout << "Computed dense AB" << std::endl;
    CHECK_CUDA( cudaMemcpy(host_buffer2, dev_ab_dense, rows * cols * sizeof(dtype), cudaMemcpyDeviceToHost) )
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            std::cout << host_buffer2[i + j * rows];
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
    matmul<dtype>('N', 'N', rows, cols, cols, dev_a_dense, dev_ab_dense, buffer);
    std::cout << "Computed dense ABAT" << std::endl;
    CHECK_CUDA( cudaMemcpy(host_buffer2, buffer, rows * cols * sizeof(dtype), cudaMemcpyDeviceToHost) )

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            std::cout << host_buffer2[i * rows + j];
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;

    // Print matrices
    for (int i = 0; i < 10; i++) {
        std::cout << host_buffer[i] << " vs " << host_buffer2[i] << std::endl;
    }

    CHECK_CUDA( cudaFree(dev_a_data) )
    CHECK_CUDA( cudaFree(dev_a_indices) )
    CHECK_CUDA( cudaFree(dev_a_indptr) )
    CHECK_CUDA( cudaFree(dev_b) )
    CHECK_CUDA( cudaFree(buffer) )
    if (work_sz > 0) {
        CHECK_CUDA( cudaFree(work) )
    }
    delete [] a_data;
    delete [] a_indices;
    delete [] a_indptr;
    delete [] b;

    return EXIT_SUCCESS;


    // // Print matrices
    // // A
    // std::cout << "A:" << std::endl;
    // for (int i = 0; i < 8; i++) {
    //     for (int j = 0; j < 8; j++) {
    //         bool found = false;
    //         for (int k = a_indptr[i]; k < a_indptr[i + 1]; k++) {
    //             if (a_indices[k] == j) {
    //                 std::cout << a_data[k] << " ";
    //                 found = true;
    //                 break;
    //             }
    //         }
    //         if (!found) {
    //             std::cout << "0 ";
    //         }
    //     }
    //     std::cout << std::endl;
    // }
    // std::cout << std::endl;
    // // B
    // std::cout << "B:" << std::endl;
    // for (int i = 0; i < 8; i++) {
    //     for (int j = 0; j < 8; j++) {
    //         bool found = false;
    //         for (int k = b_indptr[i]; k < b_indptr[i + 1]; k++) {
    //             if (b_indices[k] == j) {
    //                 std::cout << b_data[k] << " ";
    //                 found = true;
    //                 break;
    //             }
    //         }
    //         if (!found) {
    //             std::cout << "0 ";
    //         }
    //     }
    //     std::cout << std::endl;
    // }

    // // Copy data to GPU
    // dtype *dev_a_data, *dev_b_data, *dev_c_data, *dev_d_data;
    // int *dev_a_indices, *dev_a_indptr, *dev_b_indices, *dev_b_indptr, *dev_c_indices, *dev_c_indptr, *dev_d_indices, *dev_d_indptr;
    // int64_t c_nnz, d_nnz;
    // int64_t nnz_bufsz = 0;
    // int rowsp1_bufsz = 0;
    // void *buf1, *buf2;
    // size_t buf1sz = 0, buf2sz = 0;
    // cudaMemcpyHostToDevice_banded_matrix<dtype>(rows, cols, bands, a_data, a_indices, a_indptr, &dev_a_data, &dev_a_indices, &dev_a_indptr);
    // cudaMemcpyHostToDevice_banded_matrix<dtype>(rows, cols, bands, b_data, b_indices, b_indptr, &dev_b_data, &dev_b_indices, &dev_b_indptr);
    // // banded_mm<dtype>('N', 'N', rows, cols, cols, a_nnz, b_nnz, c_nnz,
    // //                  dev_a_data, dev_a_indices, dev_a_indptr,
    // //                  dev_b_data, dev_b_indices, dev_b_indptr,
    // //                  &dev_c_data, &dev_c_indices, &dev_c_indptr,
    // //                  nnz_bufsz, rowsp1_bufsz,
    // //                  buf1sz, &buf1, buf2sz, &buf2);
    // // banded_mm<dtype>('N', 'N', rows, cols, cols, c_nnz, a_nnz, d_nnz,
    // //                  dev_c_data, dev_c_indices, dev_c_indptr,
    // //                  dev_a_data, dev_a_indices, dev_a_indptr,
    // //                  &dev_c_data, &dev_c_indices, &dev_c_indptr,
    // //                  nnz_bufsz, rowsp1_bufsz,
    // //                  buf1sz, &buf1, buf2sz, &buf2);

    // std::cout << "C_nnz: " << d_nnz << std::endl;
    // dtype *c_data = new dtype[d_nnz];
    // int *c_indices = new int[d_nnz];
    // int *c_indptr = new int[rows + 1];
    // cudaMemcpyDeviceToHost_banded_matrix<dtype>(rows, cols, d_nnz, dev_c_data, dev_c_indices, dev_c_indptr, c_data, c_indices, c_indptr);

    // // // Print result
    // // std::cout << "C:" << std::endl;
    // // for (int i = 0; i < rows; i++) {
    // //     for (int j = 0; j < cols; j++) {
    // //         bool found = false;
    // //         for (int k = c_indptr[i]; k < c_indptr[i + 1]; k++) {
    // //             if (c_indices[k] == j) {
    // //                 std::cout << c_data[k] << " ";
    // //                 found = true;
    // //                 break;
    // //             }
    // //         }
    // //         if (!found) {
    // //             std::cout << "0 ";
    // //         }
    // //     }
    // //     std::cout << std::endl;
    // // }


    // CHECK_CUDA(cudaFree(dev_a_data));
    // CHECK_CUDA(cudaFree(dev_b_data));
    // CHECK_CUDA(cudaFree(dev_c_data));
    // // CHECK_CUDA(cudaFree(dev_d_data));
    // CHECK_CUDA(cudaFree(dev_a_indices));
    // CHECK_CUDA(cudaFree(dev_a_indptr));
    // CHECK_CUDA(cudaFree(dev_b_indices));
    // CHECK_CUDA(cudaFree(dev_b_indptr));
    // CHECK_CUDA(cudaFree(dev_c_indices));
    // CHECK_CUDA(cudaFree(dev_c_indptr));
    // // CHECK_CUDA(cudaFree(dev_d_indices));
    // // CHECK_CUDA(cudaFree(dev_d_indptr));
    // delete[] a_data;
    // delete[] b_data;
    // delete[] c_data;
    // delete[] a_indices;
    // delete[] a_indptr;
    // delete[] b_indices;
    // delete[] b_indptr;
    // delete[] c_indices;
    // delete[] c_indptr;
}
