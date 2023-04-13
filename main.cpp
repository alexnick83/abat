#include "spmm.hpp"
#include <iostream>

#define dtype std::complex<double>

int main(void) {

    int rows = 6336;
    int cols = 6336;
    int bands = 111;
    // int rows = 8;
    // int cols = 8;
    // int bands = 3;

    // Generate random banded matrices
    dtype *a_data, *b_data;
    int *a_indices, *a_indptr, *b_indices, *b_indptr;
    int64_t a_nnz = generate_random_banded_matrix<dtype>(rows, cols, bands, &a_data, &a_indices, &a_indptr);
    int64_t b_nnz = generate_random_banded_matrix<dtype>(rows, cols, bands, &b_data, &b_indices, &b_indptr);

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

    // Copy data to GPU
    dtype *dev_a_data, *dev_b_data, *dev_c_data, *dev_d_data;
    int *dev_a_indices, *dev_a_indptr, *dev_b_indices, *dev_b_indptr, *dev_c_indices, *dev_c_indptr, *dev_d_indices, *dev_d_indptr;
    int64_t c_nnz, d_nnz;
    int64_t nnz_bufsz = 0;
    int rowsp1_bufsz = 0;
    void *buf1, *buf2;
    size_t buf1sz = 0, buf2sz = 0;
    cudaMemcpyHostToDevice_banded_matrix<dtype>(rows, cols, bands, a_data, a_indices, a_indptr, &dev_a_data, &dev_a_indices, &dev_a_indptr);
    cudaMemcpyHostToDevice_banded_matrix<dtype>(rows, cols, bands, b_data, b_indices, b_indptr, &dev_b_data, &dev_b_indices, &dev_b_indptr);
    banded_mm<dtype>('N', 'N', rows, cols, cols, a_nnz, b_nnz, c_nnz,
                     dev_a_data, dev_a_indices, dev_a_indptr,
                     dev_b_data, dev_b_indices, dev_b_indptr,
                     &dev_c_data, &dev_c_indices, &dev_c_indptr,
                     nnz_bufsz, rowsp1_bufsz,
                     buf1sz, &buf1, buf2sz, &buf2);
    banded_mm<dtype>('N', 'N', rows, cols, cols, c_nnz, a_nnz, d_nnz,
                     dev_c_data, dev_c_indices, dev_c_indptr,
                     dev_a_data, dev_a_indices, dev_a_indptr,
                     &dev_c_data, &dev_c_indices, &dev_c_indptr,
                     nnz_bufsz, rowsp1_bufsz,
                     buf1sz, &buf1, buf2sz, &buf2);

    std::cout << "C_nnz: " << d_nnz << std::endl;
    dtype *c_data = new dtype[d_nnz];
    int *c_indices = new int[d_nnz];
    int *c_indptr = new int[rows + 1];
    cudaMemcpyDeviceToHost_banded_matrix<dtype>(rows, cols, d_nnz, dev_c_data, dev_c_indices, dev_c_indptr, c_data, c_indices, c_indptr);

    // // Print result
    // std::cout << "C:" << std::endl;
    // for (int i = 0; i < rows; i++) {
    //     for (int j = 0; j < cols; j++) {
    //         bool found = false;
    //         for (int k = c_indptr[i]; k < c_indptr[i + 1]; k++) {
    //             if (c_indices[k] == j) {
    //                 std::cout << c_data[k] << " ";
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


    CHECK_CUDA(cudaFree(dev_a_data));
    CHECK_CUDA(cudaFree(dev_b_data));
    CHECK_CUDA(cudaFree(dev_c_data));
    // CHECK_CUDA(cudaFree(dev_d_data));
    CHECK_CUDA(cudaFree(dev_a_indices));
    CHECK_CUDA(cudaFree(dev_a_indptr));
    CHECK_CUDA(cudaFree(dev_b_indices));
    CHECK_CUDA(cudaFree(dev_b_indptr));
    CHECK_CUDA(cudaFree(dev_c_indices));
    CHECK_CUDA(cudaFree(dev_c_indptr));
    // CHECK_CUDA(cudaFree(dev_d_indices));
    // CHECK_CUDA(cudaFree(dev_d_indptr));
    delete[] a_data;
    delete[] b_data;
    delete[] c_data;
    delete[] a_indices;
    delete[] a_indptr;
    delete[] b_indices;
    delete[] b_indptr;
    delete[] c_indices;
    delete[] c_indptr;
}
