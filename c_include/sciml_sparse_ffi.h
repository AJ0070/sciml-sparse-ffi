#ifndef SCIML_SPARSE_FFI_H
#define SCIML_SPARSE_FFI_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct sciml_csr_f64 {
    int32_t n_rows;
    int32_t n_cols;
    int32_t nnz;
    int32_t *row_ptr;
    int32_t *col_idx;
    double *values;
} sciml_csr_f64;

sciml_csr_f64 *sciml_csr_f64_create(int32_t n_rows, int32_t n_cols, int32_t nnz);

void sciml_csr_f64_destroy(sciml_csr_f64 *matrix);

int32_t sciml_csr_f64_copy_data(
    sciml_csr_f64 *matrix,
    const int32_t *row_ptr,
    const int32_t *col_idx,
    const double *values
);

int32_t spmv_csr_f64(const sciml_csr_f64 *matrix, const double *x, double *y);

int32_t spmm_csr_f64(
    const sciml_csr_f64 *matrix,
    const double *b,
    int32_t b_cols,
    double *c
);

#ifdef __cplusplus
}
#endif

#endif