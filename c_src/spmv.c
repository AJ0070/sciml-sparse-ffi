#include "sciml_sparse_ffi.h"

#include <stdlib.h>
#include <string.h>

sciml_csr_f64 *sciml_csr_f64_create(int32_t n_rows, int32_t n_cols, int32_t nnz) {
    if (n_rows < 0 || n_cols < 0 || nnz < 0) {
        return NULL;
    }

    sciml_csr_f64 *matrix = (sciml_csr_f64 *)malloc(sizeof(sciml_csr_f64));
    if (!matrix) {
        return NULL;
    }

    matrix->n_rows = n_rows;
    matrix->n_cols = n_cols;
    matrix->nnz = nnz;
    matrix->row_ptr = (int32_t *)malloc((size_t)(n_rows + 1) * sizeof(int32_t));
    matrix->col_idx = (int32_t *)malloc((size_t)nnz * sizeof(int32_t));
    matrix->values = (double *)malloc((size_t)nnz * sizeof(double));

    if (!matrix->row_ptr || !matrix->col_idx || !matrix->values) {
        sciml_csr_f64_destroy(matrix);
        return NULL;
    }

    return matrix;
}

void sciml_csr_f64_destroy(sciml_csr_f64 *matrix) {
    if (!matrix) {
        return;
    }

    free(matrix->row_ptr);
    free(matrix->col_idx);
    free(matrix->values);
    free(matrix);
}

int32_t sciml_csr_f64_copy_data(
    sciml_csr_f64 *matrix,
    const int32_t *row_ptr,
    const int32_t *col_idx,
    const double *values
) {
    if (!matrix || !row_ptr || !col_idx || !values) {
        return -1;
    }

    if (row_ptr[matrix->n_rows] != matrix->nnz) {
        return -2;
    }

    memcpy(matrix->row_ptr, row_ptr, (size_t)(matrix->n_rows + 1) * sizeof(int32_t));
    memcpy(matrix->col_idx, col_idx, (size_t)matrix->nnz * sizeof(int32_t));
    memcpy(matrix->values, values, (size_t)matrix->nnz * sizeof(double));

    return 0;
}

int32_t spmv_csr_f64(const sciml_csr_f64 *matrix, const double *x, double *y) {
    if (!matrix || !matrix->row_ptr || !matrix->col_idx || !matrix->values || !x || !y) {
        return -1;
    }

    for (int32_t i = 0; i < matrix->n_rows; ++i) {
        const int32_t start = matrix->row_ptr[i];
        const int32_t stop = matrix->row_ptr[i + 1];
        double sum = 0.0;

        for (int32_t k = start; k < stop; ++k) {
            const int32_t col = matrix->col_idx[k];
            if (col < 0 || col >= matrix->n_cols) {
                return -2;
            }
            sum += matrix->values[k] * x[col];
        }

        y[i] = sum;
    }

    return 0;
}

int32_t spmm_csr_f64(
    const sciml_csr_f64 *matrix,
    const double *b,
    int32_t b_cols,
    double *c
) {
    if (!matrix || !matrix->row_ptr || !matrix->col_idx || !matrix->values || !b || !c || b_cols < 0) {
        return -1;
    }

    const int32_t m = matrix->n_rows;
    const int32_t n = matrix->n_cols;

    for (int32_t j = 0; j < b_cols; ++j) {
        for (int32_t i = 0; i < m; ++i) {
            c[(size_t)i + (size_t)j * (size_t)m] = 0.0;
        }
    }

    for (int32_t i = 0; i < m; ++i) {
        const int32_t start = matrix->row_ptr[i];
        const int32_t stop = matrix->row_ptr[i + 1];

        for (int32_t k = start; k < stop; ++k) {
            const int32_t col = matrix->col_idx[k];
            if (col < 0 || col >= n) {
                return -2;
            }

            const double a = matrix->values[k];
            for (int32_t j = 0; j < b_cols; ++j) {
                c[(size_t)i + (size_t)j * (size_t)m] +=
                    a * b[(size_t)col + (size_t)j * (size_t)n];
            }
        }
    }

    return 0;
}
