#include "sciml_sparse_ffi.h"

#include <stddef.h>
#include <stdlib.h>
#include <string.h>

#if defined(__riscv_vector)
    #include <riscv_vector.h>
#endif

static int32_t spmv_csr_f64_scalar_impl(const sciml_csr_f64 *matrix, const double *x, double *y) {
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

static int32_t spmm_csr_f64_scalar_impl(
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
    return spmv_csr_f64_scalar_impl(matrix, x, y);
}

int32_t spmv_csr_rvv_f64(const sciml_csr_f64 *matrix, const double *x, double *y) {
#if defined(__riscv_vector)
    if (!matrix || !matrix->row_ptr || !matrix->col_idx || !matrix->values || !x || !y) {
        return -1;
    }

    for (int32_t i = 0; i < matrix->n_rows; ++i) {
        const int32_t start = matrix->row_ptr[i];
        const int32_t stop = matrix->row_ptr[i + 1];

        for (int32_t k = start; k < stop; ++k) {
            const int32_t col = matrix->col_idx[k];
            if (col < 0 || col >= matrix->n_cols) {
                return -2;
            }
        }

        double sum = 0.0;
        int32_t k = start;
        while (k < stop) {
            const size_t remaining = (size_t)(stop - k);
            const size_t vl = __riscv_vsetvl_e64m1(remaining);

            vfloat64m1_t v_values = __riscv_vle64_v_f64m1(&matrix->values[k], vl);
            vint32m1_t v_cols_i32 = __riscv_vle32_v_i32m1(&matrix->col_idx[k], vl);

            vuint32m1_t v_cols_u32 = __riscv_vreinterpret_v_i32m1_u32m1(v_cols_i32);
            vuint32m1_t v_offsets = __riscv_vsll_vx_u32m1(v_cols_u32, 3, vl);
            vfloat64m1_t v_x = __riscv_vluxei32_v_f64m1(x, v_offsets, vl);

            vfloat64m1_t v_acc = __riscv_vfmul_vv_f64m1(v_values, v_x, vl);

            vfloat64m1_t v_zero = __riscv_vfmv_v_f_f64m1(0.0, 1);
            vfloat64m1_t v_red = __riscv_vfredusum_vs_f64m1_f64m1(v_acc, v_zero, vl);
            sum += __riscv_vfmv_f_s_f64m1_f64(v_red);

            k += (int32_t)vl;
        }

        y[i] = sum;
    }

    return 0;
#else
    return spmv_csr_f64_scalar_impl(matrix, x, y);
#endif
}

int32_t spmm_csr_f64(
    const sciml_csr_f64 *matrix,
    const double *b,
    int32_t b_cols,
    double *c
) {
    return spmm_csr_f64_scalar_impl(matrix, b, b_cols, c);
}

int32_t spmm_csr_rvv_f64(
    const sciml_csr_f64 *matrix,
    const double *b,
    int32_t b_cols,
    double *c
) {
#if defined(__riscv_vector)
    if (!matrix || !matrix->row_ptr || !matrix->col_idx || !matrix->values || !b || !c || b_cols < 0) {
        return -1;
    }

    const int32_t m = matrix->n_rows;
    const int32_t n = matrix->n_cols;
    const ptrdiff_t b_stride = (ptrdiff_t)n * (ptrdiff_t)sizeof(double);
    const ptrdiff_t c_stride = (ptrdiff_t)m * (ptrdiff_t)sizeof(double);

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
            int32_t j = 0;
            while (j < b_cols) {
                const size_t remaining = (size_t)(b_cols - j);
                const size_t vl = __riscv_vsetvl_e64m1(remaining);

                const double *b_ptr = &b[(size_t)col + (size_t)j * (size_t)n];
                double *c_ptr = &c[(size_t)i + (size_t)j * (size_t)m];

                vfloat64m1_t v_b = __riscv_vlse64_v_f64m1(b_ptr, b_stride, vl);
                vfloat64m1_t v_c = __riscv_vlse64_v_f64m1(c_ptr, c_stride, vl);
                v_c = __riscv_vfmacc_vf_f64m1(v_c, a, v_b, vl);
                __riscv_vsse64_v_f64m1(c_ptr, c_stride, v_c, vl);

                j += (int32_t)vl;
            }
        }
    }

    return 0;
#else
    return spmm_csr_f64_scalar_impl(matrix, b, b_cols, c);
#endif
}
