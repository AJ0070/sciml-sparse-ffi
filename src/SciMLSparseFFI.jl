module SciMLSparseFFI

using Libdl
using LinearAlgebra
using SparseArrays

include("raw_bindings.jl")
include("CustomSparseMatrix.jl")
include("mul_overloads.jl")

export sciml_csr_f64,
       CSRPtr,
       csr_create,
       csr_destroy,
       csr_copy_data!,
       spmv_csr_f64!,
       spmv_csr_rvv_f64!,
       spmm_csr_f64!,
       spmm_csr_rvv_f64!,
    set_library_path!,
    CSparseMatrixCSR

end
