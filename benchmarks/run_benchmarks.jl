using BenchmarkTools
using Random
using SparseArrays
using LinearAlgebra
using SciMLSparseFFI

function build_test_matrix(m::Int, n::Int, density::Float64)
    A_native = sprand(Float64, m, n, density)
    A_native = dropzeros!(A_native)

    rows, cols, vals = findnz(A_native)
    perm = sortperm((rows .- 1) .* n .+ (cols .- 1))
    rows = rows[perm]
    cols = cols[perm]
    vals = vals[perm]

    row_counts = zeros(Int32, m)
    for r in rows
        row_counts[r] += 1
    end

    row_ptr = Vector{Int32}(undef, m + 1)
    row_ptr[1] = 0
    for i in 1:m
        row_ptr[i + 1] = row_ptr[i] + row_counts[i]
    end

    col_idx = Int32.(cols .- 1)

    A_ffi = CSparseMatrixCSR(
        m,
        n,
        row_ptr,
        col_idx,
        vals
    )

    return A_native, A_ffi
end

function run_benchmarks(; m::Int = 10_000, n::Int = 10_000, density::Float64 = 5e-4, b_cols::Int = 32)
    Random.seed!(42)

    A_native, A_ffi = build_test_matrix(m, n, density)
    x = randn(n)
    y_native = zeros(m)
    y_ffi = similar(y_native)

    B = randn(n, b_cols)
    C_native = zeros(m, b_cols)
    C_ffi_scalar = similar(C_native)
    C_ffi_rvv = similar(C_native)

    println("Matrix dimensions: $(m)x$(n), density=$(density), nnz=$(nnz(A_native))")

    println("\nSpMV: SparseArrays mul! vs FFI mul!")
    @btime mul!($y_native, $A_native, $x)
    @btime mul!($y_ffi, $A_ffi, $x)

    println("\nSpMM: SparseArrays mul! vs FFI scalar vs FFI RVV-entry")
    @btime mul!($C_native, $A_native, $B)
    @btime spmm_csr_f64!($(A_ffi.handle), $B, $C_ffi_scalar)
    @btime spmm_csr_rvv_f64!($(A_ffi.handle), $B, $C_ffi_rvv)

    println("\nVerification")
    mul!(y_native, A_native, x)
    mul!(y_ffi, A_ffi, x)
    println("SpMV relative error: ", norm(y_native - y_ffi) / (norm(y_native) + eps()))

    mul!(C_native, A_native, B)
    spmm_csr_f64!(A_ffi.handle, B, C_ffi_scalar)
    spmm_csr_rvv_f64!(A_ffi.handle, B, C_ffi_rvv)
    println("SpMM scalar-FFI relative error: ", norm(C_native - C_ffi_scalar) / (norm(C_native) + eps()))
    println("SpMM RVV-entry relative error: ", norm(C_native - C_ffi_rvv) / (norm(C_native) + eps()))

    return nothing
end

run_benchmarks()
