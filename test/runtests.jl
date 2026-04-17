using Test
using LinearAlgebra
using SparseArrays
using SciMLSparseFFI

function csr_to_sparse(row_ptr::Vector{Int32}, col_idx::Vector{Int32}, values::Vector{Float64}, m::Int, n::Int)
    rows = Int[]
    cols = Int[]
    vals = Float64[]

    for i in 1:m
        start_idx = Int(row_ptr[i]) + 1
        stop_idx = Int(row_ptr[i + 1])
        for k in start_idx:stop_idx
            push!(rows, i)
            push!(cols, Int(col_idx[k]) + 1)
            push!(vals, values[k])
        end
    end

    return sparse(rows, cols, vals, m, n)
end

@testset "SciMLSparseFFI" begin
    m = 3
    n = 4
    row_ptr = Int32[0, 2, 3, 5]
    col_idx = Int32[0, 3, 1, 0, 2]
    values = Float64[2.0, -1.0, 3.0, 4.0, 5.0]

    Aref = csr_to_sparse(row_ptr, col_idx, values, m, n)

    @testset "Raw FFI kernels" begin
        handle = csr_create(m, n, length(values))
        try
            csr_copy_data!(handle, row_ptr, col_idx, values)

            x = [1.0, 2.0, -1.0, 0.5]
            y = zeros(Float64, m)
            spmv_csr_f64!(handle, x, y)
            @test y ≈ Array(Aref * x)

            B = [
                1.0  0.0
                2.0 -1.0
                3.0  2.0
                4.0  1.0
            ]
            C = zeros(Float64, m, size(B, 2))
            spmm_csr_f64!(handle, B, C)
            @test C ≈ Matrix(Aref * B)
        finally
            csr_destroy(handle)
        end
    end

    @testset "CSparseMatrixCSR basics" begin
        A = CSparseMatrixCSR(m, n, row_ptr, col_idx, values)

        @test size(A) == (m, n)
        @test eltype(A) == Float64

        @test A[1, 1] == 2.0
        @test A[1, 4] == -1.0
        @test A[2, 2] == 3.0
        @test A[3, 3] == 5.0
        @test A[3, 2] == 0.0

        @test_throws BoundsError A[0, 1]
        @test_throws BoundsError A[1, 5]
    end

    @testset "mul! and operator bridge" begin
        A = CSparseMatrixCSR(m, n, row_ptr, col_idx, values)

        x = [1.5, -2.0, 0.25, 3.0]
        y = zeros(Float64, m)
        mul!(y, A, x)
        @test y ≈ Array(Aref * x)
        @test A * x ≈ Array(Aref * x)

        B = [
            0.5  1.0  0.0
            1.0 -1.5  2.0
           -2.0  0.0  3.0
            4.0  2.5 -1.0
        ]
        C = zeros(Float64, m, size(B, 2))
        mul!(C, A, B)
        @test C ≈ Matrix(Aref * B)
        @test A * B ≈ Matrix(Aref * B)

        @test_throws DimensionMismatch mul!(zeros(Float64, m), A, ones(Float64, n + 1))
        @test_throws DimensionMismatch mul!(zeros(Float64, m + 1), A, ones(Float64, n))
        @test_throws DimensionMismatch mul!(zeros(Float64, m, 2), A, ones(Float64, n + 1, 2))
        @test_throws DimensionMismatch mul!(zeros(Float64, m + 1, 2), A, ones(Float64, n, 2))
    end
end
