import Base: *
import LinearAlgebra: mul!

function mul!(y::StridedVector{Float64}, A::CSparseMatrixCSR, x::StridedVector{Float64})
    m, n = size(A)
    if length(x) != n
        throw(DimensionMismatch("x has length $(length(x)) but expected $(n)."))
    end
    if length(y) != m
        throw(DimensionMismatch("y has length $(length(y)) but expected $(m)."))
    end

    spmv_csr_f64!(A.handle, x, y)
    return y
end

function mul!(C::StridedMatrix{Float64}, A::CSparseMatrixCSR, B::StridedMatrix{Float64})
    m, n = size(A)
    if size(B, 1) != n
        throw(DimensionMismatch("B has $(size(B, 1)) rows but expected $(n)."))
    end
    if size(C, 1) != m || size(C, 2) != size(B, 2)
        throw(DimensionMismatch("C must have size ($(m), $(size(B, 2)))."))
    end

    spmm_csr_f64!(A.handle, B, C)
    return C
end

function *(A::CSparseMatrixCSR, x::AbstractVector{Float64})
    x_work = x isa StridedVector{Float64} ? x : collect(x)
    y = Vector{Float64}(undef, size(A, 1))
    mul!(y, A, x_work)
    return y
end

function *(A::CSparseMatrixCSR, B::AbstractMatrix{Float64})
    B_work = B isa StridedMatrix{Float64} ? B : Matrix(B)
    C = Matrix{Float64}(undef, size(A, 1), size(B_work, 2))
    mul!(C, A, B_work)
    return C
end
