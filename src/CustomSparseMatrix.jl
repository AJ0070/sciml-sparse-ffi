mutable struct CSparseMatrixCSR <: AbstractSparseMatrix{Float64, Int32}
    n_rows::Int32
    n_cols::Int32
    row_ptr::Vector{Int32}
    col_idx::Vector{Int32}
    values::Vector{Float64}
    handle::CSRPtr
end

function CSparseMatrixCSR(
    n_rows::Integer,
    n_cols::Integer,
    row_ptr::Vector{Int32},
    col_idx::Vector{Int32},
    values::Vector{Float64}
)
    n_rows_i32 = Int32(n_rows)
    n_cols_i32 = Int32(n_cols)
    nnz_i32 = Int32(length(values))

    if length(col_idx) != length(values)
        throw(ArgumentError("col_idx and values lengths must match."))
    end

    if length(row_ptr) != Int(n_rows_i32) + 1
        throw(ArgumentError("row_ptr length must be n_rows + 1."))
    end

    if row_ptr[end] != nnz_i32
        throw(ArgumentError("row_ptr[end] must equal nnz."))
    end

    handle = csr_create(n_rows_i32, n_cols_i32, nnz_i32)
    try
        csr_copy_data!(handle, row_ptr, col_idx, values)
    catch
        csr_destroy(handle)
        rethrow()
    end

    matrix = CSparseMatrixCSR(n_rows_i32, n_cols_i32, row_ptr, col_idx, values, handle)
    finalizer(matrix) do obj
        csr_destroy(obj.handle)
    end

    return matrix
end

Base.size(A::CSparseMatrixCSR) = (Int(A.n_rows), Int(A.n_cols))
Base.eltype(::Type{CSparseMatrixCSR}) = Float64

function Base.getindex(A::CSparseMatrixCSR, i::Integer, j::Integer)
    if i < 1 || i > A.n_rows || j < 1 || j > A.n_cols
        throw(BoundsError(A, (i, j)))
    end

    row = Int32(i - 1)
    col = Int32(j - 1)
    start_idx = A.row_ptr[row + 1]
    stop_idx = A.row_ptr[row + 2]

    for k in (start_idx + 1):stop_idx
        if A.col_idx[k] == col
            return A.values[k]
        end
    end

    return 0.0
end
