const _library_path = Ref{String}("")

function _default_library_path()
    root = normpath(joinpath(@__DIR__, ".."))
    candidates = (
        joinpath(root, "build", "libsciml_sparse_ffi.$(Libdl.dlext)"),
        joinpath(root, "libsciml_sparse_ffi.$(Libdl.dlext)")
    )

    for candidate in candidates
        if isfile(candidate)
            return candidate
        end
    end

    error("Unable to locate libsciml_sparse_ffi.$(Libdl.dlext). Build the C library first.")
end

function _libpath()
    if isempty(_library_path[])
        _library_path[] = _default_library_path()
    end
    return _library_path[]
end

function set_library_path!(path::AbstractString)
    _library_path[] = String(path)
    return _library_path[]
end

struct sciml_csr_f64
end

const CSRPtr = Ptr{sciml_csr_f64}

function _check_status(code::Cint, context::AbstractString)
    if code != 0
        error("$(context) failed with status code $(Int(code)).")
    end
    return nothing
end

function csr_create(n_rows::Integer, n_cols::Integer, nnz::Integer)
    ptr = @ccall _libpath().sciml_csr_f64_create(
        Int32(n_rows)::Int32,
        Int32(n_cols)::Int32,
        Int32(nnz)::Int32
    )::CSRPtr

    if ptr == C_NULL
        error("sciml_csr_f64_create returned NULL.")
    end

    return ptr
end

function csr_destroy(matrix::CSRPtr)
    @ccall _libpath().sciml_csr_f64_destroy(matrix::CSRPtr)::Cvoid
    return nothing
end

function csr_copy_data!(matrix::CSRPtr, row_ptr::Vector{Int32}, col_idx::Vector{Int32}, values::Vector{Float64})
    status = @ccall _libpath().sciml_csr_f64_copy_data(
        matrix::CSRPtr,
        row_ptr::Ptr{Int32},
        col_idx::Ptr{Int32},
        values::Ptr{Float64}
    )::Cint
    _check_status(status, "sciml_csr_f64_copy_data")
    return nothing
end

function spmv_csr_f64!(matrix::CSRPtr, x::StridedVector{Float64}, y::StridedVector{Float64})
    status = @ccall _libpath().spmv_csr_f64(
        matrix::CSRPtr,
        x::Ptr{Float64},
        y::Ptr{Float64}
    )::Cint
    _check_status(status, "spmv_csr_f64")
    return y
end

function spmm_csr_f64!(matrix::CSRPtr, b::StridedMatrix{Float64}, c::StridedMatrix{Float64})
    status = @ccall _libpath().spmm_csr_f64(
        matrix::CSRPtr,
        b::Ptr{Float64},
        Int32(size(b, 2))::Int32,
        c::Ptr{Float64}
    )::Cint
    _check_status(status, "spmm_csr_f64")
    return c
end
