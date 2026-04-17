# SciMLSparseFFI

`sciml-sparse-ffi` is an open-source Julia + C project that provides a sparse linear algebra backend for SciML-style workloads.

It implements core sparse kernels in C11 (CSR SpMV / SpMM), exposes an FFI-safe C API, and binds that API in Julia using `ccall` through a custom sparse matrix type:

- C backend: shared library (`libsciml_sparse_ffi.{so,dylib,dll}`)
- Julia wrapper: zero-copy pointer-level calls for in-place multiplication
- Matrix type: `CSparseMatrixCSR <: AbstractSparseMatrix{Float64,Int32}`

## Why this project?

This project targets a practical middle ground between:

- **native Julia ergonomics** (dispatch, `mul!`, `*`, `AbstractMatrix` interfaces), and
- **C-level kernel control** (predictable memory layout and FFI-safe ABI boundaries).

The result is a portable path to experiment with external sparse kernels while staying compatible with Julia/SciML solver interfaces.

## Current Features

- C11 CSR sparse matrix structure (`sciml_csr_f64`)
- C API for allocation, copy-in, destruction
- C kernels:
  - `spmv_csr_f64` (CSR × dense vector)
  - `spmm_csr_f64` (CSR × dense matrix)
- Julia raw FFI bindings (`@ccall`)
- Julia sparse wrapper type (`CSparseMatrixCSR`)
- Dispatch bridge:
  - `LinearAlgebra.mul!(y, A, x)`
  - `LinearAlgebra.mul!(C, A, B)`
  - `A * x`, `A * B`
- Benchmarks against `SparseArrays.jl`
- GitHub Actions CI/CD + CodeQL + Dependabot

## Repository Layout

```text
.
├── c_include/
│   └── sciml_sparse_ffi.h
├── c_src/
│   └── spmv.c
├── src/
│   ├── SciMLSparseFFI.jl
│   ├── raw_bindings.jl
│   ├── CustomSparseMatrix.jl
│   └── mul_overloads.jl
├── test/
│   └── runtests.jl
├── benchmarks/
│   └── run_benchmarks.jl
├── .github/
│   ├── dependabot.yml
│   └── workflows/
│       ├── ci.yml
│       ├── codeql.yml
│       └── release.yml
├── CMakeLists.txt
└── Project.toml
```

## Build the C Shared Library

### Requirements

- CMake >= 3.16
- C11-compatible compiler (`gcc`, `clang`, or MSVC)

### Build

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release
```

Expected artifact:

- Linux: `build/libsciml_sparse_ffi.so`
- macOS: `build/libsciml_sparse_ffi.dylib`
- Windows: `build/libsciml_sparse_ffi.dll` (or equivalent build output name)

## Julia Setup

### Requirements

- Julia >= 1.10

### Instantiate environment

```bash
julia --project=. -e "using Pkg; Pkg.instantiate()"
```

### Run tests

```bash
julia --project=. -e "using Pkg; Pkg.test()"
```

## Minimal Usage Example

```julia
using SciMLSparseFFI
using LinearAlgebra

# 2x2 matrix in CSR (0-based column indices for C backend)
# [2.0 3.0;
#  0.0 4.0]
row_ptr = Int32[0, 2, 3]
col_idx = Int32[0, 1, 1]
values  = Float64[2.0, 3.0, 4.0]

A = CSparseMatrixCSR(2, 2, row_ptr, col_idx, values)
x = [1.0, 2.0]

y = zeros(2)
mul!(y, A, x)
@show y  # [8.0, 8.0]

B = [1.0 2.0; 3.0 4.0]
C = zeros(2,2)
mul!(C, A, B)
@show C
```

## Notes on Indexing and Memory

- Julia-facing `getindex(A, i, j)` uses normal 1-based indexing.
- Internally, CSR data passed to C uses:
  - `row_ptr` as CSR row offsets
  - `col_idx` as **0-based** column indices (`Int32`)
- In-place multiplication paths (`mul!`) avoid allocations in the kernel call path.

## Benchmarks

Run:

```bash
julia --project=. benchmarks/run_benchmarks.jl
```

The benchmark script compares:

- `SparseArrays` native `mul!`
- FFI-backed `CSparseMatrixCSR` `mul!`

for both SpMV and SpMM workloads.

## CI/CD

This repository includes:

- **CI** (`.github/workflows/ci.yml`)
  - matrix build/test: Ubuntu, macOS, Windows
  - Julia versions: 1.10, 1.11
  - CMake build + Julia tests + optional coverage upload
- **CodeQL** (`.github/workflows/codeql.yml`)
  - scheduled and PR/push security analysis for C/C++
- **Release** (`.github/workflows/release.yml`)
  - on `v*` tags, builds platform artifacts and publishes release assets
- **Dependabot** (`.github/dependabot.yml`)
  - weekly updates for GitHub Actions dependencies

## Design Goals

- Stable C ABI and clean ownership boundaries across language runtime
- Explicitly typed FFI signatures (`double*`, `int32_t*`, `Ptr{T}`)
- Predictable, low-overhead data movement
- Compatibility with Julia linear algebra dispatch and SciML workflows

## Status

Current implementation is a solid prototype foundation for integrating external sparse kernels with Julia dispatch.

Likely next steps:

- expose low-level non-owning constructors for zero-copy CSR wrapping
- add direct `LinearSolve.jl` integration examples
- add thread-parallel C kernels and/or SIMD micro-optimizations
- add package docs (`Documenter.jl`) and API reference pages

## Contributing

Contributions are welcome.

Suggested workflow:

1. Fork and create a feature branch.
2. Run local checks:
   - `cmake -S . -B build && cmake --build build`
   - `julia --project=. -e "using Pkg; Pkg.test()"`
3. Open a PR with a focused change and clear benchmark or test evidence.

