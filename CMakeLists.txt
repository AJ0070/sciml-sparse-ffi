cmake_minimum_required(VERSION 3.16)
project(sciml_sparse_ffi VERSION 0.1.0 LANGUAGES C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)

add_library(sciml_sparse_ffi SHARED
    c_src/spmv.c
)

target_include_directories(sciml_sparse_ffi
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/c_include
)

target_compile_definitions(sciml_sparse_ffi
    PRIVATE
        SCIML_SPARSE_FFI_BUILD
)

if(MSVC)
    target_compile_options(sciml_sparse_ffi PRIVATE /O2)
else()
    target_compile_options(sciml_sparse_ffi PRIVATE -O3 -fPIC)
endif()

set_target_properties(sciml_sparse_ffi PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    OUTPUT_NAME "sciml_sparse_ffi"
)
