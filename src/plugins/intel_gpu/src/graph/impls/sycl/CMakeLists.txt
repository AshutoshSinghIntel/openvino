if(NOT OV_GPU_WITH_SYCL)
    return()
endif()

set(TARGET_NAME "openvino_intel_gpu_sycl_obj${OV_GPU_SUFFIX}")

set(SOURCES_OCL_RT
    "${CMAKE_CURRENT_SOURCE_DIR}/impl_example.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/fully_connected_xetla_int2.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/xetla/xetla_int2_gemv.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/xetla/xetla_int2_dpas.cpp")

if(OV_GPU_RT STREQUAL "SYCL")
    set(EXCLUDED_SOURCE_PATHS ${SOURCES_OCL_RT})
else()
    file(GLOB_RECURSE EXCLUDED_SOURCE_PATHS "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp")
    list(REMOVE_ITEM EXCLUDED_SOURCE_PATHS ${SOURCES_OCL_RT})
endif()

ov_gpu_add_backend_target(
    NAME ${TARGET_NAME}
    BYPASS
        EXCLUDED_SOURCE_PATHS ${EXCLUDED_SOURCE_PATHS}
)

add_sycl_to_target(TARGET ${TARGET_NAME})

# oneAPI's own <sycl/sycl.hpp> headers reference deprecated SYCL 1.2.1 aliases
# internally; under /WX that becomes a hard error unrelated to this target's code.
target_compile_options(${TARGET_NAME} PRIVATE -Wno-deprecated-declarations)

if(NOT DEFINED XETLA_ROOT)
    set(XETLA_ROOT "$ENV{XETLA_ROOT}")
endif()
if(XETLA_ROOT AND EXISTS "${XETLA_ROOT}/include/xetla.hpp")
    target_include_directories(${TARGET_NAME} SYSTEM PRIVATE "${XETLA_ROOT}/include")
    set_source_files_properties("${CMAKE_CURRENT_SOURCE_DIR}/xetla/xetla_int2_gemv.cpp"
                                "${CMAKE_CURRENT_SOURCE_DIR}/xetla/xetla_int2_dpas.cpp"
        PROPERTIES COMPILE_OPTIONS
            # icx's clang-cl driver mode rejects GNU-style -std=; use its -Q alias instead.
            # -Xsycl-target-backend is only consumed by the device-compilation pass, so the
            # host pass reports it (and Ninja's -QMD/-QMT/-QMF) as unused; /WX makes that fatal.
            "-Qstd=c++20;-Wno-unusable-partial-specialization;-Wno-unused-command-line-argument;-Xsycl-target-backend=-vc-codegen -doubleGRF")
else()
    message(FATAL_ERROR "XETLA_ROOT is not set or has no include/xetla.hpp; "
                        "needed by the XeTLA int2 FullyConnected impl")
endif()