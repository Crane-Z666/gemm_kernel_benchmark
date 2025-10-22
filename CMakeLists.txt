cmake_minimum_required(VERSION 3.10)
project(DirectRunKernelFromString)

include_directories(/home/junwei/LLM-Project/MNN2/MNN/3rd_party/OpenCLHeaders)

set(CMAKE_CXX_STANDARD 14)

# ===================================================================
# == 通用配置
# ===================================================================

# 当不进行 Android 交叉编译时 (即，在您的 Linux 桌面上编译)
if(NOT ANDROID)
    # 增加一个明确的诊断信息，如果执行到这里，我们就知道出错了
    message(FATAL_ERROR "Build configuration is NOT for Android! The ANDROID variable is not set. Stopping.")

    # 下面的代码在上面的 FATAL_ERROR 存在时不会被执行，但为了完整性保留
    find_package(OpenCL REQUIRED)
    if (NOT OpenCL_FOUND)
        message(FATAL_ERROR "OpenCL not found on host.")
    endif()

    add_executable(direct_run_kernel_from_string direct_run_kernel_from_string.cpp)
    add_executable(direct_run_kernel_from_string_W4A32 direct_run_kernel_from_string_W4A32.cpp)
    add_executable(direct_run_kernel_from_string_W4A16 direct_run_kernel_from_string_W4A16.cpp)

    target_include_directories(direct_run_kernel_from_string PRIVATE ${OpenCL_INCLUDE_DIRS})
    target_link_libraries(direct_run_kernel_from_string PRIVATE ${OpenCL_LIBRARIES})
    target_include_directories(direct_run_kernel_from_string_W4A32 PRIVATE ${OpenCL_INCLUDE_DIRS})
    target_link_libraries(direct_run_kernel_from_string_W4A32 PRIVATE ${OpenCL_LIBRARIES})
    target_include_directories(direct_run_kernel_from_string_W4A16 PRIVATE ${OpenCL_INCLUDE_DIRS})
    target_link_libraries(direct_run_kernel_from_string_W4A16 PRIVATE ${OpenCL_LIBRARIES})

# 当进行 Android 交叉编译时
else()
    # 增加一个成功的诊断信息
    message(STATUS "Build configuration is for Android. Using 'add_executable'.")

    add_executable(direct_run_kernel_from_string
        direct_run_kernel_from_string.cpp
        opencl_loader.cpp
    )
    add_executable(direct_run_kernel_from_string_W4A32
        direct_run_kernel_from_string_W4A32.cpp
        opencl_loader.cpp
    )
    add_executable(direct_run_kernel_from_string_W4A16
        direct_run_kernel_from_string_W4A16.cpp
        opencl_loader.cpp
    )
    add_executable(direct_run_kernel_from_string_W4A16_onlyGPU
        direct_run_kernel_from_string_W4A16_onlyGPU.cpp
        opencl_loader.cpp
    )
    add_executable(direct_run_kernel_from_string_W8A16_onlyGPU
        direct_run_kernel_from_string_W8A16_onlyGPU.cpp
        opencl_loader.cpp
    )

    target_link_libraries(direct_run_kernel_from_string PRIVATE dl)
    target_link_libraries(direct_run_kernel_from_string_W4A32 PRIVATE dl)
    target_link_libraries(direct_run_kernel_from_string_W4A16 PRIVATE dl)
    target_link_libraries(direct_run_kernel_from_string_W8A16_onlyGPU PRIVATE dl)
    target_link_libraries(direct_run_kernel_from_string_W4A16_onlyGPU PRIVATE dl)

endif()