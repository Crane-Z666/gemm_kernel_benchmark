#include <dlfcn.h>
#include <CL/cl.h>
#include <stdio.h>

// =================================================================================================
// 0. 定义所有 OpenCL 函数指针的类型
// =================================================================================================
typedef cl_int (CL_API_CALL *clGetPlatformIDs_fn)(cl_uint, cl_platform_id*, cl_uint*);
typedef cl_int (CL_API_CALL *clGetPlatformInfo_fn)(cl_platform_id, cl_platform_info, size_t, void*, size_t*);
typedef cl_int (CL_API_CALL *clGetDeviceIDs_fn)(cl_platform_id, cl_device_type, cl_uint, cl_device_id*, cl_uint*);
typedef cl_int (CL_API_CALL *clGetDeviceInfo_fn)(cl_device_id, cl_device_info, size_t, void*, size_t*);
typedef cl_int (CL_API_CALL *clRetainDevice_fn)(cl_device_id);
typedef cl_int (CL_API_CALL *clReleaseDevice_fn)(cl_device_id);
typedef cl_context (CL_API_CALL *clCreateContext_fn)(const cl_context_properties*, cl_uint, const cl_device_id*, void (CL_CALLBACK*)(const char*, const void*, size_t, void*), void*, cl_int*);
typedef cl_int (CL_API_CALL *clReleaseContext_fn)(cl_context);
typedef cl_command_queue (CL_API_CALL *clCreateCommandQueue_fn)(cl_context, cl_device_id, cl_command_queue_properties, cl_int*);
typedef cl_int (CL_API_CALL *clReleaseCommandQueue_fn)(cl_command_queue);
typedef cl_program (CL_API_CALL *clCreateProgramWithSource_fn)(cl_context, cl_uint, const char**, const size_t*, cl_int*);
typedef cl_int (CL_API_CALL *clGetProgramInfo_fn)(cl_program, cl_program_info, size_t, void*, size_t*);
typedef cl_int (CL_API_CALL *clGetProgramBuildInfo_fn)(cl_program, cl_device_id, cl_program_build_info, size_t, void*, size_t*);
typedef cl_int (CL_API_CALL *clReleaseProgram_fn)(cl_program);
typedef cl_int (CL_API_CALL *clBuildProgram_fn)(cl_program, cl_uint, const cl_device_id*, const char*, void (CL_CALLBACK*)(cl_program, void*), void*);
typedef cl_kernel (CL_API_CALL *clCreateKernel_fn)(cl_program, const char*, cl_int*);
typedef cl_int (CL_API_CALL *clReleaseKernel_fn)(cl_kernel);
typedef cl_mem (CL_API_CALL *clCreateBuffer_fn)(cl_context, cl_mem_flags, size_t, void*, cl_int*);
typedef cl_mem (CL_API_CALL *clCreateImage_fn)(cl_context, cl_mem_flags, const cl_image_format*, const cl_image_desc*, void*, cl_int*);
typedef cl_int (CL_API_CALL *clReleaseMemObject_fn)(cl_mem);
typedef cl_int (CL_API_CALL *clSetKernelArg_fn)(cl_kernel, cl_uint, size_t, const void*);
typedef cl_int (CL_API_CALL *clEnqueueNDRangeKernel_fn)(cl_command_queue, cl_kernel, cl_uint, const size_t*, const size_t*, const size_t*, cl_uint, const cl_event*, cl_event*);
typedef cl_int (CL_API_CALL *clFinish_fn)(cl_command_queue);
typedef cl_int (CL_API_CALL *clEnqueueReadBuffer_fn)(cl_command_queue, cl_mem, cl_bool, size_t, size_t, void*, cl_uint, const cl_event*, cl_event*);

// (新增) 为事件分析功能添加的函数类型定义
typedef cl_int (CL_API_CALL *clWaitForEvents_fn)(cl_uint, const cl_event*);
typedef cl_int (CL_API_CALL *clReleaseEvent_fn)(cl_event);
typedef cl_int (CL_API_CALL *clGetEventProfilingInfo_fn)(cl_event, cl_profiling_info, size_t, void*, size_t*);


// =================================================================================================
// 1. 定义所有 OpenCL 函数的指针变量
// =================================================================================================
#define OPENCL_FUNC_PTR_DEF(name) name##_fn p_##name = NULL

OPENCL_FUNC_PTR_DEF(clGetPlatformIDs);
OPENCL_FUNC_PTR_DEF(clGetPlatformInfo);
OPENCL_FUNC_PTR_DEF(clGetDeviceIDs);
OPENCL_FUNC_PTR_DEF(clGetDeviceInfo);
OPENCL_FUNC_PTR_DEF(clRetainDevice);
OPENCL_FUNC_PTR_DEF(clReleaseDevice);
OPENCL_FUNC_PTR_DEF(clCreateContext);
OPENCL_FUNC_PTR_DEF(clReleaseContext);
OPENCL_FUNC_PTR_DEF(clCreateCommandQueue);
OPENCL_FUNC_PTR_DEF(clReleaseCommandQueue);
OPENCL_FUNC_PTR_DEF(clCreateProgramWithSource);
OPENCL_FUNC_PTR_DEF(clGetProgramInfo);
OPENCL_FUNC_PTR_DEF(clGetProgramBuildInfo);
OPENCL_FUNC_PTR_DEF(clReleaseProgram);
OPENCL_FUNC_PTR_DEF(clBuildProgram);
OPENCL_FUNC_PTR_DEF(clCreateKernel);
OPENCL_FUNC_PTR_DEF(clReleaseKernel);
OPENCL_FUNC_PTR_DEF(clCreateBuffer);
OPENCL_FUNC_PTR_DEF(clCreateImage);
OPENCL_FUNC_PTR_DEF(clReleaseMemObject);
OPENCL_FUNC_PTR_DEF(clSetKernelArg);
OPENCL_FUNC_PTR_DEF(clEnqueueNDRangeKernel);
OPENCL_FUNC_PTR_DEF(clFinish);
OPENCL_FUNC_PTR_DEF(clEnqueueReadBuffer);

// (新增) 为事件分析功能添加的函数指针变量
OPENCL_FUNC_PTR_DEF(clWaitForEvents);
OPENCL_FUNC_PTR_DEF(clReleaseEvent);
OPENCL_FUNC_PTR_DEF(clGetEventProfilingInfo);


// =================================================================================================
// 2. 实现 initOpenCL 函数，用来加载动态库并获取所有函数地址
// =================================================================================================
#define LOAD_OPENCL_FUNC(name) \
    p_##name = (name##_fn)dlsym(handle, #name); \
    if (!p_##name) { \
        fprintf(stderr, "Failed to load OpenCL function: %s\n", #name); \
        dlclose(handle); \
        return 0; \
    }

// 这个函数必须被主程序调用一次
extern "C" int initOpenCL() {
    // 在 Android 上，OpenCL 库通常是 libOpenCL.so
    void* handle = dlopen("libOpenCL.so", RTLD_LAZY);
    if (!handle) {
        // 有些设备可能是别的名字
        handle = dlopen("libGLES_mali.so", RTLD_LAZY);
    }
    if (!handle) {
        fprintf(stderr, "Failed to open libOpenCL.so or libGLES_mali.so\n");
        return 0;
    }

    LOAD_OPENCL_FUNC(clGetPlatformIDs);
    LOAD_OPENCL_FUNC(clGetPlatformInfo);
    LOAD_OPENCL_FUNC(clGetDeviceIDs);
    LOAD_OPENCL_FUNC(clGetDeviceInfo);
    LOAD_OPENCL_FUNC(clRetainDevice);
    LOAD_OPENCL_FUNC(clReleaseDevice);
    LOAD_OPENCL_FUNC(clCreateContext);
    LOAD_OPENCL_FUNC(clReleaseContext);
    LOAD_OPENCL_FUNC(clCreateCommandQueue);
    LOAD_OPENCL_FUNC(clReleaseCommandQueue);
    LOAD_OPENCL_FUNC(clCreateProgramWithSource);
    LOAD_OPENCL_FUNC(clGetProgramInfo);
    LOAD_OPENCL_FUNC(clGetProgramBuildInfo);
    LOAD_OPENCL_FUNC(clReleaseProgram);
    LOAD_OPENCL_FUNC(clBuildProgram);
    LOAD_OPENCL_FUNC(clCreateKernel);
    LOAD_OPENCL_FUNC(clReleaseKernel);
    LOAD_OPENCL_FUNC(clCreateBuffer);
    LOAD_OPENCL_FUNC(clCreateImage);
    LOAD_OPENCL_FUNC(clReleaseMemObject);
    LOAD_OPENCL_FUNC(clSetKernelArg);
    LOAD_OPENCL_FUNC(clEnqueueNDRangeKernel);
    LOAD_OPENCL_FUNC(clFinish);
    LOAD_OPENCL_FUNC(clEnqueueReadBuffer);

    // (新增) 加载事件分析功能所需的函数
    LOAD_OPENCL_FUNC(clWaitForEvents);
    LOAD_OPENCL_FUNC(clReleaseEvent);
    LOAD_OPENCL_FUNC(clGetEventProfilingInfo);

    return 1; // 成功
}


// =================================================================================================
// 3. 实现所有 OpenCL API 的 "桩" 函数 (Stub Functions)
// =================================================================================================
extern "C" {

cl_int clGetPlatformIDs(cl_uint num_entries, cl_platform_id* platforms, cl_uint* num_platforms) {
    if (!p_clGetPlatformIDs) return CL_INVALID_PLATFORM;
    return p_clGetPlatformIDs(num_entries, platforms, num_platforms);
}

cl_int clGetPlatformInfo(cl_platform_id platform, cl_platform_info param_name, size_t param_value_size, void* param_value, size_t* param_value_size_ret) {
    if (!p_clGetPlatformInfo) return CL_INVALID_PLATFORM;
    return p_clGetPlatformInfo(platform, param_name, param_value_size, param_value, param_value_size_ret);
}

cl_int clGetDeviceIDs(cl_platform_id platform, cl_device_type device_type, cl_uint num_entries, cl_device_id* devices, cl_uint* num_devices) {
    if (!p_clGetDeviceIDs) return CL_INVALID_PLATFORM;
    return p_clGetDeviceIDs(platform, device_type, num_entries, devices, num_devices);
}

cl_int clGetDeviceInfo(cl_device_id device, cl_device_info param_name, size_t param_value_size, void* param_value, size_t* param_value_size_ret) {
    if (!p_clGetDeviceInfo) return CL_INVALID_DEVICE;
    return p_clGetDeviceInfo(device, param_name, param_value_size, param_value, param_value_size_ret);
}

cl_int clRetainDevice(cl_device_id device) {
    if (!p_clRetainDevice) return CL_INVALID_DEVICE;
    return p_clRetainDevice(device);
}

cl_int clReleaseDevice(cl_device_id device) {
    if (!p_clReleaseDevice) return CL_INVALID_DEVICE;
    return p_clReleaseDevice(device);
}

cl_context clCreateContext(const cl_context_properties* properties, cl_uint num_devices, const cl_device_id* devices, void (CL_CALLBACK* pfn_notify)(const char*, const void*, size_t, void*), void* user_data, cl_int* errcode_ret) {
    if (!p_clCreateContext) {
        if (errcode_ret) *errcode_ret = CL_INVALID_PLATFORM;
        return NULL;
    }
    return p_clCreateContext(properties, num_devices, devices, pfn_notify, user_data, errcode_ret);
}

cl_int clReleaseContext(cl_context context) {
    if (!p_clReleaseContext) return CL_INVALID_CONTEXT;
    return p_clReleaseContext(context);
}

cl_command_queue clCreateCommandQueue(cl_context context, cl_device_id device, cl_command_queue_properties properties, cl_int* errcode_ret) {
    if (!p_clCreateCommandQueue) {
        if (errcode_ret) *errcode_ret = CL_INVALID_CONTEXT;
        return NULL;
    }
    return p_clCreateCommandQueue(context, device, properties, errcode_ret);
}

cl_int clReleaseCommandQueue(cl_command_queue command_queue) {
    if (!p_clReleaseCommandQueue) return CL_INVALID_COMMAND_QUEUE;
    return p_clReleaseCommandQueue(command_queue);
}

cl_program clCreateProgramWithSource(cl_context context, cl_uint count, const char** strings, const size_t* lengths, cl_int* errcode_ret) {
    if (!p_clCreateProgramWithSource) {
        if (errcode_ret) *errcode_ret = CL_INVALID_CONTEXT;
        return NULL;
    }
    return p_clCreateProgramWithSource(context, count, strings, lengths, errcode_ret);
}

cl_int clGetProgramInfo(cl_program program, cl_program_info param_name, size_t param_value_size, void* param_value, size_t* param_value_size_ret) {
    if (!p_clGetProgramInfo) return CL_INVALID_PROGRAM;
    return p_clGetProgramInfo(program, param_name, param_value_size, param_value, param_value_size_ret);
}

cl_int clGetProgramBuildInfo(cl_program program, cl_device_id device, cl_program_build_info param_name, size_t param_value_size, void* param_value, size_t* param_value_size_ret) {
    if (!p_clGetProgramBuildInfo) return CL_INVALID_PROGRAM;
    return p_clGetProgramBuildInfo(program, device, param_name, param_value_size, param_value, param_value_size_ret);
}

cl_int clReleaseProgram(cl_program program) {
    if (!p_clReleaseProgram) return CL_INVALID_PROGRAM;
    return p_clReleaseProgram(program);
}

cl_int clBuildProgram(cl_program program, cl_uint num_devices, const cl_device_id* device_list, const char* options, void (CL_CALLBACK* pfn_notify)(cl_program, void*), void* user_data) {
    if (!p_clBuildProgram) return CL_INVALID_PROGRAM;
    return p_clBuildProgram(program, num_devices, device_list, options, pfn_notify, user_data);
}

cl_kernel clCreateKernel(cl_program program, const char* kernel_name, cl_int* errcode_ret) {
    if (!p_clCreateKernel) {
        if (errcode_ret) *errcode_ret = CL_INVALID_PROGRAM;
        return NULL;
    }
    return p_clCreateKernel(program, kernel_name, errcode_ret);
}

cl_int clReleaseKernel(cl_kernel kernel) {
    if (!p_clReleaseKernel) return CL_INVALID_KERNEL;
    return p_clReleaseKernel(kernel);
}

cl_mem clCreateBuffer(cl_context context, cl_mem_flags flags, size_t size, void* host_ptr, cl_int* errcode_ret) {
    if (!p_clCreateBuffer) {
        if (errcode_ret) *errcode_ret = CL_INVALID_CONTEXT;
        return NULL;
    }
    return p_clCreateBuffer(context, flags, size, host_ptr, errcode_ret);
}

cl_mem clCreateImage(cl_context context, cl_mem_flags flags, const cl_image_format* image_format, const cl_image_desc* image_desc, void* host_ptr, cl_int* errcode_ret) {
    if (!p_clCreateImage) {
        if (errcode_ret) *errcode_ret = CL_INVALID_CONTEXT;
        return NULL;
    }
    return p_clCreateImage(context, flags, image_format, image_desc, host_ptr, errcode_ret);
}

cl_int clReleaseMemObject(cl_mem memobj) {
    if (!p_clReleaseMemObject) return CL_INVALID_MEM_OBJECT;
    return p_clReleaseMemObject(memobj);
}

cl_int clSetKernelArg(cl_kernel kernel, cl_uint arg_index, size_t arg_size, const void* arg_value) {
    if (!p_clSetKernelArg) return CL_INVALID_KERNEL;
    return p_clSetKernelArg(kernel, arg_index, arg_size, arg_value);
}

cl_int clEnqueueNDRangeKernel(cl_command_queue command_queue, cl_kernel kernel, cl_uint work_dim, const size_t* global_work_offset, const size_t* global_work_size, const size_t* local_work_size, cl_uint num_events_in_wait_list, const cl_event* event_wait_list, cl_event* event) {
    if (!p_clEnqueueNDRangeKernel) return CL_INVALID_COMMAND_QUEUE;
    return p_clEnqueueNDRangeKernel(command_queue, kernel, work_dim, global_work_offset, global_work_size, local_work_size, num_events_in_wait_list, event_wait_list, event);
}

cl_int clFinish(cl_command_queue command_queue) {
    if (!p_clFinish) return CL_INVALID_COMMAND_QUEUE;
    return p_clFinish(command_queue);
}

cl_int clEnqueueReadBuffer(cl_command_queue command_queue, cl_mem buffer, cl_bool blocking_read, size_t offset, size_t size, void* ptr, cl_uint num_events_in_wait_list, const cl_event* event_wait_list, cl_event* event) {
    if (!p_clEnqueueReadBuffer) return CL_INVALID_COMMAND_QUEUE;
    return p_clEnqueueReadBuffer(command_queue, buffer, blocking_read, offset, size, ptr, num_events_in_wait_list, event_wait_list, event);
}

// (新增) 实现事件分析功能所需的桩函数
cl_int clWaitForEvents(cl_uint num_events, const cl_event* event_list) {
    if (!p_clWaitForEvents) return CL_INVALID_EVENT;
    return p_clWaitForEvents(num_events, event_list);
}

cl_int clReleaseEvent(cl_event event) {
    if (!p_clReleaseEvent) return CL_INVALID_EVENT;
    return p_clReleaseEvent(event);
}

cl_int clGetEventProfilingInfo(cl_event event, cl_profiling_info param_name, size_t param_value_size, void* param_value, size_t* param_value_size_ret) {
    if (!p_clGetEventProfilingInfo) return CL_INVALID_EVENT;
    return p_clGetEventProfilingInfo(event, param_name, param_value_size, param_value, param_value_size_ret);
}

} // extern "C"