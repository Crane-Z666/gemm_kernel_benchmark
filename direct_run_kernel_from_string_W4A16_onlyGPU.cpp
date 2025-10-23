// direct_run_kernel_from_string_w4a16.cpp
#define CL_HPP_ENABLE_EXCEPTIONS
#define CL_HPP_TARGET_OPENCL_VERSION 120
#define CL_HPP_MINIMUM_OPENCL_VERSION 120
#include <CL/cl2.hpp>

#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <chrono>
#include <cmath>
#include <numeric> // For std::iota
#include <iomanip>
#include <algorithm> // For std::max

// =======================================================================
// == MNN Kernel Source String (Unchanged from original)                ==
// =======================================================================
// 内核源代码字符串保持不变。它通过不同的编译选项来支持 W4A16, W8A32, 以及我们现在的 W8A16。
namespace MNN {
const char* gemm_conv1x1_buf =
"#ifdef MNN_SUPPORT_FP16\n"
"#pragma OPENCL EXTENSION cl_khr_fp16 : enable\n"
"#endif\n"
"#define GLOBAL_SIZE_DIM2 "" __private int global_size_dim0,__private int global_size_dim1,\n"
"#define UNIFORM_BOUNDRY_CHECK(index0, index1) "" if(index0 >= global_size_dim0 || index1 >= global_size_dim1) { "" return; "" }\n"
"#define UCHAR4_TO_CHAR8(a, c) "" a.s0=(c.s0 >> 4)-8; a.s1=(c.s0 & 15)-8; a.s2=(c.s1 >> 4)-8; a.s3=(c.s1 & 15)-8; a.s4=(c.s2 >> 4)-8; a.s5=(c.s2 & 15)-8; a.s6=(c.s3 >> 4)-8; a.s7=(c.s3 & 15)-8;\n"
"__constant sampler_t SAMPLER=CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;\n"
"__kernel void inverse_quant_weight(GLOBAL_SIZE_DIM2\n"
" #ifdef USE_IMAGE\n"
" __read_only image2d_t weight,\n"
" #else\n"
" #if QUANT_BIT == 8\n"
" __global const char *weight,\n"
" #else\n"
" __global const uchar *weight,\n"
" #endif\n"
" #endif\n"
" __global const FLOAT *dequantScaleOffset,\n"
" __global FLOAT* output,\n"
" __private const int inputChannel,\n"
" __private const int inputChannel4Align,\n"
" __private const int outputChannelAlign,\n"
" __private const int outputChannel4Align,\n"
" __private const int blockDim,\n"
" __private const float coef){\n"
" const int x=get_global_id(0); //ic\n"
" const int y=get_global_id(1); //oc\n"
" UNIFORM_BOUNDRY_CHECK(x,y);\n"
" \n"
"#if QUANT_BIT == 4\n"
" const int ic=x << 2;\n"
" const int oc=y << 3;\n"
" const int output_offset=ic*outputChannelAlign+oc;\n"
" #ifdef ASYMMETRIC\n"
" COMPUTE_FLOAT8 scale,offset;\n"
" {\n"
" COMPUTE_FLOAT16 ScaleOffset=CONVERT_COMPUTE_FLOAT16(convert_float16(vload16(0,dequantScaleOffset+((ic/blockDim)*outputChannel4Align+oc)*2))/coef);\n"
" scale=ScaleOffset.s02468ace;\n"
" offset=ScaleOffset.s13579bdf;\n"
" }\n"
" #else\n"
" COMPUTE_FLOAT8 scale=CONVERT_COMPUTE_FLOAT8(vload8(0,dequantScaleOffset+(ic/blockDim)*outputChannel4Align+oc))/coef;\n"
" #endif\n"
" COMPUTE_FLOAT8 weights0,weights1,weights2,weights3;\n"
" {\n"
" #ifdef USE_IMAGE\n"
" uchar16 charWeightsInt40=as_uchar16(read_imagei(weight,SAMPLER,(int2)(x,y)));\n"
" #else\n"
" uchar16 charWeightsInt40=vload16(x,weight+y*inputChannel4Align*4);\n"
" #endif\n"
" char8 charWeights0;\n"
" #ifdef ASYMMETRIC\n"
" UCHAR4_TO_CHAR8(charWeights0,charWeightsInt40.s0123);\n"
" weights0=CONVERT_COMPUTE_FLOAT8(charWeights0)*scale+offset;\n"
" \n"
" UCHAR4_TO_CHAR8(charWeights0,charWeightsInt40.s4567);\n"
" weights1=ic+1 >= inputChannel ? 0 : CONVERT_COMPUTE_FLOAT8(charWeights0)*scale+offset;\n"
" \n"
" UCHAR4_TO_CHAR8(charWeights0,charWeightsInt40.s89ab);\n"
" weights2=ic+2 >= inputChannel ? 0 : CONVERT_COMPUTE_FLOAT8(charWeights0)*scale+offset;\n"
" \n"
" UCHAR4_TO_CHAR8(charWeights0,charWeightsInt40.scdef);\n"
" weights3=ic+3 >= inputChannel ? 0 : CONVERT_COMPUTE_FLOAT8(charWeights0)*scale+offset;\n"
" #else\n"
" UCHAR4_TO_CHAR8(charWeights0,charWeightsInt40.s0123);\n"
" weights0=CONVERT_COMPUTE_FLOAT8(charWeights0)*scale;\n"
" \n"
" UCHAR4_TO_CHAR8(charWeights0,charWeightsInt40.s4567);\n"
" weights1=ic+1 >= inputChannel ? 0 : CONVERT_COMPUTE_FLOAT8(charWeights0)*scale;\n"
" \n"
" UCHAR4_TO_CHAR8(charWeights0,charWeightsInt40.s89ab);\n"
" weights2=ic+2 >= inputChannel ? 0 : CONVERT_COMPUTE_FLOAT8(charWeights0)*scale;\n"
" \n"
" UCHAR4_TO_CHAR8(charWeights0,charWeightsInt40.scdef);\n"
" weights3=ic+3 >= inputChannel ? 0 : CONVERT_COMPUTE_FLOAT8(charWeights0)*scale;\n"
" #endif\n"
" }\n"
" vstore8(CONVERT_FLOAT8(weights0),0,output+output_offset);\n"
" vstore8(CONVERT_FLOAT8(weights1),0,output+output_offset+outputChannelAlign);\n"
" vstore8(CONVERT_FLOAT8(weights2),0,output+output_offset+2*outputChannelAlign);\n"
" vstore8(CONVERT_FLOAT8(weights3),0,output+output_offset+3*outputChannelAlign);\n"
"#elif QUANT_BIT == 8\n"
" const int ic=x << 1;\n"
" const int oc=y << 3;\n"
" const int output_offset=ic*outputChannelAlign+oc;\n"
" \n"
" #ifdef ASYMMETRIC\n"
" COMPUTE_FLOAT8 scale,offset;\n"
" {\n"
" COMPUTE_FLOAT16 ScaleOffset=CONVERT_COMPUTE_FLOAT16(convert_float16(vload16(0,dequantScaleOffset+((ic/blockDim)*outputChannel4Align+oc)*2))/coef);\n"
" scale=ScaleOffset.s02468ace;\n"
" offset=ScaleOffset.s13579bdf;\n"
" }\n"
" #else\n"
" COMPUTE_FLOAT8 scale=CONVERT_COMPUTE_FLOAT8(vload8(0,dequantScaleOffset+(ic/blockDim)*outputChannel4Align+oc))/coef);\n"
" #endif\n"
" COMPUTE_FLOAT8 weights0,weights1;\n"
" {\n"
" #ifdef USE_IMAGE\n"
" COMPUTE_FLOAT16 wei=CONVERT_COMPUTE_FLOAT16(as_char16(read_imagei(weight,SAMPLER,(int2)(x,y))));\n"
" #else\n"
" COMPUTE_FLOAT16 wei=CONVERT_COMPUTE_FLOAT16(vload16(x,weight+y*inputChannel4Align*8));\n"
" #endif\n"
" #ifdef ASYMMETRIC\n"
" weights0=wei.s01234567*scale+offset;\n"
" weights1=ic+1 >= inputChannel ? 0 : wei.s89abcdef*scale+offset;\n"
" #else\n"
" weights0=wei.s01234567*scale;\n"
" weights1=ic+1 >= inputChannel ? 0 : wei.s89abcdef*scale;\n"
" #endif\n"
" }\n"
" vstore8(CONVERT_FLOAT8(weights0),0,output+output_offset);\n"
" vstore8(CONVERT_FLOAT8(weights1),0,output+output_offset+outputChannelAlign);\n"
" #endif\n"
"}\n"
"__kernel void gemm_c4nhw4_to_nhwc(GLOBAL_SIZE_DIM2\n"
"__global const FLOAT* input,\n"
"__global FLOAT* output,\n"
"__private const int bhw,\n"
"__private const int channel,\n"
"__private const int channelAlign\n"
"){\n"
" const int x=get_global_id(0); //b/4\n"
" const int y=get_global_id(1); //c/4\n"
" UNIFORM_BOUNDRY_CHECK(x,y);\n"
" const int out_b_idx=x << 2;\n"
" const int out_c_idx=y << 2;\n"
" const int bhw4=bhw << 2;\n"
" const int input_offset=y*bhw4+out_b_idx*4;\n"
" FLOAT4 in0,in1,in2,in3;\n"
" if(out_c_idx+3<channel && out_b_idx+3<bhw){\n"
" in0=vload4(0,input+input_offset);\n"
" in1=vload4(0,input+input_offset+4);\n"
" in2=vload4(0,input+input_offset+8);\n"
" in3=vload4(0,input+input_offset+12);\n"
" } else{\n"
" if(out_c_idx+3<channel){\n"
" in0=vload4(0,input+input_offset);\n"
" in1=out_b_idx+1<bhw ? vload4(0,input+input_offset+4) : 0;\n"
" in2=out_b_idx+2<bhw ? vload4(0,input+input_offset+8) : 0;\n"
" in3=out_b_idx+3<bhw ? vload4(0,input+input_offset+12) : 0;\n"
" } else if(out_c_idx+1 == channel){\n"
" in0=(FLOAT4)(input[input_offset],0,0,0);\n"
" in1=out_b_idx+1<bhw ? (FLOAT4)(input[input_offset+4],0,0,0) : 0;\n"
" in2=out_b_idx+2<bhw ? (FLOAT4)(input[input_offset+8],0,0,0) : 0;\n"
" in3=out_b_idx+3<bhw ? (FLOAT4)(input[input_offset+12],0,0,0) : 0;\n"
" } else if(out_c_idx+2 == channel){\n"
" in0=(FLOAT4)(input[input_offset],input[input_offset+1],0,0);\n"
" in1=out_b_idx+1<bhw ? (FLOAT4)(input[input_offset+4],input[input_offset+5],0,0) : 0;\n"
" in2=out_b_idx+2<bhw ? (FLOAT4)(input[input_offset+8],input[input_offset+9],0,0) : 0;\n"
" in3=out_b_idx+3<bhw ? (FLOAT4)(input[input_offset+12],input[input_offset+13],0,0) : 0;\n"
" } else if(out_c_idx+3 == channel){\n"
" in0=(FLOAT4)(input[input_offset],input[input_offset+1],input[input_offset+2],0);\n"
" in1=out_b_idx+1<bhw ? (FLOAT4)(input[input_offset+4],input[input_offset+5],input[input_offset+6],0) : 0;\n"
" in2=out_b_idx+2<bhw ? (FLOAT4)(input[input_offset+8],input[input_offset+9],input[input_offset+10],0) : 0;\n"
" in3=out_b_idx+3<bhw ? (FLOAT4)(input[input_offset+12],input[input_offset+13],input[input_offset+14],0) : 0;\n"
" }\n"
" }\n"
" int out_offset=out_b_idx*channelAlign+out_c_idx;\n"
" vstore4(in0,0,output+out_offset);\n"
" vstore4(in1,0,output+out_offset+channelAlign);\n"
" vstore4(in2,0,output+out_offset+channelAlign+channelAlign);\n"
" vstore4(in3,0,output+out_offset+channelAlign+channelAlign+channelAlign);\n"
"}\n"
"__kernel void gemm_nhwc_to_c4nhw4(GLOBAL_SIZE_DIM2\n"
"__global const FLOAT* input,\n"
"__global FLOAT* output,\n"
"__private const int bhw,\n"
"__private const int channelAlign\n"
"){\n"
" const int x=get_global_id(0); //b/4\n"
" const int y=get_global_id(1); //c/4\n"
" UNIFORM_BOUNDRY_CHECK(x,y);\n"
" const int out_b_idx=x << 2;\n"
" const int out_c_idx=y << 2;\n"
" const int bhw4=bhw << 2;\n"
" const int input_offset=out_b_idx*channelAlign+out_c_idx;\n"
" FLOAT4 in0=vload4(0,input+input_offset);\n"
" FLOAT4 in1=vload4(0,input+input_offset+channelAlign);\n"
" FLOAT4 in2=vload4(0,input+input_offset+channelAlign+channelAlign);\n"
" FLOAT4 in3=vload4(0,input+input_offset+channelAlign+channelAlign+channelAlign);\n"
" int out_offset=y*bhw4+out_b_idx*4;\n"
" vstore4(in0,0,output+out_offset);\n"
" if(out_b_idx+1 >= bhw) return;\n"
" vstore4(in1,0,output+out_offset+4);\n"
" if(out_b_idx+2 >= bhw) return;\n"
" vstore4(in2,0,output+out_offset+8);\n"
" if(out_b_idx+3 >= bhw) return;\n"
" vstore4(in3,0,output+out_offset+12);\n"
"}\n"
"#define UCHAR4_TO_FLOAT8(b, scale, offset) "" wei.s0 = (COMPUTE_FLOAT)((b.s0 >> 4) - 8); "" wei.s1 = (COMPUTE_FLOAT)((b.s0 & 15) - 8); "" wei.s2 = (COMPUTE_FLOAT)((b.s1 >> 4) - 8); "" wei.s3 = (COMPUTE_FLOAT)((b.s1 & 15) - 8); "" wei.s4 = (COMPUTE_FLOAT)((b.s2 >> 4) - 8); "" wei.s5 = (COMPUTE_FLOAT)((b.s2 & 15) - 8); "" wei.s6 = (COMPUTE_FLOAT)((b.s3 >> 4) - 8); "" wei.s7 = (COMPUTE_FLOAT)((b.s3 & 15) - 8); "" wei=wei*scale+offset;\n"
"__kernel void gemm_b4_c8_int4_buf(GLOBAL_SIZE_DIM2\n"
" __global const FLOAT* input,\n"
"#ifdef USE_IMAGE\n"
" __read_only image2d_t weight,\n"
"#else\n"
" __global const uchar *weight,\n"
"#endif\n"
" __global const FLOAT *dequantScaleOffset,\n"
" __global const FLOAT *bias,\n"
" __global FLOAT* output,\n"
" __private const int bhw,\n"
" __private const int dstChannelAlign,\n"
" __private const int srcChannelAlign,\n"
" __private const int blockNum,\n"
" __private const int blockDim,\n"
" __private const float coef) {\n"
" const int x=get_global_id(0); //b/4\n"
" const int y=get_global_id(1); //c/8\n"
" UNIFORM_BOUNDRY_CHECK(x,y);\n"
" \n"
" const int out_b_idx=x << 2;\n"
" const int out_c_idx=y << 1;\n"
" COMPUTE_FLOAT8 out0=CONVERT_COMPUTE_FLOAT8(vload8(0,bias+(out_c_idx << 2)));\n"
" COMPUTE_FLOAT8 out1=out0;\n"
" COMPUTE_FLOAT8 out2=out0;\n"
" COMPUTE_FLOAT8 out3=out0;\n"
" \n"
" const int bhw4=bhw << 2;\n"
" const int input_offset=out_b_idx*4;\n"
" int out_offset=out_c_idx*bhw4+out_b_idx*4;\n"
"#ifndef USE_IMAGE\n"
" const int weight_offset=y*srcChannelAlign*4;\n"
"#endif\n"
" const int loop=(blockDim+4-1)/4;\n"
"#if INPUT_CHANNEL_LEAVES_NUM != 0\n"
" const int loop_end=max(loop-1,0);\n"
"#else\n"
" const int loop_end=loop;\n"
"#endif\n"
"#if INPUT_BATCH_LEAVES_NUM != 0\n"
" if(out_b_idx+3 >= bhw){\n"
" for (int i=0; i<blockNum; i++){\n"
" #ifdef ASYMMETRIC\n"
" COMPUTE_FLOAT8 scale,offset;\n"
" {\n"
" COMPUTE_FLOAT16 scaleOffset=CONVERT_COMPUTE_FLOAT16(convert_float16(vload16(0,dequantScaleOffset+(out_c_idx << 3)+i*dstChannelAlign*2))/coef);\n"
" scale=scaleOffset.s02468ace;\n"
" offset=scaleOffset.s13579bdf;\n"
" }\n"
" #else\n"
" COMPUTE_FLOAT8 scale=CONVERT_COMPUTE_FLOAT8(vload8(0,dequantScaleOffset+(out_c_idx << 2)+i*dstChannelAlign))/coef;\n"
" COMPUTE_FLOAT8 offset=0;\n"
" #endif\n"
" for (int j=0; j<loop_end; j++) {\n"
" int k=i*loop+j;\n"
" COMPUTE_FLOAT8 wei;\n"
" #ifdef USE_IMAGE\n"
" uchar16 charWeightsInt40=as_uchar16(read_imagei(weight,SAMPLER,(int2)(k,y)));\n"
" #else\n"
" uchar16 charWeightsInt40=vload16(k,weight+weight_offset);\n"
" #endif\n"
" COMPUTE_FLOAT4 in0=CONVERT_COMPUTE_FLOAT4(vload4(0,input+input_offset+k*bhw4));\n"
" #if INPUT_BATCH_LEAVES_NUM >= 2\n"
" COMPUTE_FLOAT4 in1=CONVERT_COMPUTE_FLOAT4(vload4(0,input+input_offset+k*bhw4+4));\n"
" #endif\n"
" #if INPUT_BATCH_LEAVES_NUM >= 3\n"
" COMPUTE_FLOAT4 in2=CONVERT_COMPUTE_FLOAT4(vload4(0,input+input_offset+k*bhw4+8));\n"
" #endif\n"
" {\n"
" UCHAR4_TO_FLOAT8(charWeightsInt40.s0123,scale,offset);\n"
" out0=mad((COMPUTE_FLOAT8)in0.s0,wei,out0);\n"
" #if INPUT_BATCH_LEAVES_NUM >= 2\n"
" out1=mad((COMPUTE_FLOAT8)in1.s0,wei,out1);\n"
" #endif\n"
" #if INPUT_BATCH_LEAVES_NUM >= 3\n"
" out2=mad((COMPUTE_FLOAT8)in2.s0,wei,out2);\n"
" #endif\n"
" }\n"
" {\n"
" UCHAR4_TO_FLOAT8(charWeightsInt40.s4567,scale,offset);\n"
" out0=mad((COMPUTE_FLOAT8)in0.s1,wei,out0);\n"
" #if INPUT_BATCH_LEAVES_NUM >= 2\n"
" out1=mad((COMPUTE_FLOAT8)in1.s1,wei,out1);\n"
" #endif\n"
" #if INPUT_BATCH_LEAVES_NUM >= 3\n"
" out2=mad((COMPUTE_FLOAT8)in2.s1,wei,out2);\n"
" #endif\n"
" }\n"
" {\n"
" UCHAR4_TO_FLOAT8(charWeightsInt40.s89ab,scale,offset);\n"
" out0=mad((COMPUTE_FLOAT8)in0.s2,wei,out0);\n"
" #if INPUT_BATCH_LEAVES_NUM >= 2\n"
" out1=mad((COMPUTE_FLOAT8)in1.s2,wei,out1);\n"
" #endif\n"
" #if INPUT_BATCH_LEAVES_NUM >= 3\n"
" out2=mad((COMPUTE_FLOAT8)in2.s2,wei,out2);\n"
" #endif\n"
" }\n"
" {\n"
" UCHAR4_TO_FLOAT8(charWeightsInt40.scdef,scale,offset);\n"
" out0=mad((COMPUTE_FLOAT8)in0.s3,wei,out0);\n"
" #if INPUT_BATCH_LEAVES_NUM >= 2\n"
" out1=mad((COMPUTE_FLOAT8)in1.s3,wei,out1);\n"
" #endif\n"
" #if INPUT_BATCH_LEAVES_NUM >= 3\n"
" out2=mad((COMPUTE_FLOAT8)in2.s3,wei,out3);\n"
" #endif\n"
" }\n"
" }\n"
" #if INPUT_CHANNEL_LEAVES_NUM != 0\n"
" {\n"
" int k=i*loop+loop_end;\n"
" COMPUTE_FLOAT8 wei;\n"
" COMPUTE_FLOAT4 in0=CONVERT_COMPUTE_FLOAT4(vload4(0,input+input_offset+k*bhw4));\n"
" #if INPUT_BATCH_LEAVES_NUM >= 2\n"
" COMPUTE_FLOAT4 in1=CONVERT_COMPUTE_FLOAT4(vload4(0,input+input_offset+k*bhw4+4));\n"
" #endif\n"
" #if INPUT_BATCH_LEAVES_NUM >= 3\n"
" COMPUTE_FLOAT4 in2=CONVERT_COMPUTE_FLOAT4(vload4(0,input+input_offset+k*bhw4+8));\n"
" #endif\n"
" #ifdef USE_IMAGE\n"
" uchar16 charWeightsInt40=as_uchar16(read_imagei(weight,SAMPLER,(int2)(k,y)));\n"
" #else\n"
" uchar16 charWeightsInt40=vload16(k,weight+weight_offset);\n"
" #endif\n"
" {\n"
" UCHAR4_TO_FLOAT8(charWeightsInt40.s0123,scale,offset);\n"
" out0=mad((COMPUTE_FLOAT8)in0.s0,wei,out0);\n"
" #if INPUT_BATCH_LEAVES_NUM >= 2\n"
" out1=mad((COMPUTE_FLOAT8)in1.s0,wei,out1);\n"
" #endif\n"
" #if INPUT_BATCH_LEAVES_NUM >= 3\n"
" out2=mad((COMPUTE_FLOAT8)in2.s0,wei,out2);\n"
" #endif\n"
" }\n"
" #if INPUT_CHANNEL_LEAVES_NUM >= 2\n"
" {\n"
" UCHAR4_TO_FLOAT8(charWeightsInt40.s4567,scale,offset);\n"
" out0=mad((COMPUTE_FLOAT8)in0.s1,wei,out0);\n"
" #if INPUT_BATCH_LEAVES_NUM >= 2\n"
" out1=mad((COMPUTE_FLOAT8)in1.s1,wei,out1);\n"
" #endif\n"
" #if INPUT_BATCH_LEAVES_NUM >= 3\n"
" out2=mad((COMPUTE_FLOAT8)in2.s1,wei,out2);\n"
" #endif\n"
" }\n"
" #endif\n"
" #if INPUT_CHANNEL_LEAVES_NUM >= 3\n"
" {\n"
" UCHAR4_TO_FLOAT8(charWeightsInt40.s89ab,scale,offset);\n"
" out0=mad((COMPUTE_FLOAT8)in0.s2,wei,out0);\n"
" #if INPUT_BATCH_LEAVES_NUM >= 2\n"
" out1=mad((COMPUTE_FLOAT8)in1.s2,wei,out1);\n"
" #endif\n"
" #if INPUT_BATCH_LEAVES_NUM >= 3\n"
" out2=mad((COMPUTE_FLOAT8)in2.s2,wei,out2);\n"
" #endif\n"
" }\n"
" #endif\n"
" }\n"
" #endif\n"
" }\n"
" } else {\n"
"#endif\n"
" for (int i=0; i<blockNum; i++){\n"
" #ifdef ASYMMETRIC\n"
" COMPUTE_FLOAT8 scale,offset;\n"
" {\n"
" COMPUTE_FLOAT16 scaleOffset=CONVERT_COMPUTE_FLOAT16(convert_float16(vload16(0,dequantScaleOffset+(out_c_idx << 3)+i*dstChannelAlign*2))/coef);\n"
" scale=scaleOffset.s02468ace;\n"
" offset=scaleOffset.s13579bdf;\n"
" }\n"
" #else\n"
" COMPUTE_FLOAT8 scale=CONVERT_COMPUTE_FLOAT8(vload8(0,dequantScaleOffset+(out_c_idx << 2)+i*dstChannelAlign))/coef;\n"
" COMPUTE_FLOAT8 offset=0;\n"
" #endif\n"
" for (int j=0; j<loop_end; j++) {\n"
" int k=i*loop+j;\n"
" COMPUTE_FLOAT8 wei;\n"
" COMPUTE_FLOAT16 in=CONVERT_COMPUTE_FLOAT16(vload16(0,input+input_offset+k*bhw4));\n"
" #ifdef USE_IMAGE\n"
" uchar16 charWeightsInt40=as_uchar16(read_imagei(weight,SAMPLER,(int2)(k,y)));\n"
" #else\n"
" uchar16 charWeightsInt40=vload16(k,weight+weight_offset);\n"
" #endif\n"
" {\n"
" UCHAR4_TO_FLOAT8(charWeightsInt40.s0123,scale,offset);\n"
" out0=mad((COMPUTE_FLOAT8)in.s0,wei,out0);\n"
" out1=mad((COMPUTE_FLOAT8)in.s4,wei,out1);\n"
" out2=mad((COMPUTE_FLOAT8)in.s8,wei,out2);\n"
" out3=mad((COMPUTE_FLOAT8)in.sc,wei,out3);\n"
" }\n"
" {\n"
" UCHAR4_TO_FLOAT8(charWeightsInt40.s4567,scale,offset);\n"
" out0=mad((COMPUTE_FLOAT8)in.s1,wei,out0);\n"
" out1=mad((COMPUTE_FLOAT8)in.s5,wei,out1);\n"
" out2=mad((COMPUTE_FLOAT8)in.s9,wei,out2);\n"
" out3=mad((COMPUTE_FLOAT8)in.sd,wei,out3);\n"
" }\n"
" {\n"
" UCHAR4_TO_FLOAT8(charWeightsInt40.s89ab,scale,offset);\n"
" out0=mad((COMPUTE_FLOAT8)in.s2,wei,out0);\n"
" out1=mad((COMPUTE_FLOAT8)in.s6,wei,out1);\n"
" out2=mad((COMPUTE_FLOAT8)in.sa,wei,out2);\n"
" out3=mad((COMPUTE_FLOAT8)in.se,wei,out3);\n"
" }\n"
" {\n"
" UCHAR4_TO_FLOAT8(charWeightsInt40.scdef,scale,offset);\n"
" out0=mad((COMPUTE_FLOAT8)in.s3,wei,out0);\n"
" out1=mad((COMPUTE_FLOAT8)in.s7,wei,out1);\n"
" out2=mad((COMPUTE_FLOAT8)in.sb,wei,out2);\n"
" out3=mad((COMPUTE_FLOAT8)in.sf,wei,out3);\n"
" }\n"
" }\n"
" #if INPUT_CHANNEL_LEAVES_NUM != 0\n"
" {\n"
" int k=i*loop+loop_end;\n"
" COMPUTE_FLOAT8 wei;\n"
" COMPUTE_FLOAT16 in=CONVERT_COMPUTE_FLOAT16(vload16(0,input+input_offset+k*bhw4));\n"
" #ifdef USE_IMAGE\n"
" uchar16 charWeightsInt40=as_uchar16(read_imagei(weight,SAMPLER,(int2)(k,y)));\n"
" #else\n"
" uchar16 charWeightsInt40=vload16(k,weight+weight_offset);\n"
" #endif\n"
" {\n"
" UCHAR4_TO_FLOAT8(charWeightsInt40.s0123,scale,offset);\n"
" out0=mad((COMPUTE_FLOAT8)in.s0,wei,out0);\n"
" out1=mad((COMPUTE_FLOAT8)in.s4,wei,out1);\n"
" out2=mad((COMPUTE_FLOAT8)in.s8,wei,out2);\n"
" out3=mad((COMPUTE_FLOAT8)in.sc,wei,out3);\n"
" }\n"
" #if INPUT_CHANNEL_LEAVES_NUM >= 2\n"
" {\n"
" UCHAR4_TO_FLOAT8(charWeightsInt40.s4567,scale,offset);\n"
" out0=mad((COMPUTE_FLOAT8)in.s1,wei,out0);\n"
" out1=mad((COMPUTE_FLOAT8)in.s5,wei,out1);\n"
" out2=mad((COMPUTE_FLOAT8)in.s9,wei,out2);\n"
" out3=mad((COMPUTE_FLOAT8)in.sd,wei,out3);\n"
" }\n"
" #endif\n"
" #if INPUT_CHANNEL_LEAVES_NUM >= 3\n"
" {\n"
" UCHAR4_TO_FLOAT8(charWeightsInt40.s89ab,scale,offset);\n"
" out0=mad((COMPUTE_FLOAT8)in.s2,wei,out0);\n"
" out1=mad((COMPUTE_FLOAT8)in.s6,wei,out1);\n"
" out2=mad((COMPUTE_FLOAT8)in.sa,wei,out2);\n"
" out3=mad((COMPUTE_FLOAT8)in.se,wei,out3);\n"
" }\n"
" #endif\n"
" }\n"
" #endif\n"
" }\n"
"#if INPUT_BATCH_LEAVES_NUM != 0\n"
" }\n"
"#endif\n"
" \n"
"#ifdef RELU\n"
" out0=fmax(out0,(COMPUTE_FLOAT8)0);\n"
" out1=fmax(out1,(COMPUTE_FLOAT8)0);\n"
" out2=fmax(out2,(COMPUTE_FLOAT8)0);\n"
" out3=fmax(out3,(COMPUTE_FLOAT8)0);\n"
"#endif\n"
"#ifdef RELU6\n"
" out0=clamp(out0,(COMPUTE_FLOAT8)0,(COMPUTE_FLOAT8)6);\n"
" out1=clamp(out1,(COMPUTE_FLOAT8)0,(COMPUTE_FLOAT8)6);\n"
" out2=clamp(out2,(COMPUTE_FLOAT8)0,(COMPUTE_FLOAT8)6);\n"
" out3=clamp(out3,(COMPUTE_FLOAT8)0,(COMPUTE_FLOAT8)6);\n"
"#endif\n"
"#if INPUT_BATCH_LEAVES_NUM != 0\n"
" if(out_b_idx+3 >= bhw){\n"
" #if INPUT_BATCH_LEAVES_NUM == 3\n"
" vstore8(CONVERT_FLOAT8((COMPUTE_FLOAT8)(out0.s0123,out1.s0123)),0,output+out_offset);\n"
" vstore4(CONVERT_FLOAT4(out2.s0123),0,output+out_offset+8);\n"
" if((out_c_idx << 2)+4<dstChannelAlign){\n"
" vstore8(CONVERT_FLOAT8((COMPUTE_FLOAT8)(out0.s4567,out1.s4567)),0,output+out_offset+bhw4);\n"
" vstore4(CONVERT_FLOAT4(out2.s4567),0,output+out_offset+bhw4+8);\n"
" }\n"
" #elif INPUT_BATCH_LEAVES_NUM == 2\n"
" vstore8(CONVERT_FLOAT8((COMPUTE_FLOAT8)(out0.s0123,out1.s0123)),0,output+out_offset);\n"
" if((out_c_idx << 2)+4<dstChannelAlign){\n"
" vstore8(CONVERT_FLOAT8((COMPUTE_FLOAT8)(out0.s4567,out1.s4567)),0,output+out_offset+bhw4);\n"
" }\n"
" #elif INPUT_BATCH_LEAVES_NUM == 1\n"
" vstore4(CONVERT_FLOAT4(out0.s0123),0,output+out_offset);\n"
" if((out_c_idx << 2)+4<dstChannelAlign){\n"
" vstore4(CONVERT_FLOAT4(out0.s4567),0,output+out_offset+bhw4);\n"
" }\n"
" #endif\n"
" }else{\n"
"#endif\n"
" vstore16(CONVERT_FLOAT16((COMPUTE_FLOAT16)(out0.s0123,out1.s0123,out2.s0123,out3.s0123)),0,output+out_offset);\n"
" if((out_c_idx << 2)+4<dstChannelAlign){\n"
" vstore16(CONVERT_FLOAT16((COMPUTE_FLOAT16)(out0.s4567,out1.s4567,out2.s4567,out3.s4567)),0,output+out_offset+bhw4);\n"
" }\n"
"#if INPUT_BATCH_LEAVES_NUM != 0\n"
" }\n"
"#endif\n"
"}\n"
"__kernel void gemm_b4_c8_int8_buf(GLOBAL_SIZE_DIM2\n"
" __global const FLOAT* input,\n"
"#ifdef USE_IMAGE\n"
" __read_only image2d_t weight,\n"
"#else\n"
" __global const char *weight,\n"
"#endif\n"
" __global const FLOAT *dequantScaleOffset,\n"
" __global const FLOAT *bias,\n"
" __global FLOAT* output,\n"
" __private const int bhw,\n"
" __private const int dstChannelAlign,\n"
" __private const int srcChannelAlign,\n"
" __private const int blockNum,\n"
" __private const int blockDim,\n"
" __private const float coef) {\n"
" const int x=get_global_id(0); //b/4\n"
" const int y=get_global_id(1); //c/8\n"
" UNIFORM_BOUNDRY_CHECK(x,y);\n"
" \n"
" const int out_b_idx=x << 2;\n"
" const int out_c_idx=y << 1;\n"
" COMPUTE_FLOAT8 out0=CONVERT_COMPUTE_FLOAT8(vload8(0,bias+(out_c_idx << 2)));\n"
" COMPUTE_FLOAT8 out1=out0;\n"
" COMPUTE_FLOAT8 out2=out0;\n"
" COMPUTE_FLOAT8 out3=out0;\n"
" \n"
" const int bhw4=bhw << 2;\n"
" const int input_offset=out_b_idx*4;\n"
" int out_offset=out_c_idx*bhw4+out_b_idx*4;\n"
"#ifndef USE_IMAGE\n"
" const int weight_offset=y*srcChannelAlign*8;\n"
"#endif\n"
" const int loop=(blockDim+4-1)/4;\n"
"#if INPUT_CHANNEL_LEAVES_NUM != 0\n"
" const int loop_end=max(loop-1,0);\n"
"#else\n"
" const int loop_end=loop;\n"
"#endif\n"
"#if INPUT_BATCH_LEAVES_NUM != 0\n"
" if(out_b_idx+3 >= bhw){\n"
" for (int i=0; i<blockNum; i++){\n"
" COMPUTE_FLOAT16 scale,offset;\n"
" {\n"
" #ifdef ASYMMETRIC\n"
" COMPUTE_FLOAT16 scaleOffset=CONVERT_COMPUTE_FLOAT16(convert_float16(vload16(0,dequantScaleOffset+(out_c_idx << 3)+i*dstChannelAlign*2))/coef);\n"
" scale=(COMPUTE_FLOAT16)(scaleOffset.s02468ace,scaleOffset.s02468ace);\n"
" offset=(COMPUTE_FLOAT16)(scaleOffset.s13579bdf,scaleOffset.s13579bdf);\n"
" #else\n"
" scale.s01234567=CONVERT_COMPUTE_FLOAT8(vload8(0,dequantScaleOffset+(out_c_idx << 2)+i*dstChannelAlign))/coef;\n"
" scale.s89abcdef=scale.s01234567;\n"
" offset=0;\n"
" #endif\n"
" }\n"
" for (int j=0; j<loop_end; j++) {\n"
" int k=i*loop+j;\n"
" int k2=k << 1;\n"
" #ifdef USE_IMAGE\n"
" COMPUTE_FLOAT16 wei0=CONVERT_COMPUTE_FLOAT16(as_char16(read_imagei(weight,SAMPLER,(int2)(k2,y))));\n"
" COMPUTE_FLOAT16 wei1=CONVERT_COMPUTE_FLOAT16(as_char16(read_imagei(weight,SAMPLER,(int2)(k2+1,y))));\n"
" #else\n"
" COMPUTE_FLOAT16 wei0=CONVERT_COMPUTE_FLOAT16(vload16(k2,weight+weight_offset));\n"
" COMPUTE_FLOAT16 wei1=CONVERT_COMPUTE_FLOAT16(vload16(k2+1,weight+weight_offset));\n"
" #endif\n"
" #ifdef ASYMMETRIC\n"
" wei0=wei0*scale+offset;\n"
" wei1=wei1*scale+offset;\n"
" #else\n"
" wei0=wei0*scale;\n"
" wei1=wei1*scale;\n"
" #endif\n"
" COMPUTE_FLOAT4 in0=CONVERT_COMPUTE_FLOAT4(vload4(0,input+input_offset+k*bhw4));\n"
" #if INPUT_BATCH_LEAVES_NUM >= 2\n"
" COMPUTE_FLOAT4 in1=CONVERT_COMPUTE_FLOAT4(vload4(0,input+input_offset+k*bhw4+4));\n"
" #endif\n"
" #if INPUT_BATCH_LEAVES_NUM >= 3\n"
" COMPUTE_FLOAT4 in2=CONVERT_COMPUTE_FLOAT4(vload4(0,input+input_offset+k*bhw4+8));\n"
" #endif\n"
" out0=mad((COMPUTE_FLOAT8)in0.s0,wei0.s01234567,out0);\n"
" out0=mad((COMPUTE_FLOAT8)in0.s1,wei0.s89abcdef,out0);\n"
" out0=mad((COMPUTE_FLOAT8)in0.s2,wei1.s01234567,out0);\n"
" out0=mad((COMPUTE_FLOAT8)in0.s3,wei1.s89abcdef,out0);\n"
" #if INPUT_BATCH_LEAVES_NUM >= 2\n"
" out1=mad((COMPUTE_FLOAT8)in1.s0,wei0.s01234567,out1);\n"
" out1=mad((COMPUTE_FLOAT8)in1.s1,wei0.s89abcdef,out1);\n"
" out1=mad((COMPUTE_FLOAT8)in1.s2,wei1.s01234567,out1);\n"
" out1=mad((COMPUTE_FLOAT8)in1.s3,wei1.s89abcdef,out1);\n"
" #endif\n"
" #if INPUT_BATCH_LEAVES_NUM >= 3\n"
" out2=mad((COMPUTE_FLOAT8)in2.s0,wei0.s01234567,out2);\n"
" out2=mad((COMPUTE_FLOAT8)in2.s1,wei0.s89abcdef,out2);\n"
" out2=mad((COMPUTE_FLOAT8)in2.s2,wei1.s01234567,out2);\n"
" out2=mad((COMPUTE_FLOAT8)in2.s3,wei1.s89abcdef,out2);\n"
" #endif\n"
" }\n"
" #if INPUT_CHANNEL_LEAVES_NUM != 0\n"
" {\n"
" int k=i*loop+loop_end;\n"
" int k2=k << 1;\n"
" #ifdef USE_IMAGE\n"
" COMPUTE_FLOAT16 wei0=CONVERT_COMPUTE_FLOAT16(as_char16(read_imagei(weight,SAMPLER,(int2)(k2,y))));\n"
" #else\n"
" COMPUTE_FLOAT16 wei0=CONVERT_COMPUTE_FLOAT16(vload16(k2,weight+weight_offset));\n"
" #endif\n"
" #ifdef ASYMMETRIC\n"
" wei0=wei0*scale+offset;\n"
" #else\n"
" wei0=wei0*scale;\n"
" #endif\n"
" COMPUTE_FLOAT4 in0=CONVERT_COMPUTE_FLOAT4(vload4(0,input+input_offset+k*bhw4));\n"
" #if INPUT_BATCH_LEAVES_NUM >= 2\n"
" COMPUTE_FLOAT4 in1=CONVERT_COMPUTE_FLOAT4(vload4(0,input+input_offset+k*bhw4+4));\n"
" #endif\n"
" #if INPUT_BATCH_LEAVES_NUM >= 3\n"
" COMPUTE_FLOAT4 in2=CONVERT_COMPUTE_FLOAT4(vload4(0,input+input_offset+k*bhw4+8));\n"
" #endif\n"
" out0=mad((COMPUTE_FLOAT8)in0.s0,wei0.s01234567,out0);\n"
" #if INPUT_BATCH_LEAVES_NUM >= 2\n"
" out1=mad((COMPUTE_FLOAT8)in1.s0,wei0.s01234567,out1);\n"
" #endif\n"
" #if INPUT_BATCH_LEAVES_NUM >= 3\n"
" out2=mad((COMPUTE_FLOAT8)in2.s0,wei0.s01234567,out2);\n"
" #endif\n"
" #if INPUT_CHANNEL_LEAVES_NUM >= 2\n"
" out0=mad((COMPUTE_FLOAT8)in0.s1,wei0.s89abcdef,out0);\n"
" #if INPUT_BATCH_LEAVES_NUM >= 2\n"
" out1=mad((COMPUTE_FLOAT8)in1.s1,wei0.s89abcdef,out1);\n"
" #endif\n"
" #if INPUT_BATCH_LEAVES_NUM >= 3\n"
" out2=mad((COMPUTE_FLOAT8)in2.s1,wei0.s89abcdef,out2);\n"
" #endif\n"
" #endif\n"
" #if INPUT_CHANNEL_LEAVES_NUM >= 3\n"
" #ifdef USE_IMAGE\n"
" COMPUTE_FLOAT16 wei1=CONVERT_COMPUTE_FLOAT16(as_char16(read_imagei(weight,SAMPLER,(int2)(k2+1,y))));\n"
" #else\n"
" COMPUTE_FLOAT16 wei1=CONVERT_COMPUTE_FLOAT16(vload16(k2+1,weight+weight_offset));\n"
" #endif\n"
" #ifdef ASYMMETRIC\n"
" wei1=wei1*scale+offset;\n"
" #else\n"
" wei1=wei1*scale;\n"
" #endif\n"
" out0=mad((COMPUTE_FLOAT8)in0.s2,wei1.s01234567,out0);\n"
" #if INPUT_BATCH_LEAVES_NUM >= 2\n"
" out1=mad((COMPUTE_FLOAT8)in1.s2,wei1.s01234567,out1);\n"
" #endif\n"
" #if INPUT_BATCH_LEAVES_NUM >= 3\n"
" out2=mad((COMPUTE_FLOAT8)in2.s2,wei1.s01234567,out2);\n"
" #endif\n"
" #endif\n"
" }\n"
" #endif\n"
" }\n"
" } else {\n"
"#endif\n"
" for (int i=0; i<blockNum; i++){\n"
" COMPUTE_FLOAT16 scale,offset;\n"
" {\n"
" #ifdef ASYMMETRIC\n"
" COMPUTE_FLOAT16 scaleOffset=CONVERT_COMPUTE_FLOAT16(convert_float16(vload16(0,dequantScaleOffset+(out_c_idx << 3)+i*dstChannelAlign*2))/coef);\n"
" scale=(COMPUTE_FLOAT16)(scaleOffset.s02468ace,scaleOffset.s02468ace);\n"
" offset=(COMPUTE_FLOAT16)(scaleOffset.s13579bdf,scaleOffset.s13579bdf);\n"
" #else\n"
" scale.s01234567=CONVERT_COMPUTE_FLOAT8(vload8(0,dequantScaleOffset+(out_c_idx << 2)+i*dstChannelAlign))/coef;\n"
" scale.s89abcdef=scale.s01234567;\n"
" offset=0;\n"
" #endif\n"
" }\n"
" for (int j=0; j<loop_end; j++) {\n"
" int k=i*loop+j;\n"
" int k2=k << 1;\n"
" #ifdef USE_IMAGE\n"
" COMPUTE_FLOAT16 wei0=CONVERT_COMPUTE_FLOAT16(as_char16(read_imagei(weight,SAMPLER,(int2)(k2,y))));\n"
" COMPUTE_FLOAT16 wei1=CONVERT_COMPUTE_FLOAT16(as_char16(read_imagei(weight,SAMPLER,(int2)(k2+1,y))));\n"
" #else\n"
" COMPUTE_FLOAT16 wei0=CONVERT_COMPUTE_FLOAT16(vload16(k2,weight+weight_offset));\n"
" COMPUTE_FLOAT16 wei1=CONVERT_COMPUTE_FLOAT16(vload16(k2+1,weight+weight_offset));\n"
" #endif\n"
" #ifdef ASYMMETRIC\n"
" wei0=wei0*scale+offset;\n"
" wei1=wei1*scale+offset;\n"
" #else\n"
" wei0=wei0*scale;\n"
" wei1=wei1*scale;\n"
" #endif\n"
" COMPUTE_FLOAT16 in=CONVERT_COMPUTE_FLOAT16(vload16(0,input+input_offset+k*bhw4));\n"
" out0=mad((COMPUTE_FLOAT8)in.s0,wei0.s01234567,out0);\n"
" out1=mad((COMPUTE_FLOAT8)in.s4,wei0.s01234567,out1);\n"
" out2=mad((COMPUTE_FLOAT8)in.s8,wei0.s01234567,out2);\n"
" out3=mad((COMPUTE_FLOAT8)in.sc,wei0.s01234567,out3);\n"
" out0=mad((COMPUTE_FLOAT8)in.s1,wei0.s89abcdef,out0);\n"
" out1=mad((COMPUTE_FLOAT8)in.s5,wei0.s89abcdef,out1);\n"
" out2=mad((COMPUTE_FLOAT8)in.s9,wei0.s89abcdef,out2);\n"
" out3=mad((COMPUTE_FLOAT8)in.sd,wei0.s89abcdef,out3);\n"
" out0=mad((COMPUTE_FLOAT8)in.s2,wei1.s01234567,out0);\n"
" out1=mad((COMPUTE_FLOAT8)in.s6,wei1.s01234567,out1);\n"
" out2=mad((COMPUTE_FLOAT8)in.sa,wei1.s01234567,out2);\n"
" out3=mad((COMPUTE_FLOAT8)in.se,wei1.s01234567,out3);\n"
" out0=mad((COMPUTE_FLOAT8)in.s3,wei1.s89abcdef,out0);\n"
" out1=mad((COMPUTE_FLOAT8)in.s7,wei1.s89abcdef,out1);\n"
" out2=mad((COMPUTE_FLOAT8)in.sb,wei1.s89abcdef,out2);\n"
" out3=mad((COMPUTE_FLOAT8)in.sf,wei1.s89abcdef,out3);\n"
" }\n"
" #if INPUT_CHANNEL_LEAVES_NUM != 0\n"
" {\n"
" int k=i*loop+loop_end;\n"
" int k2=k << 1;\n"
" #ifdef USE_IMAGE\n"
" COMPUTE_FLOAT16 wei0=CONVERT_COMPUTE_FLOAT16(as_char16(read_imagei(weight,SAMPLER,(int2)(k2,y))));\n"
" #else\n"
" COMPUTE_FLOAT16 wei0=CONVERT_COMPUTE_FLOAT16(vload16(k2,weight+weight_offset));\n"
" #endif\n"
" #ifdef ASYMMETRIC\n"
" wei0=wei0*scale+offset;\n"
" #else\n"
" wei0=wei0*scale;\n"
" #endif\n"
" COMPUTE_FLOAT16 in=CONVERT_COMPUTE_FLOAT16(vload16(0,input+input_offset+k*bhw4));\n"
" out0=mad((COMPUTE_FLOAT8)in.s0,wei0.s01234567,out0);\n"
" out1=mad((COMPUTE_FLOAT8)in.s4,wei0.s01234567,out1);\n"
" out2=mad((COMPUTE_FLOAT8)in.s8,wei0.s01234567,out2);\n"
" out3=mad((COMPUTE_FLOAT8)in.sc,wei0.s01234567,out3);\n"
" #if INPUT_CHANNEL_LEAVES_NUM >= 2\n"
" out0=mad((COMPUTE_FLOAT8)in.s1,wei0.s89abcdef,out0);\n"
" out1=mad((COMPUTE_FLOAT8)in.s5,wei0.s89abcdef,out1);\n"
" out2=mad((COMPUTE_FLOAT8)in.s9,wei0.s89abcdef,out2);\n"
" out3=mad((COMPUTE_FLOAT8)in.sd,wei0.s89abcdef,out3);\n"
" #endif\n"
" #if INPUT_CHANNEL_LEAVES_NUM >= 3\n"
" #ifdef USE_IMAGE\n"
" COMPUTE_FLOAT16 wei1=CONVERT_COMPUTE_FLOAT16(as_char16(read_imagei(weight,SAMPLER,(int2)(k2+1,y))));\n"
" #else\n"
" COMPUTE_FLOAT16 wei1=CONVERT_COMPUTE_FLOAT16(vload16(k2+1,weight+weight_offset));\n"
" #endif\n"
" #ifdef ASYMMETRIC\n"
" wei1=wei1*scale+offset;\n"
" #else\n"
" wei1=wei1*scale;\n"
" #endif\n"
" out0=mad((COMPUTE_FLOAT8)in.s2,wei1.s01234567,out0);\n"
" out1=mad((COMPUTE_FLOAT8)in.s6,wei1.s01234567,out1);\n"
" out2=mad((COMPUTE_FLOAT8)in.sa,wei1.s01234567,out2);\n"
" out3=mad((COMPUTE_FLOAT8)in.se,wei1.s01234567,out3);\n"
" #endif\n"
" }\n"
" #endif\n"
" }\n"
"#if INPUT_BATCH_LEAVES_NUM != 0\n"
" }\n"
"#endif\n"
" \n"
"#ifdef RELU\n"
" out0=fmax(out0,(COMPUTE_FLOAT8)0);\n"
" out1=fmax(out1,(COMPUTE_FLOAT8)0);\n"
" out2=fmax(out2,(COMPUTE_FLOAT8)0);\n"
" out3=fmax(out3,(COMPUTE_FLOAT8)0);\n"
"#endif\n"
"#ifdef RELU6\n"
" out0=clamp(out0,(COMPUTE_FLOAT8)0,(COMPUTE_FLOAT8)6);\n"
" out1=clamp(out1,(COMPUTE_FLOAT8)0,(COMPUTE_FLOAT8)6);\n"
" out2=clamp(out2,(COMPUTE_FLOAT8)0,(COMPUTE_FLOAT8)6);\n"
" out3=clamp(out3,(COMPUTE_FLOAT8)0,(COMPUTE_FLOAT8)6);\n"
"#endif\n"
"#if INPUT_BATCH_LEAVES_NUM != 0\n"
" if(out_b_idx+3 >= bhw){\n"
" #if INPUT_BATCH_LEAVES_NUM == 3\n"
" vstore8(CONVERT_FLOAT8((COMPUTE_FLOAT8)(out0.s0123,out1.s0123)),0,output+out_offset);\n"
" vstore4(CONVERT_FLOAT4(out2.s0123),0,output+out_offset+8);\n"
" if((out_c_idx << 2)+4<dstChannelAlign){\n"
" vstore8(CONVERT_FLOAT8((COMPUTE_FLOAT8)(out0.s4567,out1.s4567)),0,output+out_offset+bhw4);\n"
" vstore4(CONVERT_FLOAT4(out2.s4567),0,output+out_offset+bhw4+8);\n"
" }\n"
" #elif INPUT_BATCH_LEAVES_NUM == 2\n"
" vstore8(CONVERT_FLOAT8((COMPUTE_FLOAT8)(out0.s0123,out1.s0123)),0,output+out_offset);\n"
" if((out_c_idx << 2)+4<dstChannelAlign){\n"
" vstore8(CONVERT_FLOAT8((COMPUTE_FLOAT8)(out0.s4567,out1.s4567)),0,output+out_offset+bhw4);\n"
" }\n"
" #elif INPUT_BATCH_LEAVES_NUM == 1\n"
" vstore4(CONVERT_FLOAT4(out0.s0123),0,output+out_offset);\n"
" if((out_c_idx << 2)+4<dstChannelAlign){\n"
" vstore4(CONVERT_FLOAT4(out0.s4567),0,output+out_offset+bhw4);\n"
" }\n"
" #endif\n"
" }else{\n"
"#endif\n"
" vstore16(CONVERT_FLOAT16((COMPUTE_FLOAT16)(out0.s0123,out1.s0123,out2.s0123,out3.s0123)),0,output+out_offset);\n"
" if((out_c_idx << 2)+4<dstChannelAlign){\n"
" vstore16(CONVERT_FLOAT16((COMPUTE_FLOAT16)(out0.s4567,out1.s4567,out2.s4567,out3.s4567)),0,output+out_offset+bhw4);\n"
" }\n"
"#if INPUT_BATCH_LEAVES_NUM != 0\n"
" }\n"
"#endif\n"
"}\n"
;
} // namespace MNN

// W4A16 CHANGE: Define cl_half for host-side storage
using cl_half = uint16_t;

// =======================================================================
// == NEW: FP32 <-> FP16 Conversion Utilities ==
// =======================================================================
// Converts a 32-bit float to a 16-bit half-float
cl_half float_to_half_bits(float f) {
    union {
        float f;
        uint32_t u;
    } converter = {f};
    uint32_t u = converter.u;

    uint32_t sign = (u >> 31) & 0x1;
    int32_t exp = ((u >> 23) & 0xff) - 127;
    uint32_t mant = u & 0x7fffff;

    if (exp > 15) { // Overflow -> infinity
        return (sign << 15) | 0x7c00;
    }
    if (exp < -14) { // Underflow -> denormal or zero
        mant |= 0x800000; // Add hidden bit
        int shift = -14 - exp;
        if (shift > 24) {
            return sign << 15; // Too small, flush to zero
        }
        mant >>= shift;
        return (sign << 15) | (mant >> 13);
    }
    return (sign << 15) | ((exp + 15) << 10) | (mant >> 13);
}

// Converts a 16-bit half-float to a 32-bit float
float half_bits_to_float(cl_half h) {
    uint32_t sign = (h >> 15) & 0x1;
    uint32_t exp = (h >> 10) & 0x1f;
    uint32_t mant = h & 0x3ff;

    union {
        uint32_t u;
        float f;
    } converter;

    if (exp == 0) { // Denormal or zero
        if (mant == 0) { // Zero
            converter.u = sign << 31;
        } else { // Denormal
            exp = 1 - 15;
            while ((mant & 0x400) == 0) {
                mant <<= 1;
                exp--;
            }
            mant &= 0x3ff;
            converter.u = (sign << 31) | ((exp + 127) << 23) | (mant << 13);
        }
    } else if (exp == 31) { // Infinity or NaN
        converter.u = (sign << 31) | (0xff << 23) | (mant << 13);
    } else { // Normal
        converter.u = (sign << 31) | ((exp - 15 + 127) << 23) | (mant << 13);
    }
    return converter.f;
}


// Helper function to get platform and device
void select_opencl_device(cl::Platform& platform, cl::Device& device) {
    std::vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);

    if (platforms.empty()) {
        throw std::runtime_error("No OpenCL platforms found.");
    }

    platform = platforms[0];
    std::vector<cl::Device> devices;
    platform.getDevices(CL_DEVICE_TYPE_GPU, &devices);

    if (devices.empty()) {
        platform.getDevices(CL_DEVICE_TYPE_CPU, &devices);
        if(devices.empty()){
            throw std::runtime_error("No OpenCL GPU or CPU devices found.");
        }
    }

    device = devices[0];
}

// =======================================================================
// == INT4 Weight Packing Helper for Image2D layout (Unchanged) ==
// =======================================================================
void pack_weights_to_image_buffer_int4(
    std::vector<uint8_t>& dst_image_buffer,
    const std::vector<uint8_t>& src_weights_packed,
    int oc, int ic,
    int oc_padded, int ic_padded)
{
    const int image_width = (ic_padded + 3) / 4;
    const int image_height = (oc_padded + 7) / 8;
    const int bytes_per_pixel = 16;
    dst_image_buffer.assign(image_width * image_height * bytes_per_pixel, 0);

    for (int o = 0; o < oc; ++o) {
        for (int i = 0; i < ic; ++i) {
            int src_byte_idx = o * ((ic + 1) / 2) + (i / 2);
            uint8_t src_byte = src_weights_packed[src_byte_idx];
            uint8_t weight_4bit = (i % 2 == 0) ? (src_byte & 0x0F) : (src_byte >> 4);

            int x = i / 4;
            int y = o / 8;
            int ic_inner = i % 4;
            int oc_inner = o % 8;
            int chunk_offset_in_pixel = ic_inner * 4;
            int byte_offset_in_chunk = oc_inner / 2;
            int dst_pixel_base_idx = (y * image_width + x) * bytes_per_pixel;
            int final_dst_byte_idx = dst_pixel_base_idx + chunk_offset_in_pixel + byte_offset_in_chunk;

            if (oc_inner % 2 == 0) {
                dst_image_buffer[final_dst_byte_idx] |= (weight_4bit << 4);
            } else {
                dst_image_buffer[final_dst_byte_idx] |= weight_4bit;
            }
        }
    }
}


// CPU verification function (operates on simple, logical layouts)
// W4A16 CHANGE: This function now takes FP16 bias/dequant and converts them back to FP32 for calculation.
// It still uses FP32 for input and accumulation for simplicity and to get a "golden" reference.
void verify_on_cpu(
    std::vector<float>& cpu_output,
    const std::vector<float>& input_nhwc_f32,
    const std::vector<uint8_t>& weight_oic_packed,
    const std::vector<cl_half>& dequant_f16,
    const std::vector<cl_half>& bias_f16,
    int bhw, int dst_c, int src_c)
{
    cpu_output.assign(bhw * dst_c, 0.0f);

    for (int i = 0; i < bhw; ++i) {
        for (int j = 0; j < dst_c; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < src_c; ++k) {
                // NOTE: Using the original FP32 input for CPU verification
                float in_val = input_nhwc_f32[i * src_c + k];

                int src_byte_idx = j * ((src_c + 1) / 2) + (k / 2);
                uint8_t src_byte = weight_oic_packed[src_byte_idx];
                uint8_t weight_4bit_unsigned = (k % 2 == 0) ? (src_byte & 0x0F) : (src_byte >> 4);

                char w_int4 = static_cast<char>(weight_4bit_unsigned) - 8;

                // W4A16 CHANGE: Convert scale from FP16 to FP32 for calculation
                // NOTE: For CPU verification, we only use the first block of dequant scales,
                // as the GPU version (with the fix) replicates them.
                float scale = half_bits_to_float(dequant_f16[j]);
                float w_dequantized = static_cast<float>(w_int4) * scale;

                sum += in_val * w_dequantized;
            }
            // W4A16 CHANGE: Convert bias from FP16 to FP32 for calculation
            cpu_output[i * dst_c + j] = sum + half_bits_to_float(bias_f16[j]);
        }
    }
}

extern "C" int initOpenCL();

int main(int argc, char** argv) {
    if (initOpenCL() == 0) {
        std::cerr << "Failed to initialize OpenCL loader." << std::endl;
        return -1;
    }
    if (argc != 5) {
        std::cerr << "Usage: " << argv[0] << " <BHW> <DST_C> <SRC_C> <BLKNUM>" << std::endl;
        return 1;
    }

    try {
        const int bhw = std::stoi(argv[1]);
        const int dst_c = std::stoi(argv[2]);
        const int src_c = std::stoi(argv[3]);
        const int block_num = std::stoi(argv[4]);

        const std::string pack_kernel_name = "gemm_nhwc_to_c4nhw4";
        const std::string compute_kernel_name = "gemm_b4_c8_int4_buf";
        const std::string unpack_kernel_name = "gemm_c4nhw4_to_nhwc";

        // W4A16 CHANGE: Updated title
        std::cout << "Benchmarking & Verifying OpenCL Pipeline (W4A16 - INT4 weights, FP16 activations on Image2D)" << std::endl;
        std::cout << "Input Dimensions: BHW=" << bhw << ", DST_C=" << dst_c << ", SRC_C=" << src_c << ", block_num=" << block_num << std::endl;

        cl::Platform platform;
        cl::Device device;
        select_opencl_device(platform, device);

        if (device.getInfo<CL_DEVICE_IMAGE_SUPPORT>() == CL_FALSE) {
            throw std::runtime_error("Selected OpenCL device does not support images.");
        }
        // W4A16 CHANGE: Check for FP16 support on the device
        std::string extensions = device.getInfo<CL_DEVICE_EXTENSIONS>();
        if (extensions.find("cl_khr_fp16") == std::string::npos) {
            throw std::runtime_error("Selected OpenCL device does not support cl_khr_fp16 extension.");
        }

        cl::Context context(device);
        cl::CommandQueue queue(context, device, cl::QueueProperties::Profiling);

        std::cout << "Using Platform: " << platform.getInfo<CL_PLATFORM_NAME>() << std::endl;
        std::cout << "Using Device: " << device.getInfo<CL_DEVICE_NAME>() << std::endl;

        // W4A16 CHANGE: CRITICAL - Update build options to use 'half' precision types
        std::stringstream build_opts;
        build_opts << "-D MNN_SUPPORT_FP16 ";
        // Define compute types as half
        build_opts << "-DCOMPUTE_FLOAT=half -DCOMPUTE_FLOAT4=half4 -DCOMPUTE_FLOAT8=half8 -DCOMPUTE_FLOAT16=half16 ";
        build_opts << "-DCONVERT_COMPUTE_FLOAT=convert_half -DCONVERT_COMPUTE_FLOAT4=convert_half4 -DCONVERT_COMPUTE_FLOAT8=convert_half8 -DCONVERT_COMPUTE_FLOAT16=convert_half16 ";
        // Define storage/IO types as half
        build_opts << "-DFLOAT=half -DFLOAT4=half4 -DFLOAT8=half8 -DFLOAT16=half16 ";
        build_opts << "-DCONVERT_FLOAT4=convert_half4 -DCONVERT_FLOAT8=convert_half8 -DCONVERT_FLOAT16=convert_half16 ";

        build_opts << "-D INPUT_CHANNEL_LEAVES_NUM=0 ";
        build_opts << "-D INPUT_BATCH_LEAVES_NUM=0 ";
        build_opts << "-D USE_IMAGE ";
        build_opts << "-D QUANT_BIT=4 ";

        cl::Program program(context, MNN::gemm_conv1x1_buf);
        program.build({device}, build_opts.str().c_str());

        cl::Kernel kernel_pack(program, pack_kernel_name.c_str());
        cl::Kernel kernel_compute(program, compute_kernel_name.c_str());
        cl::Kernel kernel_unpack(program, unpack_kernel_name.c_str());

        const int BATCH_PACK = 4;
        const int DST_C_PACK = 8;
        const int SRC_C_PACK = 4;

        const int bhw_padded = (bhw + BATCH_PACK - 1) / BATCH_PACK * BATCH_PACK;
        const int dst_c_padded = (dst_c + DST_C_PACK - 1) / DST_C_PACK * DST_C_PACK;
        const int src_c_padded = (src_c + SRC_C_PACK - 1) / SRC_C_PACK * SRC_C_PACK;

        // ======================= 核心修正点 1: 增加对齐检查 =======================
        // 为了让分块逻辑正确，src_c_padded 必须能被 block_num 整除。
        if (src_c_padded % block_num != 0) {
            std::stringstream ss;
            ss << "Error: src_c_padded (" << src_c_padded << ") must be divisible by block_num (" << block_num << ").";
            throw std::runtime_error(ss.str());
        }
        // ========================================================================

        // W4A16 CHANGE: Host buffers are now cl_half (uint16_t).
        // We also keep an FP32 copy of the input for CPU verification.
        std::vector<float> host_input_nhwc_f32(bhw_padded * src_c_padded, 0.f);
        std::vector<cl_half> host_input_nhwc_f16(bhw_padded * src_c_padded, 0);
        std::vector<uint8_t> host_weight_oic_packed((dst_c * src_c + 1) / 2);
        std::vector<cl_half> host_bias_f16(dst_c_padded, 0);
        
        // ======================= FIX 1: Correct buffer size =======================
        // The dequant buffer size must match what the kernel expects: block_num * dst_c_padded
        std::vector<cl_half> host_dequant_f16(block_num * dst_c_padded, 0);
        // ==========================================================================

        std::cout << "Initializing and converting host data to FP16..." << std::endl;
        // A16: 初始化输入数据并转换为 FP16
        for(int b=0; b<bhw; ++b) {
            for(int c=0; c<src_c; ++c) {
                size_t idx = b*src_c_padded + c;
                float val_f32 = (static_cast<float>((b*src_c+c) % 5) - 2.0f) * 0.5f;
                host_input_nhwc_f32[idx] = val_f32;
                host_input_nhwc_f16[idx] = float_to_half_bits(val_f32);
            }
        }

        // W8: 初始化 INT4 权重数据
        for(size_t i = 0; i < host_weight_oic_packed.size(); ++i) {
            uint8_t val1 = (i * 2) % 16;
            uint8_t val2 = (i * 2 + 1) % 16;
            host_weight_oic_packed[i] = (val2 << 4) | val1;
        }

        // A16: 初始化偏置和反量化尺度数据并转换为 FP16
        for(size_t i = 0; i < dst_c; ++i) {
            host_bias_f16[i] = float_to_half_bits(static_cast<float>(i % 10) * 0.1f);
        }

        // ======================= FIX 2: Correct buffer initialization =======================
        // Replicate the scale values for each of the `block_num` blocks.
        for (int i = 0; i < block_num; ++i) {
            for(size_t j = 0; j < dst_c; ++j) {
                // The offset for the current block is i * dst_c_padded
                host_dequant_f16[i * dst_c_padded + j] = float_to_half_bits(0.01f);
            }
        }
        // ====================================================================================

        std::cout << "Packing INT4 weights for Image2D layout..." << std::endl;
        std::vector<uint8_t> host_weight_image_buffer;
        pack_weights_to_image_buffer_int4(host_weight_image_buffer, host_weight_oic_packed, dst_c, src_c, dst_c_padded, src_c_padded);

        // W4A16 CHANGE: Buffer sizes now use sizeof(cl_half)
        size_t nhwc_input_size = host_input_nhwc_f16.size() * sizeof(cl_half);
        size_t packed_tensor_size = (size_t)bhw_padded * std::max(src_c_padded, dst_c_padded) * sizeof(cl_half);
        
        // ======================= FIX 3: Correct buffer size calculation =======================
        size_t dequant_buf_size = host_dequant_f16.size() * sizeof(cl_half); // This now correctly reflects the larger size
        // ======================================================================================

        size_t bias_buf_size = host_bias_f16.size() * sizeof(cl_half);
        size_t nhwc_output_size = (size_t)bhw_padded * dst_c_padded * sizeof(cl_half);

        cl::Buffer input_nhwc_buffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, nhwc_input_size, host_input_nhwc_f16.data());
        cl::Buffer packed_input_buffer(context, CL_MEM_READ_WRITE, packed_tensor_size);
        cl::Buffer dequant_buffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, dequant_buf_size, host_dequant_f16.data());
        cl::Buffer bias_buffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, bias_buf_size, host_bias_f16.data());
        cl::Buffer packed_output_buffer(context, CL_MEM_READ_WRITE, packed_tensor_size);
        cl::Buffer output_nhwc_buffer(context, CL_MEM_WRITE_ONLY, nhwc_output_size);

        cl::ImageFormat weight_image_format(CL_RGBA, CL_SIGNED_INT32);
        cl::Image2D weight_image(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                 weight_image_format,
                                 (src_c_padded + 3) / 4,
                                 (dst_c_padded + 7) / 8,
                                 0,
                                 host_weight_image_buffer.data());

        // Set Kernel Arguments (unchanged logic, but buffers now point to FP16 data)
        cl::NDRange pack_global_size((bhw_padded / 4), (src_c_padded / 4));
        kernel_pack.setArg(0, (int)pack_global_size.get()[0]);
        kernel_pack.setArg(1, (int)pack_global_size.get()[1]);
        kernel_pack.setArg(2, input_nhwc_buffer);
        kernel_pack.setArg(3, packed_input_buffer);
        kernel_pack.setArg(4, bhw);
        kernel_pack.setArg(5, src_c_padded);

        const int x_tiles = bhw_padded / BATCH_PACK;
        const int y_tiles = dst_c_padded / DST_C_PACK;
        cl::NDRange compute_global_size(x_tiles, y_tiles);

        // ======================= 核心修正点 2: 计算正确的 block_dim =======================
        // blockDim 参数应该是每个块的维度，而不是总维度
        const int block_dim = src_c_padded / block_num;
        // ==============================================================================

        int compute_arg_idx = 0;
        kernel_compute.setArg(compute_arg_idx++, (int)compute_global_size.get()[0]);
        kernel_compute.setArg(compute_arg_idx++, (int)compute_global_size.get()[1]);
        kernel_compute.setArg(compute_arg_idx++, packed_input_buffer);
        kernel_compute.setArg(compute_arg_idx++, weight_image);
        kernel_compute.setArg(compute_arg_idx++, dequant_buffer);
        kernel_compute.setArg(compute_arg_idx++, bias_buffer);
        kernel_compute.setArg(compute_arg_idx++, packed_output_buffer);
        kernel_compute.setArg(compute_arg_idx++, bhw);
        kernel_compute.setArg(compute_arg_idx++, dst_c_padded);
        kernel_compute.setArg(compute_arg_idx++, src_c_padded);
        kernel_compute.setArg(compute_arg_idx++, block_num);
        // ======================= 核心修正点 3: 传递正确的 block_dim =======================
        kernel_compute.setArg(compute_arg_idx++, block_dim); // 原来是 src_c_padded，这是错误的
        // ==============================================================================
        kernel_compute.setArg(compute_arg_idx++, 1.0f);

        cl::NDRange unpack_global_size((bhw_padded / 4), (dst_c_padded / 4));
        kernel_unpack.setArg(0, (int)unpack_global_size.get()[0]);
        kernel_unpack.setArg(1, (int)unpack_global_size.get()[1]);
        kernel_unpack.setArg(2, packed_output_buffer);
        kernel_unpack.setArg(3, output_nhwc_buffer);
        kernel_unpack.setArg(4, bhw);
        kernel_unpack.setArg(5, dst_c);
        kernel_unpack.setArg(6, dst_c_padded);

        // Execute Pipeline
        queue.enqueueNDRangeKernel(kernel_pack, cl::NullRange, pack_global_size, cl::NullRange);
        queue.enqueueNDRangeKernel(kernel_compute, cl::NullRange, compute_global_size, cl::NullRange);
        queue.enqueueNDRangeKernel(kernel_unpack, cl::NullRange, unpack_global_size, cl::NullRange);
        queue.finish();

        // ======================= MODIFICATION 2 =======================
        // Use OpenCL events for precise timing.
        const int iterations = 10;
        double total_gpu_time_ms = 0.0;
        double total_pack_time_ms = 0.0;
        double total_compute_time_ms = 0.0;
        double total_unpack_time_ms = 0.0;

        std::cout << "Starting benchmark with " << iterations << " iterations..." << std::endl;

        for (int i = 0; i < iterations; ++i) {
            cl::Event pack_event, compute_event, unpack_event;

            // Enqueue kernels and associate them with event objects
            queue.enqueueNDRangeKernel(kernel_pack, cl::NullRange, pack_global_size, cl::NullRange, nullptr, &pack_event);
            queue.enqueueNDRangeKernel(kernel_compute, cl::NullRange, compute_global_size, cl::NullRange, nullptr, &compute_event);
            queue.enqueueNDRangeKernel(kernel_unpack, cl::NullRange, unpack_global_size, cl::NullRange, nullptr, &unpack_event);

            // Wait for the last event in the chain to complete
            unpack_event.wait();

            // Get profiling info. Times are in nanoseconds.
            cl_ulong pack_start = pack_event.getProfilingInfo<CL_PROFILING_COMMAND_START>();
            cl_ulong pack_end = pack_event.getProfilingInfo<CL_PROFILING_COMMAND_END>();

            cl_ulong compute_start = compute_event.getProfilingInfo<CL_PROFILING_COMMAND_START>();
            cl_ulong compute_end = compute_event.getProfilingInfo<CL_PROFILING_COMMAND_END>();

            cl_ulong unpack_start = unpack_event.getProfilingInfo<CL_PROFILING_COMMAND_START>();
            cl_ulong unpack_end = unpack_event.getProfilingInfo<CL_PROFILING_COMMAND_END>();

            // Accumulate times in milliseconds
            total_pack_time_ms += (pack_end - pack_start) / 1e6;
            total_compute_time_ms += (compute_end - compute_start) / 1e6;
            total_unpack_time_ms += (unpack_end - unpack_start) / 1e6;

            // Total pipeline time on GPU is from the start of the first kernel
            // to the end of the last kernel.
            total_gpu_time_ms += (unpack_end - pack_start) / 1e6;
        }
        // ==============================================================

        // W4A16 CHANGE: Read back FP16 results from GPU
        std::vector<cl_half> gpu_output_nhwc_f16(nhwc_output_size / sizeof(cl_half));
        queue.enqueueReadBuffer(output_nhwc_buffer, CL_TRUE, 0, nhwc_output_size, gpu_output_nhwc_f16.data());

        /*
        // ===== Verification =====
        std::cout << "\n--- Verification ---" << std::endl;
        std::cout << "Running CPU calculation for verification..." << std::endl;
        std::vector<float> cpu_output_nhwc;
        verify_on_cpu(cpu_output_nhwc, host_input_nhwc_f32, host_weight_oic_packed, host_dequant_f16, host_bias_f16, bhw, dst_c, src_c);

        int error_count = 0;
        double max_diff = 0.0;
        // W4A16 CHANGE: Increased tolerance due to lower precision of FP16 arithmetic
        const double tolerance = 5e-2;

        for (int i=0; i<bhw; ++i) {
            for (int j=0; j<dst_c; ++j) {
                size_t index = i * dst_c_padded + j;
                size_t cpu_index = i * dst_c + j;
                // W4A16 CHANGE: Convert GPU's FP16 result to FP32 before comparison
                float gpu_val = half_bits_to_float(gpu_output_nhwc_f16[index]);
                float cpu_val = cpu_output_nhwc[cpu_index];
                double diff = std::abs(gpu_val - cpu_val);

                if (diff > tolerance) {
                    if (error_count < 10) {
                        std::cerr << std::fixed << std::setprecision(6)
                                  << "Mismatch at (bhw=" << i << ", c=" << j << "): "
                                  << "GPU=" << gpu_val << ", CPU=" << cpu_val
                                  << ", Diff=" << diff << std::endl;
                    }
                    error_count++;
                    max_diff = std::max(max_diff, diff);
                }
            }
        }

        if (error_count == 0) {
            std::cout << "Verification PASSED! GPU and CPU results match within tolerance." << std::endl;
        } else {
            std::cerr << "Verification FAILED! Found " << error_count << " mismatches." << std::endl;
            std::cerr << "Maximum difference: " << max_diff << std::endl;
        }
        */

        std::cout << "\n--- First 16 values (NHWC layout) ---" << std::endl;
        std::cout << "GPU: ";
        for (int i = 0; i < 16 && i < bhw*dst_c; ++i) {
            int b = i / dst_c;
            int c = i % dst_c;
            std::cout << half_bits_to_float(gpu_output_nhwc_f16[b*dst_c_padded + c]) << " ";
        }
        std::cout << std::endl;
        /*
        std::cout << "\nCPU: ";
        for (int i = 0; i < 16 && i < cpu_output_nhwc.size(); ++i) std::cout << cpu_output_nhwc[i] << " ";
        std::cout << std::endl;
        */

        // ======================= MODIFICATION FOR GFLOPS =======================
        // Calculate and report performance metrics
        double avg_compute_time_ms = total_compute_time_ms / iterations;

        // Calculate total theoretical operations for the matrix multiplication
        // GEMM Ops = 2 * M * N * K
        // Here, M=bhw, N=dst_c, K=src_c
        double total_ops = 2.0 * bhw * dst_c * src_c;

        // GFLOPS = (Total Operations / 1,000,000,000) / (Time in Seconds)
        double gflops = (avg_compute_time_ms > 0) ? (total_ops / 1e9) / (avg_compute_time_ms / 1000.0) : 0.0;

        std::cout << "\n----------------------------------------" << std::endl;
        std::cout << " Performance Summary" << std::endl;
        std::cout << "----------------------------------------" << std::endl;
        std::cout << std::fixed << std::setprecision(4);
        std::cout << "Average GPU Pipeline Time : " << total_gpu_time_ms / iterations << " ms" << std::endl;
        std::cout << " - Avg Pack Kernel Time : " << total_pack_time_ms / iterations << " ms" << std::endl;
        std::cout << " - Avg Compute Kernel Time: " << avg_compute_time_ms << " ms" << std::endl;
        std::cout << " - Avg Unpack Kernel Time: " << total_unpack_time_ms / iterations << " ms" << std::endl;
        std::cout << "----------------------------------------" << std::endl;
        std::cout << "Effective GFLOPS (Compute): " << std::fixed << std::setprecision(2) << gflops << " GFLOPS" << std::endl;
        std::cout << "----------------------------------------" << std::endl;
        // =======================================================================

    } catch (const cl::BuildError& e) {
        std::cerr << "OpenCL Build Error: " << e.what() << std::endl;
        for (const auto& log : e.getBuildLog()) {
            std::cerr << " Device: " << log.first.getInfo<CL_DEVICE_NAME>() << std::endl;
            std::cerr << " Log: " << log.second << std::endl;
        }
        return 1;
    } catch (const cl::Error& e) {
        std::cerr << "OpenCL Error: " << e.what() << " (" << e.err() << ")" << std::endl;
        return 1;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}