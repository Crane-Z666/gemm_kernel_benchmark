// direct_run_kernel_from_string_w8a16.cpp
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

// W8A16 CHANGE: 定义 cl_half 用于主机端存储，继承自 W4A16 版本
using cl_half = uint16_t;

// =======================================================================
// == FP32 <-> FP16 Conversion Utilities (from W4A16)                   ==
// =======================================================================
// 将 32-bit float 转换为 16-bit half-float
cl_half float_to_half_bits(float f) {
    union {
        float f;
        uint32_t u;
    } converter = {f};
    uint32_t u = converter.u;

    uint32_t sign = (u >> 31) & 0x1;
    int32_t exp = ((u >> 23) & 0xff) - 127;
    uint32_t mant = u & 0x7fffff;

    if (exp > 15) { // 上溢 -> 无穷大
        return (sign << 15) | 0x7c00;
    }
    if (exp < -14) { // 下溢 -> 非规格化数或零
        mant |= 0x800000; // 添加隐藏位
        int shift = -14 - exp;
        if (shift > 24) {
            return sign << 15; // 太小，刷新为零
        }
        mant >>= shift;
        return (sign << 15) | (mant >> 13);
    }
    return (sign << 15) | ((exp + 15) << 10) | (mant >> 13);
}

// 将 16-bit half-float 转换为 32-bit float
float half_bits_to_float(cl_half h) {
    uint32_t sign = (h >> 15) & 0x1;
    uint32_t exp = (h >> 10) & 0x1f;
    uint32_t mant = h & 0x3ff;

    union {
        uint32_t u;
        float f;
    } converter;

    if (exp == 0) { // 非规格化数或零
        if (mant == 0) { // 零
            converter.u = sign << 31;
        } else { // 非规格化数
            exp = 1 - 15;
            while ((mant & 0x400) == 0) {
                mant <<= 1;
                exp--;
            }
            mant &= 0x3ff;
            converter.u = (sign << 31) | ((exp + 127) << 23) | (mant << 13);
        }
    } else if (exp == 31) { // 无穷大或 NaN
        converter.u = (sign << 31) | (0xff << 23) | (mant << 13);
    } else { // 规格化数
        converter.u = (sign << 31) | ((exp - 15 + 127) << 23) | (mant << 13);
    }
    return converter.f;
}


// 帮助函数，用于选择 OpenCL 平台和设备
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
// == W8A16 CHANGE: INT8 Weight Packing Helper for Image2D (from W8A32) ==
// =======================================================================
// 这个函数将逻辑布局为 [OC][IC] 的 INT8 权重打包到一个线性缓冲区中，
// 该缓冲区将用于为内核创建 2D Image。
void pack_weights_to_image_buffer_int8(
    std::vector<char>& dst, 
    const std::vector<char>& src, 
    int oc, int ic, 
    int oc_padded, int ic_padded) 
{
    // 内核看到的图像尺寸
    const int image_height = oc_padded / 8;
    const int image_width = ic_padded / 2;
    
    // 每个像素是 16 字节 (CL_RGBA, CL_SIGNED_INT32)
    const int bytes_per_pixel = 16;
    dst.assign(image_width * image_height * bytes_per_pixel, 0);

    for (int o = 0; o < oc; ++o) {
        for (int i = 0; i < ic; ++i) {
            // 逻辑源索引
            int src_idx = o * ic + i;

            // 将逻辑坐标 (o, i) 映射到图像坐标 (x, y) 和像素内偏移
            int y = o / 8;
            int x = i / 2;

            int o_inner = o % 8;
            int i_inner = i % 2;

            // 内核读取一个像素（16字节），并获取 8 个输出通道和 2 个输入通道的权重。
            // 16字节像素内的内存布局是 [i_inner][o_inner]。
            int dst_pixel_offset = (y * image_width + x) * bytes_per_pixel;
            int dst_inner_offset = i_inner * 8 + o_inner;
            int dst_idx = dst_pixel_offset + dst_inner_offset;
            
            dst[dst_idx] = src[src_idx];
        }
    }
}


// =======================================================================
// == W8A16 CHANGE: CPU verification function                         ==
// =======================================================================
// 此函数在简单、逻辑布局上操作，用于验证结果。
// 它接收 FP16 的偏置/反量化参数，并将其转换回 FP32 进行计算。
// 它仍然使用 FP32 进行输入和累加以获得“黄金”参考值。
void verify_on_cpu(
    std::vector<float>& cpu_output,
    const std::vector<float>& input_nhwc_f32,
    const std::vector<char>& weight_oic, // W8: 权重是 INT8
    const std::vector<cl_half>& dequant_f16, // A16: 反量化尺度是 FP16
    const std::vector<cl_half>& bias_f16,    // A16: 偏置是 FP16
    int bhw, int dst_c, int src_c)
{
    cpu_output.assign(bhw * dst_c, 0.0f);

    for (int i = 0; i < bhw; ++i) {
        for (int j = 0; j < dst_c; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < src_c; ++k) {
                // 注意：使用原始的 FP32 输入进行 CPU 验证
                float in_val = input_nhwc_f32[i * src_c + k];
                
                // W8: 直接读取 INT8 权重
                char w_int8 = weight_oic[j * src_c + k];

                // A16: 将 FP16 尺度转换为 FP32 进行计算
                float scale = half_bits_to_float(dequant_f16[j]);
                float w_dequantized = static_cast<float>(w_int8) * scale;
                
                sum += in_val * w_dequantized;
            }
            // A16: 将 FP16 偏置转换为 FP32 进行计算
            cpu_output[i * dst_c + j] = sum + half_bits_to_float(bias_f16[j]);
        }
    }
}

// =======================================================================
// == 新增：日志记录函数                                                ==
// =======================================================================
/*
 * @brief 将性能数据记录到 CSV 文件中，该文件可以由 Excel 打开。
 *
 * @param filename 要写入的文件名 (例如 "data.xlsx")。
 * @param compute_float 计算精度 (例如 "FP32")。
 * @param quant 量化方案 (例如 "W4A16")。
 * @param sequence BHW 值。
 * @param cin 输入通道数。
 * @param cout 输出通道数。
 * @param block_num BlockNum 值。
 * @param pack_latency Pack Kernel 的平均延迟 (ms)。
 * @param compute_latency Compute Kernel 的平均延迟 (ms)。
 * @param unpack_latency Unpack Kernel 的平均延迟 (ms)。
 * @param gflops 计算出的 GFLOPS。
 */
void log_to_excel_csv(
    const std::string& filename,
    const std::string& compute_float,
    const std::string& quant,
    int sequence,
    int cin,
    int cout,
    int block_num,
    double pack_latency,
    double compute_latency,
    double unpack_latency,
    double gflops)
{
    // 1. 检查文件是否存在以决定是否写入表头
    std::ifstream file_check(filename);
    bool file_exists = file_check.good();
    file_check.close();

    // 2. 以追加模式打开文件。如果文件不存在，这会自动创建它。
    std::ofstream log_file(filename, std::ios_base::app);
    if (!log_file.is_open()) {
        std::cerr << "Error: Could not open log file " << filename << " for writing." << std::endl;
        return;
    }

    // 3. 如果文件是新创建的，则写入表头
    if (!file_exists) {
        log_file << "COMPUTE_FLOAT,Quant,Sequence,Cin,Cout,BlockNum,"
                 << "Pack Latency/ms,Compute Latency/ms,Unpack Latency/ms,GFLOPS\n";
    }

    // 4. 写入数据行
    log_file << std::fixed << std::setprecision(4) // 设置浮点数格式
             << compute_float << ","
             << quant << ","
             << sequence << ","
             << cin << ","
             << cout << ","
             << block_num << ","
             << pack_latency << ","
             << compute_latency << ","
             << unpack_latency << ","
             << std::setprecision(2) // GFLOPS 使用两位小数
             << gflops << "\n";

    log_file.close();
    std::cout << "\nSuccessfully logged results to " << filename << std::endl;
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
        // W8A16 CHANGE: 使用 gemm_b4_c8_int8_buf 内核
        const std::string compute_kernel_name = "gemm_b4_c8_int8_buf";
        const std::string unpack_kernel_name = "gemm_c4nhw4_to_nhwc";

        // W8A16 CHANGE: 更新标题
        std::cout << "Benchmarking & Verifying OpenCL Pipeline (W8A16 - INT8 weights, FP16 activations on Image2D)" << std::endl;
        std::cout << "Input Dimensions: BHW=" << bhw << ", DST_C=" << dst_c << ", SRC_C=" << src_c << ", block_num=" << block_num << std::endl;
        
        cl::Platform platform;
        cl::Device device;
        select_opencl_device(platform, device);

        if (device.getInfo<CL_DEVICE_IMAGE_SUPPORT>() == CL_FALSE) {
            throw std::runtime_error("Selected OpenCL device does not support images.");
        }
        // W8A16 CHANGE: 检查设备是否支持 FP16，继承自 W4A16
        std::string extensions = device.getInfo<CL_DEVICE_EXTENSIONS>();
        if (extensions.find("cl_khr_fp16") == std::string::npos) {
            throw std::runtime_error("Selected OpenCL device does not support cl_khr_fp16 extension.");
        }

        cl::Context context(device);
        cl::CommandQueue queue(context, device, cl::QueueProperties::Profiling);

        std::cout << "Using Platform: " << platform.getInfo<CL_PLATFORM_NAME>() << std::endl;
        std::cout << "Using Device: " << device.getInfo<CL_DEVICE_NAME>() << std::endl;

        // W8A16 CHANGE: CRITICAL - 更新编译选项以使用 'half' 精度类型
        // 这些选项与 W4A16 版本完全相同，因为它们定义了激活、计算和存储的精度
        std::stringstream build_opts;
        build_opts << "-D MNN_SUPPORT_FP16 ";
        // 将计算类型定义为 half
        build_opts << "-DCOMPUTE_FLOAT=float -DCOMPUTE_FLOAT4=float4 -DCOMPUTE_FLOAT8=float8 -DCOMPUTE_FLOAT16=float16 ";
        build_opts << "-DCONVERT_COMPUTE_FLOAT=convert_float -DCONVERT_COMPUTE_FLOAT4=convert_float4 -DCONVERT_COMPUTE_FLOAT8=convert_float8 -DCONVERT_COMPUTE_FLOAT16=convert_float16 ";
        // 将存储/IO类型定义为 half
        build_opts << "-DFLOAT=half -DFLOAT4=half4 -DFLOAT8=half8 -DFLOAT16=half16 ";
        build_opts << "-DCONVERT_FLOAT4=convert_half4 -DCONVERT_FLOAT8=convert_half8 -DCONVERT_FLOAT16=convert_half16 ";
        
        build_opts << "-D INPUT_CHANNEL_LEAVES_NUM=0 ";
        build_opts << "-D INPUT_BATCH_LEAVES_NUM=0 ";
        build_opts << "-D USE_IMAGE "; // 启用 Image2D 路径

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

        // W8A16 CHANGE: 主机端缓冲区现在是 cl_half (A16) 和 char (W8) 的混合
        // 我们还保留一份输入的 FP32 副本用于 CPU 验证。
        std::vector<float>   host_input_nhwc_f32(bhw_padded * src_c_padded, 0.f);
        std::vector<cl_half> host_input_nhwc_f16(bhw_padded * src_c_padded, 0);
        std::vector<char>    host_weight_oic(dst_c * src_c); // W8: 权重是 INT8
        std::vector<cl_half> host_bias_f16(dst_c_padded, 0); // A16: 偏置是 FP16

        // ======================= FIX 1: Correct buffer size =======================
        // The dequant buffer size must match what the kernel expects: block_num * dst_c_padded
        std::vector<cl_half> host_dequant_f16(block_num * dst_c_padded, 0);
        // ==========================================================================

        std::cout << "Initializing and converting host data (W8 weights, A16 activations)..." << std::endl;
        // A16: 初始化输入数据并转换为 FP16
        for(int b=0; b<bhw; ++b) {
            for(int c=0; c<src_c; ++c) {
                size_t idx = b*src_c_padded + c;
                float val_f32 = (static_cast<float>((b*src_c+c) % 5) - 2.0f) * 0.5f;
                host_input_nhwc_f32[idx] = val_f32;
                host_input_nhwc_f16[idx] = float_to_half_bits(val_f32);
            }
        }
        
        // W8: 初始化 INT8 权重数据
        for(size_t i = 0; i < host_weight_oic.size(); ++i) {
            host_weight_oic[i] = static_cast<char>((i % 25) - 12);
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

        // W8A16 CHANGE: 将 INT8 权重打包到 Image2D 布局中
        std::cout << "Packing INT8 weights for Image2D layout..." << std::endl;
        std::vector<char> host_weight_image_buffer;
        pack_weights_to_image_buffer_int8(host_weight_image_buffer, host_weight_oic, dst_c, src_c, dst_c_padded, src_c_padded);

        // W8A16 CHANGE: 缓冲区大小现在使用 sizeof(cl_half)
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

        // W8A16 CHANGE: 创建用于 INT8 权重的 Image2D
        cl::ImageFormat weight_image_format(CL_RGBA, CL_SIGNED_INT32);
        cl::Image2D weight_image(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
            weight_image_format,
            src_c_padded / 2, // 图像宽度，根据 INT8 打包逻辑
            dst_c_padded / 8, // 图像高度，根据 INT8 打包逻辑
            0,
            host_weight_image_buffer.data());

        // 设置内核参数（逻辑不变，但缓冲区指向 FP16 数据）
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
        kernel_compute.setArg(compute_arg_idx++, weight_image); // 传递 Image2D 对象
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

        // 首次执行以确保所有内容都已编译和加载
        queue.enqueueNDRangeKernel(kernel_pack, cl::NullRange, pack_global_size, cl::NullRange);
        queue.enqueueNDRangeKernel(kernel_compute, cl::NullRange, compute_global_size, cl::NullRange);
        queue.enqueueNDRangeKernel(kernel_unpack, cl::NullRange, unpack_global_size, cl::NullRange);
        queue.finish();

        // 使用 OpenCL 事件进行精确计时
        const int iterations = 10;
        double total_gpu_time_ms = 0.0;
        double total_pack_time_ms = 0.0;
        double total_compute_time_ms = 0.0;
        double total_unpack_time_ms = 0.0;
        
        std::cout << "Starting benchmark with " << iterations << " iterations..." << std::endl;

        for (int i = 0; i < iterations; ++i) {
            cl::Event pack_event, compute_event, unpack_event;

            queue.enqueueNDRangeKernel(kernel_pack, cl::NullRange, pack_global_size, cl::NullRange, nullptr, &pack_event);
            queue.enqueueNDRangeKernel(kernel_compute, cl::NullRange, compute_global_size, cl::NullRange, nullptr, &compute_event);
            queue.enqueueNDRangeKernel(kernel_unpack, cl::NullRange, unpack_global_size, cl::NullRange, nullptr, &unpack_event);

            unpack_event.wait();

            cl_ulong pack_start = pack_event.getProfilingInfo<CL_PROFILING_COMMAND_START>();
            cl_ulong pack_end = pack_event.getProfilingInfo<CL_PROFILING_COMMAND_END>();
            
            cl_ulong compute_start = compute_event.getProfilingInfo<CL_PROFILING_COMMAND_START>();
            cl_ulong compute_end = compute_event.getProfilingInfo<CL_PROFILING_COMMAND_END>();

            cl_ulong unpack_start = unpack_event.getProfilingInfo<CL_PROFILING_COMMAND_START>();
            cl_ulong unpack_end = unpack_event.getProfilingInfo<CL_PROFILING_COMMAND_END>();

            total_pack_time_ms += (pack_end - pack_start) / 1e6;
            total_compute_time_ms += (compute_end - compute_start) / 1e6;
            total_unpack_time_ms += (unpack_end - unpack_start) / 1e6;
            total_gpu_time_ms += (unpack_end - pack_start) / 1e6;
        }

        // W8A16 CHANGE: 从 GPU 读回 FP16 结果
        std::vector<cl_half> gpu_output_nhwc_f16(nhwc_output_size / sizeof(cl_half));
        queue.enqueueReadBuffer(output_nhwc_buffer, CL_TRUE, 0, nhwc_output_size, gpu_output_nhwc_f16.data());

        /*
        // ===== Verification =====
        std::cout << "\n--- Verification ---" << std::endl;
        std::cout << "Running CPU calculation for verification..." << std::endl;
        std::vector<float> cpu_output_nhwc;
        // W8A16 CHANGE: 调用新的 W8A16 CPU 验证函数
        verify_on_cpu(cpu_output_nhwc, host_input_nhwc_f32, host_weight_oic, host_dequant_f16, host_bias_f16, bhw, dst_c, src_c);
        
        int error_count = 0;
        double max_diff = 0.0;
        // W8A16 CHANGE: 由于 FP16 算术的精度较低，使用与 W4A16 相同的较高容差
        const double tolerance = 5e-2;

        for (int i=0; i<bhw; ++i) {
            for (int j=0; j<dst_c; ++j) {
                size_t index = i * dst_c_padded + j;
                size_t cpu_index = i * dst_c + j;
                // W8A16 CHANGE: 在比较前将 GPU 的 FP16 结果转换为 FP32
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
        std::cout << "            Performance Summary" << std::endl;
        std::cout << "----------------------------------------" << std::endl;
        std::cout << std::fixed << std::setprecision(4);
        std::cout << "Average GPU Pipeline Time : " << total_gpu_time_ms / iterations << " ms" << std::endl;
        std::cout << "  - Avg Pack Kernel Time  : " << total_pack_time_ms / iterations << " ms" << std::endl;
        std::cout << "  - Avg Compute Kernel Time: " << avg_compute_time_ms << " ms" << std::endl;
        std::cout << "  - Avg Unpack Kernel Time: " << total_unpack_time_ms / iterations << " ms" << std::endl;
        std::cout << "----------------------------------------" << std::endl;
        std::cout << "Effective GFLOPS (Compute): " << std::fixed << std::setprecision(2) << gflops << " GFLOPS" << std::endl;
        std::cout << "----------------------------------------" << std::endl;
        // =======================================================================

        // ======================= 新增：调用日志记录函数 =======================
        log_to_excel_csv(
            "data.csv",         // 文件名
            "FP32",              // COMPUTE_FLOAT
            "W8A16",             // Quant
            bhw,                 // Sequence
            src_c,               // Cin
            dst_c,               // Cout
            block_num,           // BlockNum
            total_pack_time_ms / iterations,    // Pack Latency/ms
            avg_compute_time_ms, // Compute Latency
            total_unpack_time_ms / iterations,  // Unpack Latency
            gflops               // GFLOPS
        );
        // =======================================================================
        
    } catch (const cl::BuildError& e) {
        std::cerr << "OpenCL Build Error: " << e.what() << std::endl;
        for (const auto& log : e.getBuildLog()) {
            std::cerr << "  Device: " << log.first.getInfo<CL_DEVICE_NAME>() << std::endl;
            std::cerr << "  Log: " << log.second << std::endl;
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