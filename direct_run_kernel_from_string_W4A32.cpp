// direct_run_kernel_from_string.cpp
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

// =======================================================================
// ==  MNN Kernel Source String                                         ==
// =======================================================================
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
" COMPUTE_FLOAT8 scale=CONVERT_COMPUTE_FLOAT8(convert_float8(vload8(0,dequantScaleOffset+(ic/blockDim)*outputChannel4Align+oc))/coef);\n"
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
" COMPUTE_FLOAT8 scale=CONVERT_COMPUTE_FLOAT8(convert_float8(vload8(0,dequantScaleOffset+(ic/blockDim)*outputChannel4Align+oc))/coef);\n"
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
" COMPUTE_FLOAT8 scale=CONVERT_COMPUTE_FLOAT8(convert_float8(vload8(0,dequantScaleOffset+(out_c_idx << 2)+i*dstChannelAlign))/coef);\n"
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
" COMPUTE_FLOAT8 scale=CONVERT_COMPUTE_FLOAT8(convert_float8(vload8(0,dequantScaleOffset+(out_c_idx << 2)+i*dstChannelAlign))/coef);\n"
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
" scale.s01234567=CONVERT_COMPUTE_FLOAT8(convert_float8(vload8(0,dequantScaleOffset+(out_c_idx << 2)+i*dstChannelAlign))/coef);\n"
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
" scale.s01234567=CONVERT_COMPUTE_FLOAT8(convert_float8(vload8(0,dequantScaleOffset+(out_c_idx << 2)+i*dstChannelAlign))/coef);\n"
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


// Helper function to get platform and device
void select_opencl_device(cl::Platform& platform, cl::Device& device) {
    std::vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);

    if (platforms.empty()) {
        throw std::runtime_error("No OpenCL platforms found.");
    }

    // Simple selection: just use the first platform
    platform = platforms[0];
    std::vector<cl::Device> devices;
    platform.getDevices(CL_DEVICE_TYPE_GPU, &devices);

    if (devices.empty()) {
        platform.getDevices(CL_DEVICE_TYPE_CPU, &devices);
        if(devices.empty()){
            throw std::runtime_error("No OpenCL GPU or CPU devices found.");
        }
    }
    
    // Simple selection: just use the first device
    device = devices[0];
}

// =======================================================================
// ==  NEW: INT4 Weight Packing Helper for Image2D layout               ==
// =======================================================================
// This function packs weights from a logical [OC][IC] layout (where each byte holds two 4-bit weights)
// into the swizzled memory layout expected by the gemm_b4_c8_int4_buf kernel's read_imagei.
void pack_weights_to_image_buffer_int4(
    std::vector<uint8_t>& dst_image_buffer,
    const std::vector<uint8_t>& src_weights_packed,
    int oc, int ic,
    int oc_padded, int ic_padded)
{
    // Image dimensions as seen by the kernel
    const int image_width = (ic_padded + 3) / 4;
    const int image_height = (oc_padded + 7) / 8;

    // Each pixel is 16 bytes (RGBA, SIGNED_INT32)
    const int bytes_per_pixel = 16;
    dst_image_buffer.assign(image_width * image_height * bytes_per_pixel, 0);

    // Iterate through each logical weight
    for (int o = 0; o < oc; ++o) {
        for (int i = 0; i < ic; ++i) {
            // 1. Get the source 4-bit weight value.
            // We assume the source is packed as [oc][ic], with two ic values per byte.
            int src_byte_idx = o * ((ic + 1) / 2) + (i / 2);
            uint8_t src_byte = src_weights_packed[src_byte_idx];
            
            // In MNN, weights are packed as [i=1, i=0], [i=3, i=2], etc.
            // High 4 bits for odd index, low 4 bits for even index.
            // Here, we'll use a simpler convention for clarity: low bits for even, high for odd.
            uint8_t weight_4bit = (i % 2 == 0) ? (src_byte & 0x0F) : (src_byte >> 4);

            // 2. Determine the destination pixel's coordinates (x, y) in the Image2D.
            int x = i / 4;
            int y = o / 8;

            // 3. Determine the position *within* that 128-bit (16-byte) pixel.
            // The kernel's access pattern is [ic_inner][oc_inner].
            int ic_inner = i % 4; // which of the 4 input channels this pixel covers
            int oc_inner = o % 8; // which of the 8 output channels this pixel covers

            // The 16-byte pixel is structured as four 4-byte chunks. Each chunk is for one ic_inner.
            int chunk_offset_in_pixel = ic_inner * 4;

            // Within a 4-byte chunk, the 8 weights for oc_inner are stored.
            // Two 4-bit weights per byte.
            int byte_offset_in_chunk = oc_inner / 2;

            // 4. Calculate the final destination byte index.
            int dst_pixel_base_idx = (y * image_width + x) * bytes_per_pixel;
            int final_dst_byte_idx = dst_pixel_base_idx + chunk_offset_in_pixel + byte_offset_in_chunk;

            // 5. Place the 4-bit weight into the correct half of the destination byte.
            // Since we initialized dst_image_buffer to 0, we can use |=
            // The kernel macro UCHAR4_TO_CHAR8 expects a specific order.
            // It unpacks (byte >> 4) first, then (byte & 15).
            // This means the weight for oc_inner=0 must be in the high bits of the first byte,
            // oc_inner=1 in the low bits, oc_inner=2 in high bits of second byte etc.
            if (oc_inner % 2 == 0) { // Even oc_inner goes into the high 4 bits
                dst_image_buffer[final_dst_byte_idx] |= (weight_4bit << 4);
            } else { // Odd oc_inner goes into the low 4 bits
                dst_image_buffer[final_dst_byte_idx] |= weight_4bit;
            }
        }
    }
}


// CPU verification function (operates on simple, logical layouts)
void verify_on_cpu(
    std::vector<float>& cpu_output,
    const std::vector<float>& input_nhwc,
    const std::vector<uint8_t>& weight_oic_packed,
    const std::vector<float>& dequant,
    const std::vector<float>& bias,
    int bhw, int dst_c, int src_c)
{
    cpu_output.assign(bhw * dst_c, 0.0f);

    for (int i = 0; i < bhw; ++i) {
        for (int j = 0; j < dst_c; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < src_c; ++k) {
                float in_val = input_nhwc[i * src_c + k];
                
                // Unpack the 4-bit weight
                int src_byte_idx = j * ((src_c + 1) / 2) + (k / 2);
                uint8_t src_byte = weight_oic_packed[src_byte_idx];
                uint8_t weight_4bit_unsigned = (k % 2 == 0) ? (src_byte & 0x0F) : (src_byte >> 4);
                
                // The kernel macro subtracts 8 to convert from [0, 15] to [-8, 7]
                char w_int4 = static_cast<char>(weight_4bit_unsigned) - 8;

                float scale = dequant[j]; // Assuming non-asymmetric for simplicity
                float w_dequantized = static_cast<float>(w_int4) * scale;
                
                sum += in_val * w_dequantized;
            }
            cpu_output[i * dst_c + j] = sum + bias[j];
        }
    }
}

extern "C" int initOpenCL();

int main(int argc, char** argv) {
    if (initOpenCL() == 0) {
        std::cerr << "Failed to initialize OpenCL loader." << std::endl;
        return -1;
    }
    if (argc != 4) {
        std::cerr << "Usage: " << argv[0] << " <BHW> <DST_C> <SRC_C>" << std::endl;
        return 1;
    }

    try {
        const int bhw = std::stoi(argv[1]);
        const int dst_c = std::stoi(argv[2]);
        const int src_c = std::stoi(argv[3]);
        
        const std::string pack_kernel_name = "gemm_nhwc_to_c4nhw4";
        const std::string compute_kernel_name = "gemm_b4_c8_int4_buf"; // Use the INT4 kernel
        const std::string unpack_kernel_name = "gemm_c4nhw4_to_nhwc";

        std::cout << "Benchmarking & Verifying OpenCL Pipeline (INT4 weights on Image2D)" << std::endl;
        std::cout << "Input Dimensions: BHW=" << bhw << ", DST_C=" << dst_c << ", SRC_C=" << src_c << std::endl;
        
        cl::Platform platform;
        cl::Device device;
        select_opencl_device(platform, device);

        if (device.getInfo<CL_DEVICE_IMAGE_SUPPORT>() == CL_FALSE) {
            throw std::runtime_error("Selected OpenCL device does not support images.");
        }

        cl::Context context(device);
        cl::CommandQueue queue(context, device, cl::QueueProperties::Profiling);

        std::cout << "Using Platform: " << platform.getInfo<CL_PLATFORM_NAME>() << std::endl;
        std::cout << "Using Device: " << device.getInfo<CL_DEVICE_NAME>() << std::endl;

        std::stringstream build_opts;
        build_opts << "-D MNN_SUPPORT_FP16 ";
        build_opts << "-DCOMPUTE_FLOAT=float -DCOMPUTE_FLOAT4=float4 -DCOMPUTE_FLOAT8=float8 -DCOMPUTE_FLOAT16=float16 ";
        build_opts << "-DCONVERT_COMPUTE_FLOAT=convert_float -DCONVERT_COMPUTE_FLOAT4=convert_float4 -DCONVERT_COMPUTE_FLOAT8=convert_float8 -DCONVERT_COMPUTE_FLOAT16=convert_float16 ";
        build_opts << "-DFLOAT=float -DFLOAT4=float4 -DFLOAT8=float8 -DFLOAT16=float16 "; // Note: FLOAT16 is float16
        build_opts << "-DCONVERT_FLOAT4=convert_float4 -DCONVERT_FLOAT8=convert_float8 -DCONVERT_FLOAT16=convert_float16 ";
        
        build_opts << "-D INPUT_CHANNEL_LEAVES_NUM=0 ";
        build_opts << "-D INPUT_BATCH_LEAVES_NUM=0 ";
        build_opts << "-D USE_IMAGE "; // CRITICAL: Enable the Image2D path in the kernel
        build_opts << "-D QUANT_BIT=4 "; // CRITICAL: Specify INT4 quantization

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
        
        std::vector<float> host_input_nhwc(bhw_padded * src_c_padded, 0.f);
        std::vector<uint8_t> host_weight_oic_packed((dst_c * src_c + 1) / 2);
        std::vector<float> host_bias(dst_c_padded, 0.f);
        std::vector<float> host_dequant(dst_c_padded, 0.f); // Non-asymmetric for simplicity

        std::cout << "Initializing host data..." << std::endl;
        for(int b=0; b<bhw; ++b) for(int c=0; c<src_c; ++c) host_input_nhwc[b*src_c_padded + c] = (static_cast<float>((b*src_c+c) % 5) - 2.0f) * 0.5f;
        
        // Initialize packed 4-bit weights
        for(size_t i = 0; i < host_weight_oic_packed.size(); ++i) {
            uint8_t val1 = (i * 2) % 16; // Value in [0, 15]
            uint8_t val2 = (i * 2 + 1) % 16;
            host_weight_oic_packed[i] = (val2 << 4) | val1;
        }

        for(size_t i = 0; i < dst_c; ++i) host_bias[i] = static_cast<float>(i % 10) * 0.1f;
        for(size_t i = 0; i < dst_c; ++i) host_dequant[i] = 0.01f;

        // Pack weights into a linear buffer suitable for Image2D creation
        std::cout << "Packing INT4 weights for Image2D layout..." << std::endl;
        std::vector<uint8_t> host_weight_image_buffer;
        pack_weights_to_image_buffer_int4(host_weight_image_buffer, host_weight_oic_packed, dst_c, src_c, dst_c_padded, src_c_padded);

        // ===== GPU Buffers and Images =====
        size_t nhwc_input_size = host_input_nhwc.size() * sizeof(float);
        size_t packed_tensor_size = (size_t)bhw_padded * std::max(src_c_padded, dst_c_padded) * sizeof(float);
        size_t dequant_buf_size = host_dequant.size() * sizeof(float);
        size_t bias_buf_size = host_bias.size() * sizeof(float);
        size_t nhwc_output_size = (size_t)bhw_padded * dst_c_padded * sizeof(float);

        cl::Buffer input_nhwc_buffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, nhwc_input_size, host_input_nhwc.data());
        cl::Buffer packed_input_buffer(context, CL_MEM_READ_WRITE, packed_tensor_size);
        cl::Buffer dequant_buffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, dequant_buf_size, host_dequant.data());
        cl::Buffer bias_buffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, bias_buf_size, host_bias.data());
        cl::Buffer packed_output_buffer(context, CL_MEM_READ_WRITE, packed_tensor_size);
        cl::Buffer output_nhwc_buffer(context, CL_MEM_WRITE_ONLY, nhwc_output_size);

        // *** Create Image2D for weights ***
        cl::ImageFormat weight_image_format(CL_RGBA, CL_SIGNED_INT32);
        cl::Image2D weight_image(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
            weight_image_format,
            (src_c_padded + 3) / 4, // Image Width for INT4
            (dst_c_padded + 7) / 8, // Image Height
            0,
            host_weight_image_buffer.data());

        // ===== Set Kernel Arguments =====
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
        kernel_compute.setArg(compute_arg_idx++, 1); // blockNum
        kernel_compute.setArg(compute_arg_idx++, src_c_padded); // blockDim
        kernel_compute.setArg(compute_arg_idx++, 1.0f); // coef

        cl::NDRange unpack_global_size((bhw_padded / 4), (dst_c_padded / 4));
        kernel_unpack.setArg(0, (int)unpack_global_size.get()[0]);
        kernel_unpack.setArg(1, (int)unpack_global_size.get()[1]);
        kernel_unpack.setArg(2, packed_output_buffer);
        kernel_unpack.setArg(3, output_nhwc_buffer);
        kernel_unpack.setArg(4, bhw);
        kernel_unpack.setArg(5, dst_c);
        kernel_unpack.setArg(6, dst_c_padded);

        // ===== Execute Pipeline =====
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

        std::vector<float> gpu_output_nhwc(nhwc_output_size / sizeof(float));
        queue.enqueueReadBuffer(output_nhwc_buffer, CL_TRUE, 0, nhwc_output_size, gpu_output_nhwc.data());

        // ===== Verification =====
        std::cout << "\n--- Verification ---" << std::endl;
        std::cout << "Running CPU calculation for verification..." << std::endl;
        std::vector<float> cpu_output_nhwc;
        verify_on_cpu(cpu_output_nhwc, host_input_nhwc, host_weight_oic_packed, host_dequant, host_bias, bhw, dst_c, src_c);
        
        int error_count = 0;
        double max_diff = 0.0;
        const double tolerance = 1e-2; // Tolerance might need to be higher for INT4

        for (int i=0; i<bhw; ++i) {
            for (int j=0; j<dst_c; ++j) {
                size_t index = i * dst_c_padded + j;
                size_t cpu_index = i * dst_c + j;
                float gpu_val = gpu_output_nhwc[index];
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
            std::cout << "Verification PASSED! GPU and CPU results match." << std::endl;
        } else {
            std::cerr << "Verification FAILED! Found " << error_count << " mismatches." << std::endl;
            std::cerr << "Maximum difference: " << max_diff << std::endl;
        }

        std::cout << "\n--- First 16 values (NHWC layout) ---" << std::endl;
        std::cout << "GPU: ";
        for (int i = 0; i < 16 && i < bhw*dst_c; ++i) {
            int b = i / dst_c;
            int c = i % dst_c;
            std::cout << gpu_output_nhwc[b*dst_c_padded + c] << " ";
        }
        std::cout << "\nCPU: ";
        for (int i = 0; i < 16 && i < cpu_output_nhwc.size(); ++i) std::cout << cpu_output_nhwc[i] << " ";
        std::cout << std::endl;

        // ======================= MODIFICATION 3 =======================
        // Report the new, more accurate times.
        std::cout << "\n----------------------------------------" << std::endl;
        std::cout << std::fixed << std::setprecision(4);
        std::cout << "Average GPU Pipeline Time : " << total_gpu_time_ms / iterations << " ms" << std::endl;
        std::cout << "  - Avg Pack Kernel Time  : " << total_pack_time_ms / iterations << " ms" << std::endl;
        std::cout << "  - Avg Compute Kernel Time: " << total_compute_time_ms / iterations << " ms" << std::endl;
        std::cout << "  - Avg Unpack Kernel Time: " << total_unpack_time_ms / iterations << " ms" << std::endl;
        std::cout << "----------------------------------------" << std::endl;
        // ==============================================================

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