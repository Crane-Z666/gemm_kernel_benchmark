## 文件清单

**CMakeList:** 编译用，**其中的`include_directories`需要改为cl头文件地址**，分成了带ANDROID宏和不带ANDROID宏两部分，因为我的A16不能在电脑上跑，所以不带ANDROID的部分就没编译A16的两个文件

**opencl_loader.cpp:** 用于安卓的交叉运行，动态链接库

#### 以下三个文件带GPU和CPU的计算结果比较，用来测试GPU计算的有效性用

**direct_run_kernel_from_string:** W8A32，带GPU和CPU的计算结果比较

**direct_run_kernel_from_string_W4A32:** W4A32，带GPU和CPU的计算结果比较  

**direct_run_kernel_from_string_W4A16:** W4A16，带GPU和CPU的计算结果比较  

#### 以下两个文件只有GPU的计算，用来测试latency和GFLOPS

**direct_run_kernel_from_string_W4A16_onlyGPU:** W4A16，没有GPU和CPU的计算校验，只有GPU的计算

**direct_run_kernel_from_string_W8A16_onlyGPU:** W8A16，没有GPU和CPU的计算校验，只有GPU的计算

## 编译的宏

```
mkdir build_androd && cd build_android
export ANDROID_NDK_HOME="/opt/android-ndk-r26d/"
cmake ..     -DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK_HOME/build/cmake/android.toolchain.cmake     -DANDROID_ABI=arm64-v8a     -DANDROID_PLATFORM=android-24     -DCMAKE_BUILD_TYPE=Release
```

**注意`ANDROID_NDK_HOME`可能要改成对应NDK的路径**

## push哪些文件

用哪个push哪个就行，没有其它.so文件要另外找和push的

## 执行

需要送入三个参数，分别是sequence length、cout、cin

## 数据来源

激活数据是通过一个简单的数学公式生成的确定性序列，其值在 `[-1.0, 1.0]` 之间循环；

权重数据是通过生成 `[0, 15]` 的循环整数序列；

偏置和反量化参数也是通过简单的确定性算法生成的。偏置是一个循环的小浮点数序列，而反量化尺度在这个例子中是一个固定的常量。

## 其他说明

在电脑上算的GPU和CPU结果是一致的，在手机上算的GPU结果和电脑上算的GPU/CPU结果是一致的，应该能说明是正确调用kernel并送入数据，只是手机上算的GPU和CPU结果不同，原因应该是char这个东西在电脑和手机上默认的分别是unsigned和signed char导致区别
