set +x
set -e

work_path=${PWD}

# 1. check paddle_inference exists
if [ ! -d "${work_path}/paddle_inference" ]; then
  echo work_path no exist paddle_inference: ${work_path}
  exit 1
fi

# 2. check CMakeLists exists
if [ ! -f "${work_path}/CMakeLists.txt" ]; then
  cp -a "${work_path}/../../lib/CMakeLists.txt" "${work_path}/"
fi


# 是否使用GPU(即是否使用 CUDA)
WITH_GPU=ON

# 是否使用MKL or openblas，TX2需要设置为OFF
WITH_MKL=ON

# 是否集成 TensorRT(仅WITH_GPU=ON 有效)
WITH_TENSORRT=ON

# paddle 预测库lib名称，由于不同平台不同版本预测库lib名称不同，请查看所下载的预测库中`paddle_inference/lib/`文件夹下`lib`的名称
PADDLE_LIB_NAME=libpaddle_inference

# TensorRT 的include路径
TENSORRT_INC_DIR=/home/nfs/workspace/paddle/PaddleDetection/deploy/cpp/paddle_inference/TensorRT-8.4.0.6/include

# TensorRT 的lib路径
TENSORRT_LIB_DIR=/home/nfs/workspace/paddle/PaddleDetection/deploy/cpp/paddle_inference/TensorRT-8.4.0.6/targets/x86_64-linux-gnu/lib

# Paddle 预测库路径
PADDLE_DIR=${work_path}/paddle_inference

# CUDA 的 lib 路径
CUDA_LIB=/usr/local/cuda/lib64

# CUDNN 的 lib 路径
CUDNN_LIB=/usr/local/cuda-11.7/targets/x86_64-linux/lib


# 是否开启关键点模型预测功能
WITH_KEYPOINT=OFF

# 是否开启跟踪模型预测功能
WITH_MOT=OFF

MACHINE_TYPE=`uname -m`
echo "MACHINE_TYPE: "${MACHINE_TYPE}


# if [ "$MACHINE_TYPE" = "x86_64" ]
# then
#   echo "set OPENCV_DIR for x86_64"
#   # linux系统通过以下命令下载预编译的opencv
#   mkdir -p $(pwd)/deps && cd $(pwd)/deps
#   wget -c https://paddledet.bj.bcebos.com/data/opencv-3.4.16_gcc8.2_ffmpeg.tar.gz
#   tar -xvf opencv-3.4.16_gcc8.2_ffmpeg.tar.gz && cd ..

#   # set OPENCV_DIR
#   OPENCV_DIR=$(pwd)/deps/opencv-3.4.16_gcc8.2_ffmpeg

# elif [ "$MACHINE_TYPE" = "aarch64" ]
# then
#   echo "set OPENCV_DIR for aarch64"
#   # TX2平台通过以下命令下载预编译的opencv
#   mkdir -p $(pwd)/deps && cd $(pwd)/deps
#   wget -c https://bj.bcebos.com/v1/paddledet/data/TX2_JetPack4.3_opencv_3.4.6_gcc7.5.0.tar.gz
#   tar -xvf TX2_JetPack4.3_opencv_3.4.6_gcc7.5.0.tar.gz && cd ..

#   # set OPENCV_DIR
#   OPENCV_DIR=$(pwd)/deps/TX2_JetPack4.3_opencv_3.4.6_gcc7.5.0/

# else
#   echo "Please set OPENCV_DIR manually"
# fi

# echo "OPENCV_DIR: "$OPENCV_DIR

# 以下无需改动
# rm -rf build
# mkdir -p build
cd build
rm CMakeCache.txt
cmake .. \
    -GNinja \
    -DWITH_GPU=${WITH_GPU} \
    -DWITH_MKL=${WITH_MKL} \
    -DWITH_TENSORRT=${WITH_TENSORRT} \
    -DTENSORRT_LIB_DIR=${TENSORRT_LIB_DIR} \
    -DTENSORRT_INC_DIR=${TENSORRT_INC_DIR} \
    -DPADDLE_DIR=${PADDLE_DIR} \
    -DWITH_STATIC_LIB=${WITH_STATIC_LIB} \
    -DCUDA_LIB=${CUDA_LIB} \
    -DCUDNN_LIB=${CUDNN_LIB} \
    -DPADDLE_LIB_NAME=${PADDLE_LIB_NAME} \
    -DWITH_KEYPOINT=${WITH_KEYPOINT} \
    -DWITH_MOT=${WITH_MOT} \
    # -DCMAKE_BUILD_TYPE=Debug 

# make -j12
ninja -j12
echo "make finished!"
