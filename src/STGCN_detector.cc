//   Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
#include <sstream>
// for setprecision
#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>

#include "STGCN_detector.h"
#include "STGCN_postprocess.h"
#include "keypoint_postprocess.h"
#include "spdlog/cfg/env.h"
#include "spdlog/spdlog.h"

using namespace paddle_infer;

namespace PaddleDetection {
#define OPTION_30
// Load Model and create model predictor
void FallDetector::LoadModel(const std::string& model_dir, const int batch_size,
                             const std::string& run_mode) {
    paddle_infer::Config config;
    std::string prog_file = model_dir + OS_PATH_SEP + "model.pdmodel";
    std::string params_file = model_dir + OS_PATH_SEP + "model.pdiparams";

    config.SetModel(prog_file, params_file);
    if (this->device_ == "GPU") {
        config.EnableUseGpu(100, this->gpu_id_);
        config.SwitchIrOptim(true);
        // use tensorrt
        if (run_mode != "paddle") {
            auto precision = paddle_infer::Config::Precision::kFloat32;
            if (run_mode == "trt_fp32") {
                precision = paddle_infer::Config::Precision::kFloat32;
            } else if (run_mode == "trt_fp16") {
                precision = paddle_infer::Config::Precision::kHalf;
            } else if (run_mode == "trt_int8") {
                precision = paddle_infer::Config::Precision::kInt8;
            } else {
                printf(
                    "run_mode should be 'paddle', 'trt_fp32', 'trt_fp16' or "
                    "'trt_int8'");
            }
            // set tensorrt
            config.EnableTensorRtEngine(1 << 30, batch_size,
                                        this->min_subgraph_size_, precision,
                                        false, this->trt_calib_mode_);

            // set use dynamic shape
            if (this->use_dynamic_shape_) {
                // set DynamicShsape for image tensor
                const std::vector<int> min_input_shape = {
                    1, 3, this->trt_min_shape_, this->trt_min_shape_};
                const std::vector<int> max_input_shape = {
                    1, 3, this->trt_max_shape_, this->trt_max_shape_};
                const std::vector<int> opt_input_shape = {
                    1, 3, this->trt_opt_shape_, this->trt_opt_shape_};
                const std::map<std::string, std::vector<int>>
                    map_min_input_shape = {{"image", min_input_shape}};
                const std::map<std::string, std::vector<int>>
                    map_max_input_shape = {{"image", max_input_shape}};
                const std::map<std::string, std::vector<int>>
                    map_opt_input_shape = {{"image", opt_input_shape}};

                config.SetTRTDynamicShapeInfo(map_min_input_shape,
                                              map_max_input_shape,
                                              map_opt_input_shape);
                std::cout << "TensorRT dynamic shape enabled" << std::endl;
            }
        }

    } else if (this->device_ == "XPU") {
        config.EnableXpu(10 * 1024 * 1024);
    } else {
        config.DisableGpu();
        if (this->use_mkldnn_) {
            config.EnableMKLDNN();
            // cache 10 different shapes for mkldnn to avoid memory leak
            config.SetMkldnnCacheCapacity(10);
        }
        config.SetCpuMathLibraryNumThreads(this->cpu_math_library_num_threads_);
    }
    config.SwitchUseFeedFetchOps(false);
    config.SwitchIrOptim(true);
    config.DisableGlogInfo();
    // Memory optimization
    config.EnableMemoryOptim();
    // std::string info = config.Summary();
    // fprintf(stdout, "%s\n", info.c_str());
    predictor_ = std::move(CreatePredictor(config));
}

/**
 * 根据给定的结果和输入数据，对关键点数据进行细化处理，将关键点数据缩放至模型输入的尺寸。
 *
 * @param results 表示关键点结果的 KeyPointResult 对象的向量
 * @param inputs 表示输入数据的 FallInput 对象
 *
 * @throws None
 */
void RefineKeypointdata(std::vector<PaddleDetection::KeyPointResult>& results,
                        PaddleDetection::FallInput& inputs) {
#ifdef OPTION_30
    int value = 30;  // 当 OPTION_30 定义时，变量值为 30
#else
    int value = 50;  // 当 OPTION_30 未定义时，变量值为 50
#endif
    int targetW = 384;
    int targetH = 512;
    if (results.size() != value)
        return;
    else if (inputs.xInputs.size() == 0 && inputs.yInputs.size() == 0) {
        for (int i = 0; i < value; i++) {
            std::vector<float> xKeypoint;
            std::vector<float> yKeypoint;
            int objectX = results[i].rect[0];
            int objectY = results[i].rect[1];
            int objectW = results[i].rect[2] - objectX;
            int objectH = results[i].rect[3] - objectY;
            for (size_t j = 0; j < inputs.valsize; j++) {
                xKeypoint.push_back(
                    ((results[i].keypoints[1 + 3 * j] - objectX) / objectW) *
                    targetW);
                yKeypoint.push_back(
                    ((results[i].keypoints[2 + 3 * j] - objectY) / objectH) *
                    targetH);
            }
            inputs.xInputs.push_back(xKeypoint);
            inputs.yInputs.push_back(yKeypoint);
            xKeypoint.clear();
            yKeypoint.clear();
        }
    } else {
        std::vector<float> xKeypoint;
        std::vector<float> yKeypoint;
        int objectX = results[results.size() - 1].rect[0];
        int objectY = results[results.size() - 1].rect[1];
        int objectW = results[results.size() - 1].rect[2] - objectX;
        int objectH = results[results.size() - 1].rect[3] - objectY;
        for (size_t j = 0; j < inputs.valsize; j++) {
            xKeypoint.push_back(
                ((results[results.size() - 1].keypoints[1 + 3 * j] - objectX) /
                 objectW) *
                targetW);
            yKeypoint.push_back(
                ((results[results.size() - 1].keypoints[2 + 3 * j] - objectY) /
                 objectH) *
                targetH);
        }
        for (size_t i = 0; i < 1; i++) {
            inputs.xInputs.erase(inputs.xInputs.begin());
            inputs.yInputs.erase(inputs.yInputs.begin());
            inputs.xInputs.push_back(xKeypoint);
            inputs.yInputs.push_back(yKeypoint);
        }
        xKeypoint.clear();
        yKeypoint.clear();
    }
}
void FallDetector::Preprocess(const cv::Mat& ori_im) {
    // Clone the image : keep the original mat for postprocess
    cv::Mat im = ori_im.clone();
    cv::cvtColor(im, im, cv::COLOR_BGR2RGB);
    preprocessor_.Run(&im, &inputs_);
}
void FallDetector::Preprocess(std::vector<KeyPointResult>& results,
                              ImageBlob* data, FallInput& inputs) {
    RefineKeypointdata(results, inputs);
#ifdef OPTION_30
    int value = 30;  // 当 OPTION_30 定义时，变量值为 30
#else
    int value = 50;  // 当 OPTION_30 未定义时，变量值为 50
#endif
    (data->im_data_).resize(2 * value * 17 * 1);

    float* base = (data->im_data_).data();

    // float* base = data.data();

    for (int j = 0; j < value; ++j) {
        for (int i = 0; i < 17; ++i) {
            base[j * 17 + i] = inputs.xInputs[j][i];  // 存储X的数据
            base[j * 17 + i + value * 17] =
                inputs.yInputs[j][i];  // 存储Y的数据
        }
    }
}

void FallDetector::Predict(std::vector<KeyPointResult>& results,
                           FallInput& inputs) {
    // if (frame_ID % 2 == 0) {
    std::cout << "\033[2J\033[1;1H";
    // }
    // in_data_batch
    std::vector<float> in_data_all;
    std::vector<float> im_shape_all(1 * 2);
    std::vector<float> scale_factor_all(1 * 2);
    // Preprocess image
    Preprocess(results, &inputs_, inputs);

    in_data_all.insert(in_data_all.end(), inputs_.im_data_.begin(),
                       inputs_.im_data_.end());

    // Prepare input tensor
#ifdef OPTION_30
    int value = 30;  // 当 OPTION_30 定义时，变量值为 30
#else
    int value = 50;  // 当 OPTION_30 未定义时，变量值为 50
#endif
    auto input_names = predictor_->GetInputNames();
    for (const auto& tensor_name : input_names) {
        auto in_tensor = predictor_->GetInputHandle(tensor_name);
        if (tensor_name == "data_batch_0") {
            in_tensor->Reshape({1, 2, value, 17, 1});
            in_tensor->CopyFromCpu(in_data_all.data());
        }
    }
    std::vector<int> output_shape, idx_shape;
    // Run predictor

    predictor_->Run();
    // Get output tensor
    auto output_names = predictor_->GetOutputNames();
    auto out_tensor = predictor_->GetOutputHandle(output_names[0]);
    output_shape = out_tensor->shape();

    // Calculate output length
    int output_size = 1;
    for (int j = 0; j < output_shape.size(); ++j) {
        output_size *= output_shape[j];
    }

    output_data_.resize(output_size);
    out_tensor->CopyToCpu(output_data_.data());
    int maxID = -1;
    float maxval = -1;
    for (size_t i = 0; i < output_size; i = i + 1) {
        if (output_data_[i] > maxval) {
            maxval = output_data_[i];
            maxID = i;
        }

        SPDLOG_INFO("output data: {}", output_data_[i]);
    }
    SPDLOG_INFO("maxID: {}", maxID);
    results[results.size() - 1].ID = maxID;
}

}  // namespace PaddleDetection
