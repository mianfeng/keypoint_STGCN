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

#include <glog/logging.h>
#include <math.h>
#include <sys/stat.h>
#include <sys/types.h>

#include <algorithm>
#include <chrono>
#include <fstream>
#include <iostream>
#include <memory>
#include <numeric>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/videoio.hpp>
#include <queue>
#include <string>
#include <vector>

#include "STGCN_detector.h"
#include "myqueue.h"
#include "spdlog/cfg/env.h"
#include "spdlog/spdlog.h"
#ifdef _WIN32
#include <direct.h>
#include <io.h>
#elif LINUX
#include <stdarg.h>
#endif

#include <gflags/gflags.h>

#include "keypoint_detector.h"
#include "object_detector.h"
#include "preprocess_op.h"
#define OPTION_30

DEFINE_string(model_dir, "", "Path of object detector inference model");
DEFINE_string(model_dir_keypoint, "",
              "Path of keypoint detector inference model");
DEFINE_string(model_dir_STGCN, "", "Path of keypoint detector inference model");
DEFINE_string(image_file, "", "Path of input image");
DEFINE_string(image_dir, "",
              "Dir of input image, `image_file` has a higher priority.");
DEFINE_int32(batch_size, 1, "batch_size of object detector");
DEFINE_int32(batch_size_keypoint, 1, "batch_size of keypoint detector");
DEFINE_string(
    video_file, "",
    "Path of input video, `video_file` or `camera_id` has a highest priority.");
DEFINE_int32(camera_id, -1, "Device id of camera to predict");
DEFINE_bool(
    use_gpu, false,
    "Deprecated, please use `--device` to set the device you want to run.");
DEFINE_string(device, "CPU",
              "Choose the device you want to run, it can be: CPU/GPU/XPU, "
              "default is CPU.");
DEFINE_double(threshold, 0.5, "Threshold of score.");
DEFINE_double(threshold_keypoint, 0.5, "Threshold of score.");
DEFINE_string(output_dir, "output", "Directory of output visualization files.");
DEFINE_string(run_mode, "paddle",
              "Mode of running(paddle/trt_fp32/trt_fp16/trt_int8)");
DEFINE_int32(gpu_id, 0, "Device id of GPU to execute");
DEFINE_bool(run_benchmark, false,
            "Whether to predict a image_file repeatedly for benchmark");
DEFINE_bool(use_mkldnn, false, "Whether use mkldnn with CPU");
DEFINE_int32(cpu_threads, 1, "Num of threads with CPU");
DEFINE_int32(trt_min_shape, 1, "Min shape of TRT DynamicShapeI");
DEFINE_int32(trt_max_shape, 1280, "Max shape of TRT DynamicShapeI");
DEFINE_int32(trt_opt_shape, 640, "Opt shape of TRT DynamicShapeI");
DEFINE_bool(trt_calib_mode, false,
            "If the model is produced by TRT offline quantitative calibration, "
            "trt_calib_mode need to set True");
DEFINE_bool(use_dark, true, "Whether use dark decode in keypoint postprocess");

void captureVideo(const std::string& video_path,
                  std::shared_ptr<AysncQueue<cv::Mat>> queue, int threadID) {
    cv::Mat frame;
    cv::VideoCapture capture;
    SPDLOG_INFO("  thread {} captureVideo thread start", threadID);
    capture.open(video_path.c_str(), cv::CAP_FFMPEG);
    while (1) {
        capture.read(frame);
        if (!frame.empty()) {
            queue->enqueue(frame);
            // SPDLOG_INFO("captureVideo thread start");
        }
    }
}

void PredictVideo(const std::string& video_path,
                  PaddleDetection::ObjectDetector* det,
                  PaddleDetection::KeyPointDetector* keypoint,
                  PaddleDetection::FallDetector* fallDet,
                  const std::string& output_dir = "output") {
    // Open video
    cv::VideoCapture capture;
    std::string video_out_name = "output.mp4";
    if (FLAGS_camera_id != -1) {
        capture.open(FLAGS_camera_id);
    } else {
        capture.open(video_path.c_str());
        video_out_name =
            video_path.substr(video_path.find_last_of(OS_PATH_SEP) + 1);
    }
    if (!capture.isOpened()) {
        printf("can not open video : %s\n", video_path.c_str());
        return;
    }

    // Get Video info : resolution, fps, frame count
    int video_width = static_cast<int>(capture.get(cv::CAP_PROP_FRAME_WIDTH));
    int video_height = static_cast<int>(capture.get(cv::CAP_PROP_FRAME_HEIGHT));
    int video_fps = static_cast<int>(capture.get(cv::CAP_PROP_FPS));
    int video_frame_count =
        static_cast<int>(capture.get(cv::CAP_PROP_FRAME_COUNT));
    printf("fps: %d, frame_count: %d*****\n", video_fps, video_frame_count);

    // Create VideoWriter for output
    cv::VideoWriter video_out;
    std::string video_out_path(output_dir);
    if (output_dir.rfind(OS_PATH_SEP) != output_dir.size() - 1) {
        video_out_path += OS_PATH_SEP;
    }
    video_out_path += "output.avi";
    video_out.open(video_out_path.c_str(),
                   cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), video_fps,
                   cv::Size(video_width, video_height), true);
    if (!video_out.isOpened()) {
        printf("create video writer failed!\n");
        return;
    }
    PaddleDetection::PoseSmooth smoother =
        PaddleDetection::PoseSmooth(video_width, video_height);

    std::vector<PaddleDetection::ObjectResult> result;

    std::vector<int> bbox_num;
    std::vector<double> det_times;
    auto labels = det->GetLabelList();
    auto colormap = PaddleDetection::GenerateColorMap(labels.size());

    // Store keypoint results
    std::vector<PaddleDetection::KeyPointResult> result_kpts;
    PaddleDetection::FallInput inputs;

    std::vector<cv::Mat> imgs_kpts;
    std::vector<std::vector<float>> center_bs;
    std::vector<std::vector<float>> scale_bs;
    std::vector<int> colormap_kpts = PaddleDetection::GenerateColorMap(20);
#if defined(rtsp)

    std::shared_ptr<AysncQueue<cv::Mat>> qu(new AysncQueue<cv::Mat>);
    std::thread capImg(captureVideo, video_path, qu, 1);
#endif

    // Capture all frames and do inference
    cv::Mat frame;
    int frame_id = 1;
    bool is_rbox = false;
    while (1) {
#if defined(rtsp)
        frame = qu->dequeue();

#else
        capture.read(frame);
#endif
        if (frame.empty()) {
            break;
        }
        std::vector<cv::Mat> imgs;
        imgs.push_back(frame);
        printf("detect frame: %d\n", frame_id);
        det->Predict(imgs, FLAGS_threshold, 0, 1, &result, &bbox_num,
                     &det_times);
        std::vector<PaddleDetection::ObjectResult> out_result;
        for (const auto& item : result) {
            if (item.confidence < FLAGS_threshold || item.class_id == -1) {
                continue;
            }
            out_result.push_back(item);
            if (item.rect.size() > 6) {
                is_rbox = true;
                printf(
                    "class=%d confidence=%.4f rect=[%d %d %d %d %d %d %d %d]\n",
                    item.class_id, item.confidence, item.rect[0], item.rect[1],
                    item.rect[2], item.rect[3], item.rect[4], item.rect[5],
                    item.rect[6], item.rect[7]);
            } else {
                printf("class=%d confidence=%.4f rect=[%d %d %d %d]\n",
                       item.class_id, item.confidence, item.rect[0],
                       item.rect[1], item.rect[2], item.rect[3]);
            }
        }

        if (keypoint && out_result.size() > 0) {
            // result_kpts.clear();
            int imsize = out_result.size();
            for (int i = 0; i < imsize; i++) {
                auto item = out_result[i];
                cv::Mat crop_img;
                std::vector<double> keypoint_times;
                std::vector<int> rect = {item.rect[0], item.rect[1],
                                         item.rect[2], item.rect[3]};
                std::vector<float> center;
                std::vector<float> scale;
                if (item.class_id == 0) {
                    PaddleDetection::CropImg(frame, crop_img, rect, center,
                                             scale);
                    // emplace_back允许我们通过传递构造函数的参数来构造新元素，
                    // 而不是先创建一个临时对象再将其复制或移动到容器中。
                    center_bs.emplace_back(center);
                    scale_bs.emplace_back(scale);
                    imgs_kpts.emplace_back(crop_img);
                }

                if (imgs_kpts.size() == FLAGS_batch_size_keypoint ||
                    ((i == imsize - 1) && !imgs_kpts.empty())) {
                    keypoint->Predict(imgs_kpts, center_bs, scale_bs, rect,
                                      FLAGS_threshold, 0, 1, &result_kpts,
                                      &keypoint_times);
                    imgs_kpts.clear();
                    center_bs.clear();
                    scale_bs.clear();
                }
            }
#ifdef OPTION_30
            int value = 30;  // 当 OPTION_30 定义时，变量值为 30
#else
            int value = 50;  // 当 OPTION_30 未定义时，变量值为 50
#endif
            SPDLOG_INFO("result_kpts size: {}", result_kpts.size());
            if (result_kpts.size() > value) {
                result_kpts.erase(
                    result_kpts.begin(),
                    result_kpts.begin() + (result_kpts.size() - value));
                // result_kpts.erase(result_kpts.begin());
                fallDet->Predict(result_kpts, inputs);
            }

            // if (result_kpts.size() == 1) {
            //     for (int i = 0; i < result_kpts.size(); i++) {
            //         result_kpts[i] =
            //         smoother.smooth_process(&(result_kpts[i]));
            //     }
            // }

            cv::Mat out_im = VisualizeKptsResult(frame, result_kpts,
                                                 colormap_kpts, frame_id);
            video_out.write(out_im);
        } else {
            // Visualization result
            cv::Mat out_im = PaddleDetection::VisualizeResult(
                frame, out_result, labels, colormap, is_rbox);
            video_out.write(out_im);
        }

        frame_id += 1;
    }
#if defined(rtsp)

    capImg.join();

#endif  // rtsp

    capture.release();
    video_out.release();
}

static void initLogger() noexcept {
    spdlog::cfg::load_env_levels();
#ifdef NDEBUG
    spdlog::set_pattern("%^[%L] %v [%Y-%m-%d %H:%M:%S.%e]%$");
#else
    spdlog::set_pattern("%^[%L] %v [%Y-%m-%d %H:%M:%S.%e] [%@]%$");
#endif
}
int main(int argc, char** argv) {
    // Parsing command-line
    initLogger();
    google::ParseCommandLineFlags(&argc, &argv, true);
    if (FLAGS_model_dir.empty() ||
        (FLAGS_image_file.empty() && FLAGS_image_dir.empty() &&
         FLAGS_video_file.empty())) {
        std::cout << "Usage: ./main --model_dir=/PATH/TO/INFERENCE_MODEL/ "
                     "(--model_dir_keypoint=/PATH/TO/INFERENCE_MODEL/)"
                  << "--image_file=/PATH/TO/INPUT/IMAGE/" << std::endl;
        return -1;
    }
    if (!(FLAGS_run_mode == "paddle" || FLAGS_run_mode == "trt_fp32" ||
          FLAGS_run_mode == "trt_fp16" || FLAGS_run_mode == "trt_int8")) {
        std::cout << "run_mode should be 'paddle', 'trt_fp32', 'trt_fp16' or "
                     "'trt_int8'.";
        return -1;
    }
    transform(FLAGS_device.begin(), FLAGS_device.end(), FLAGS_device.begin(),
              ::toupper);
    if (!(FLAGS_device == "CPU" || FLAGS_device == "GPU" ||
          FLAGS_device == "XPU")) {
        std::cout << "device should be 'CPU', 'GPU' or 'XPU'.";
        return -1;
    }
    if (FLAGS_use_gpu) {
        std::cout
            << "Deprecated, please use `--device` to set the device you want "
               "to run.";
        return -1;
    }
    // Load model and create a object detector
    PaddleDetection::ObjectDetector det(
        FLAGS_model_dir, FLAGS_device, FLAGS_use_mkldnn, FLAGS_cpu_threads,
        FLAGS_run_mode, FLAGS_batch_size, FLAGS_gpu_id, FLAGS_trt_min_shape,
        FLAGS_trt_max_shape, FLAGS_trt_opt_shape, FLAGS_trt_calib_mode);

    PaddleDetection::KeyPointDetector* keypoint = nullptr;
    if (!FLAGS_model_dir_keypoint.empty()) {
        keypoint = new PaddleDetection::KeyPointDetector(
            FLAGS_model_dir_keypoint, FLAGS_device, FLAGS_use_mkldnn,
            FLAGS_cpu_threads, FLAGS_run_mode, FLAGS_batch_size_keypoint,
            FLAGS_gpu_id, FLAGS_trt_min_shape, FLAGS_trt_max_shape,
            FLAGS_trt_opt_shape, FLAGS_trt_calib_mode, FLAGS_use_dark);
    }
    PaddleDetection::FallDetector* fallDet = nullptr;
    if (!FLAGS_model_dir_STGCN.empty()) {
        fallDet = new PaddleDetection::FallDetector(
            FLAGS_model_dir_STGCN, FLAGS_device, FLAGS_use_mkldnn,
            FLAGS_cpu_threads, FLAGS_run_mode, FLAGS_batch_size_keypoint,
            FLAGS_gpu_id, FLAGS_trt_min_shape, FLAGS_trt_max_shape,
            FLAGS_trt_opt_shape, FLAGS_trt_calib_mode, FLAGS_use_dark);
    }
    // Do inference on input video or image
    if (!FLAGS_video_file.empty() || FLAGS_camera_id != -1) {
        PredictVideo(FLAGS_video_file, &det, keypoint, fallDet,
                     FLAGS_output_dir);
    } else if (!FLAGS_image_file.empty() || !FLAGS_image_dir.empty()) {
        std::vector<std::string> all_img_paths;
        std::vector<cv::String> cv_all_img_paths;
        if (!FLAGS_image_file.empty()) {
            all_img_paths.push_back(FLAGS_image_file);
            if (FLAGS_batch_size > 1) {
                std::cout << "batch_size should be 1, when set `image_file`."
                          << std::endl;
                return -1;
            }
        } else {
            cv::glob(FLAGS_image_dir, cv_all_img_paths);
            for (const auto& img_path : cv_all_img_paths) {
                all_img_paths.push_back(img_path);
            }
        }
    }
    delete keypoint;
    keypoint = nullptr;
    return 0;
}
