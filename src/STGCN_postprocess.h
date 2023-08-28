#pragma once

#include <math.h>

#include <map>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <vector>

namespace PaddleDetection {
typedef std::map<int, int> Match;

struct FallInput {
    std::vector<std::vector<float>> xInputs;
    std::vector<std::vector<float>> yInputs;
    // 关键点的个数17
    int valsize = 17;
    // 输入的帧数 50
    int framesize = 50;
};

struct FallResult {
    // Rectangle coordinates of detected object: left, right, top, down

    std::vector<int> rect;
    std::vector<float> feat;
    int timestamp = 0;
    int id = -1;
    bool trackState = false;
};

}  // namespace PaddleDetection