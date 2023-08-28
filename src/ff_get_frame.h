#ifndef FF_GET_FRAME_H_
#define FF_GET_FRAME_H_

#include <poll.h>
#include <unistd.h>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/videoio.hpp>

#include "common.h"
#include "ffcxx/codec_context.h"
#include "ffcxx/format_context.h"
#include "format.h"
#include "myqueue.h"
#include "spdlog/cfg/env.h"
#include "spdlog/spdlog.h"
#include "websocket_client.h"

// 并行
#include <condition_variable>
#include <thread>
// 互斥访问
#include <atomic>
#include <mutex>

// static std::optional<char> readKey() noexcept;
int getFrame(std::string url, std::shared_ptr<AysncQueue<MyVariant>> queue,
             int index);

#endif