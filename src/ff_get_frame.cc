#include "ff_get_frame.h"

using namespace ffcxx;

static std::optional<char> readKey() noexcept {
    pollfd pfd{STDIN_FILENO, POLLIN, 0};
    auto ret = poll(&pfd, 1, 0);
    if (ret > 0) {
        return getchar();
    } else {
        return std::nullopt;
    }
}

int getFrame(std::string url, std::shared_ptr<AysncQueue<MyVariant>> queue,
             int index) {
    spdlog::cfg::load_env_levels();

    SPDLOG_INFO("start decode example. input: {}", url);

    bool exit = false;

    auto intputResult = FormatContext::openInput(url, [&]() {
        if (readKey() == 'q') exit = true;
        return exit;
    });
    if (!intputResult) {
        SPDLOG_ERROR("openInput failed, {}", intputResult.error());
        return 1;
    }

    auto &input = intputResult.value();

    auto result = input->findStreamInfo();
    if (!result) {
        SPDLOG_ERROR("{}", result.error());
        return 1;
    }
    input->dumpFormat(-1, url, false);

    auto mediaInfo = input->mediaInfo();

    std::shared_ptr<CodecContext> videoDecoder;
    if (mediaInfo.videoStream().has_value()) {
        auto decoderResult =
            CodecContext::open(mediaInfo.videoStream().value());
        if (!decoderResult) {
            SPDLOG_ERROR("{}", decoderResult.error());
            return 1;
        }
        videoDecoder = decoderResult.value();
    }

    while (!exit) {
        if (readKey() == 'q') break;

        auto packetResult = input->read();
        if (!packetResult) {
            SPDLOG_ERROR("read failed, {}", packetResult.error());
            break;
        }
        auto &packet = packetResult.value();
        if (packet.streamIndex() != mediaInfo.videoIndex()) continue;

        result = videoDecoder->send(packet);
        if (!result && result.error().code() != AVERROR(EAGAIN)) {
            SPDLOG_ERROR("{}", result.error());
            break;
        }

        while (!exit) {
            auto frameResult = videoDecoder->receive();
            if (!frameResult) {
                if (frameResult.error().code() != AVERROR(EAGAIN))
                    SPDLOG_ERROR("{}", result.error());
                break;
            } else {
                auto &frame = frameResult.value();
                // SPDLOG_INFO("frame width {}, height {}, pts {}",
                // frame.width(),
                //             frame.height(), frame.pts());

                int height = frame.height();
                int width = frame.lineSize(0);
                cv::Mat tmpImg =
                    cv::Mat::zeros(height * 3 / 2, frame.lineSize(0), CV_8UC1);
                memcpy(tmpImg.data, frame.data(0), width * height);
                memcpy(tmpImg.data + width * height, frame.data(1),
                       width * height / 4);
                memcpy(tmpImg.data + width * height * 5 / 4, frame.data(2),
                       width * height / 4);
                cv::Mat rgb;
                cv::cvtColor(tmpImg, rgb, cv::COLOR_YUV2RGB_I420);
                MatWrapper matWrapper{MatSource::SOURCE_1, rgb};
                // 传入detect端的图像
                if (!rgb.empty()) {
                    switch (index) {
                        case 1:
                            matWrapper = {MatSource::SOURCE_1, rgb};
                            break;
                        case 2:
                            matWrapper = {MatSource::SOURCE_2, rgb};
                            break;
                        case 3:
                            matWrapper = {MatSource::SOURCE_3, rgb};
                            break;
                        default:
                            spdlog::info("Mat index error!");
                            break;
                    }
                    MyVariant myVariant = matWrapper;
                    queue->enqueue(myVariant);
                }
            }
        }
    }

    return 0;
}