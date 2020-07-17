#pragma once

#include <opencv2/dnn.hpp>

#include <functional>

class UNet
{
public:
    static void applyClahe(cv::Mat& inputOutput);
    UNet(std::string const& modelFile, std::string const& weights, cv::Size inputSize, std::vector<float> thresholds, bool halfSize = false);

    static auto foundBoundingBoxes(std::vector<cv::Mat> const& masks) -> std::vector<std::vector<cv::Rect>>;
    auto performPrediction(cv::Mat const& frame,
                           std::function<void(std::vector<cv::Mat> const&)>&& cb = [](std::vector<cv::Mat> const&){},
                           bool isNeededToBeResizedToInputSize = true,
                           bool isNeededToBeSwappedRAndB = true) -> std::vector<cv::Mat>;

private:
    cv::Size           _inputSize;
    std::vector<float> _thresholds;
    cv::dnn::Net       _net;
    bool               _halfSize{};
};

