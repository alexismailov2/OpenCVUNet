#include <opencv_unet/UNet.hpp>

#include "TimeStamp.hpp"

#include <opencv2/imgproc.hpp>

#include <iostream>
#include <vector>

namespace {
auto toClassesMapsThreshold(cv::Mat const& score,
                            cv::Size const& inputSize,
                            std::vector<float> threshold) -> std::vector<cv::Mat>
{
  TAKEN_TIME();
  auto const rows = score.size[2];
  auto const cols = score.size[3];
  auto const channels = score.size[1];

  std::vector<cv::Mat> classesMaps(static_cast<size_t>(channels));
  for (auto ch = 0; ch < channels; ++ch)
  {
    cv::Mat channelScore = cv::Mat(rows, cols, CV_32FC1, const_cast<float *>(score.ptr<float>(0, ch, 0)));
    cv::inRange(channelScore, threshold[ch], 1000000.0, classesMaps[ch]);
    if ((inputSize.width != 0) && (inputSize.height != 0))
    {
       cv::resize(classesMaps[ch], classesMaps[ch], inputSize);
    }
  }
  return classesMaps;
}
} /// end namespace anonymous

UNet::UNet(std::string const& modelFile,
           std::string const& weights,
           cv::Size inputSize,
           std::vector<float> thresholds,
           bool halfSize)
    : _inputSize{inputSize}
    , _thresholds{thresholds}
    , _net{weights.empty() ? cv::dnn::readNet(modelFile) : cv::dnn::readNetFromDarknet(modelFile, weights)}
    , _halfSize{halfSize}
{
  _net.setPreferableBackend(::cv::dnn::DNN_BACKEND_CUDA);
  _net.setPreferableTarget(::cv::dnn::DNN_TARGET_CUDA);
}

auto UNet::performPrediction(cv::Mat const &frame,
                             std::function<void(std::vector<cv::Mat> const&)>&& cb,
                             bool isNeededToBeResizedToInputSize,
                             bool isNeededToBeSwappedRAndB) -> std::vector<cv::Mat>
{
  auto const nearHalfHeightDiv32 = ((frame.size().height / 2) & (~0x1F));
  auto const nearFullWidthDiv32 = (frame.size().width & (~0x1F));
  if (_halfSize)
  {
     TAKEN_TIME();
     auto croppedAndResizedFrame = frame({0, frame.size().height - nearHalfHeightDiv32, frame.size().width, nearHalfHeightDiv32});
     _net.setInput(cv::dnn::blobFromImage(
        croppedAndResizedFrame, 1.0 / 255.0, {nearFullWidthDiv32 / 4, nearHalfHeightDiv32}, {0, 0, 0, 0}, isNeededToBeSwappedRAndB, false));
  }
  else
  {
     TAKEN_TIME();
     _net.setInput(cv::dnn::blobFromImage(frame, 1.0 / 255.0, _inputSize, {0, 0, 0, 0}, isNeededToBeSwappedRAndB, false));
  }
  cv::Mat predict;
  {
    TAKEN_TIME();
    predict = _net.forward();
  }
  auto classesMaps = toClassesMapsThreshold(predict, isNeededToBeResizedToInputSize ? frame.size() : cv::Size{}, _thresholds);
  cb(classesMaps);
  if (_halfSize)
  {
     TAKEN_TIME();
     for (auto& item : classesMaps)
     {
        cv::Mat resized = cv::Mat::zeros(frame.rows, frame.cols, CV_8UC1);
        cv::Mat roi = resized({0, frame.size().height - nearHalfHeightDiv32, frame.size().width, nearHalfHeightDiv32});
        cv::resize(item, item, cv::Size{nearFullWidthDiv32, nearHalfHeightDiv32}, 0, 0, cv::INTER_NEAREST);
        item.copyTo(roi);
        item = resized;
     }
  }
  return classesMaps;
}

auto UNet::foundBoundingBoxes(std::vector<cv::Mat> const& masks) -> std::vector<std::vector<cv::Rect>>
{
  std::vector<std::vector<cv::Point>> contours;
  std::vector<cv::Vec4i> hierarchy;
  std::vector<std::vector<cv::Rect>> boundingBoxes;
  boundingBoxes.reserve(masks.size());
  for (auto const& mask : masks)
  {
    cv::findContours(mask, contours, hierarchy, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE);
    boundingBoxes.emplace_back(std::vector<cv::Rect>(contours.size()));
    for (auto j = 0; j < contours.size(); ++j)
    {
      boundingBoxes.back()[j] = cv::boundingRect(contours[j]);
    }
  }
  return boundingBoxes;
}

void UNet::applyClahe(cv::Mat& inputOutput)
{
  TAKEN_TIME();
  auto clahe = cv::createCLAHE();
  clahe->apply(inputOutput, inputOutput);
}
