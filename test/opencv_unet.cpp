#if 1
#include <opencv_unet/UNet.hpp>

#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include <iostream>
#include <vector>
#include <experimental/filesystem>
#include <boost/algorithm/string/split.hpp>
#include <boost/algorithm/string/classification.hpp>
#include <fstream>

#ifdef _MSC_VER
#include <filesystem>
namespace fs = std::filesystem;
#else
#include <experimental/filesystem>
namespace fs = std::experimental::filesystem;
#endif

#include "../src/TimeStamp.hpp"

namespace {
auto toColorMask(std::vector<cv::Mat> const& masks, std::vector<cv::Scalar> const& colors) -> cv::Mat
{
   cv::Mat coloredMasks(masks[0].rows, masks[0].cols, CV_8UC3, cv::Scalar(0, 0, 0));
   for (size_t i = 0; i < masks.size(); ++i)
   {
      coloredMasks.setTo(colors[i], masks[i]);
   }
   return coloredMasks;
}
} /// end namespace anonymous

auto ParseOptions(int argc, char *argv[]) -> std::map<std::string, std::vector<std::string>>
{
  std::map<std::string, std::vector<std::string>> params;
  for (auto i = 0; i < argc; ++i)
  {
    auto currentOption = std::string(argv[i]);
    auto delimiter = currentOption.find_first_of('=');
    if (delimiter != std::string::npos)
    {
      auto input = currentOption.substr(delimiter + 1);
      boost::split(params[currentOption.substr(0, delimiter)], input, boost::is_any_of(","));
    }
  }
  return params;
}

auto treshold1 = 0;
auto treshold2 = 650;
auto houghTreshold = 170;
auto houghMinLineLength = 300;
auto houghMaxLineGap = 100;
auto houghTresholdMax = 1000;
auto houghMinLineLengthMax = 500;
auto houghMaxLineGapMax = 500;
auto claheClipLimit = 0;
auto claheClipLimitMax = 100;
cv::Mat resultImg;
cv::Mat inputImg;
cv::Mat showImg;
std::vector<cv::Vec4i> linesSelected;
static void on_trackbar( int, void* )
{
  auto clahe = cv::createCLAHE(claheClipLimit, cv::Size(20,20));
  cv::Mat clacheResult;
  clahe->apply(inputImg, clacheResult);
  cv::Mat detected_edges;
  showImg = resultImg.clone();
  cv::blur(clacheResult, detected_edges, cv::Size(25,25));
  cv::Canny(detected_edges, detected_edges, treshold1, treshold2, 5, true);
  std::vector<cv::Vec4i> linesP;
  cv::HoughLinesP(detected_edges, linesP, 1, CV_PI/180, houghTreshold, houghMinLineLength, houghMaxLineGap);
  std::sort(linesP.begin(), linesP.end(), [](auto& a,auto& b) {
    return std::fabs(a[0] + a[2])/2.0f < std::fabs(b[0] + b[2])/2.0f;
  });
  for (auto const& line : linesP) {
    cv::line(showImg, cv::Point(line[0], line[1]), cv::Point(line[2], line[3]), cv::Scalar{0xFF, 0, 0xFF}, 4);
    //break;
  }
  if (!linesP.empty())
  {
    auto const& line = linesP.back();
    cv::line(showImg, cv::Point(line[0], line[1]), cv::Point(line[2], line[3]), cv::Scalar{0x00, 0xFF, 0x00}, 4);
  }
  //alpha = (double) alpha_slider/alpha_slider_max ;
  //beta = ( 1.0 - alpha );
  //addWeighted( src1, alpha, src2, beta, 0.0, dst);
  cv::imshow("trackbar", showImg);
  cv::imshow("Detected edges", detected_edges);
}

void runOpts(std::map<std::string, std::vector<std::string>> params)
{
  if (params["--video-output-path"].size() != 1) {
    std::cout << "--video-output-path was not set!" << std::endl;
    return;
  }
  if (params["--unet-darknet-model"].size() != 2) {
    std::cout << "--unet-darknet-model was not set!" << std::endl;
    return;
  }
  if (params["--input"].empty()) {
    std::cout << "--input was not set!" << std::endl;
    return;
  }
  if (params["--output"].size() != 1) {
    std::cout << "--output was not set!" << std::endl;
    return;
  }
  if (params["--roi"].size() != 4) {
    std::cout << "--roi was not set!" << std::endl;
    return;
  }
  auto roi = cv::Rect{std::stoi(params["--roi"][0]),
                      std::stoi(params["--roi"][1]),
                      std::stoi(params["--roi"][2]),
                      std::stoi(params["--roi"][3])};

  if (params["--downscale"].size() != 1) {
    std::cout << "--downscal was not set!" << std::endl;
    return;
  }

  auto downscaledSize = cv::Size{roi.width / std::stoi(params["--downscale"][0]),
                       roi.height / std::stoi(params["--downscale"][0])};

  static const std::string kWinName = "OpenCV UNet Demo";
  cv::namedWindow(kWinName, cv::WINDOW_NORMAL);

  auto unet = UNet{params["--unet-darknet-model"][0], params["--unet-darknet-model"][1], downscaledSize, {0.3f, 0.3f, 0.3f, 0.5f, 0.5f, 0.5f, 0.5f}, false};
  std::vector<std::string> skippedFiles;
  if (std::experimental::filesystem::is_directory(params["--input"][0]))
  {
    cv::VideoWriter video(params["--video-output-path"][0],
                          cv::VideoWriter::fourcc('M','J','P','G'),
                          0.5, cv::Size(roi.width, roi.height));
    auto files = std::vector<std::experimental::filesystem::directory_entry>{std::experimental::filesystem::directory_iterator(params["--input"][0]),
                                                                             std::experimental::filesystem::directory_iterator()};
    std::sort(files.begin(), files.end(), [](auto& a, auto& b) { return a.path().string() > b.path().string(); });
    for (auto file : files)
    {
      auto filePath = file.path().string();
      auto fileName = file.path().filename().string();
      if (fileName.substr(fileName.size() - 3) == "jpg")
      {
        fileName.replace(fileName.size() - 3, 3, "png");
      }
      cv::Mat frame = cv::imread(filePath);
      cv::Mat mask = cv::imread(params["--input"][1] + "/" + fileName);
      //cv::rotate(frame, frame, cv::ROTATE_90_CLOCKWISE);
      if (mask.empty() || frame.empty() || (cv::waitKey(1) == 27))
      {
        continue;
      }
      //if ((frame.cols != 1072) && (frame.cols != 1080) && (frame.cols != 1440) && (frame.cols))
      if ((frame.cols < (roi.width + roi.x)) || (frame.rows < (roi.height + roi.y)))
      {
        skippedFiles.push_back(file.path().string());
        continue;
      }
      //frame = frame(roi);
      cv::Mat gray;
      cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
      UNet::applyClahe(gray);
      auto masks = unet.performPrediction(gray, [](std::vector<cv::Mat> const&){}, true, false);
      auto boundingBoxes = UNet::foundBoundingBoxes(masks);
#if 0
      for (auto index = 0; index < masks.size(); ++index)
      {
//        std::vector<std::vector<cv::Point>> contours;
//        std::vector<cv::Vec4i> hierarchy;
//        cv::findContours(masks[index], contours, hierarchy, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE);
//        std::vector<cv::Rect> rects(contours.size());
//        for (auto j = 0; j < contours.size(); ++j)
//        {
//          rects[j] = cv::boundingRect(contours[j]);
//        }
        auto biggestIt = std::max_element(boundingBoxes[index].cbegin(), boundingBoxes[index].cend(), [](auto const& a, auto const& b) {
          return (a.width < b.width) && (a.height < b.height);
        });
        std::cout << "6" << std::endl;
        if (biggestIt != boundingBoxes[index].cend())
        {
          std::cout << "7" << std::endl;
          auto cropRect = *biggestIt;
          auto fract = cropRect.width % 32;
          if (fract == 0)
          {
            std::cout << "8" << std::endl;
            cropRect.width += 32;
            if (cropRect.x >= 16)
            {
              cropRect.x -= 16;
            }
            else
            {
              cropRect.width += (16 - cropRect.x);
              cropRect.x = 0;
            }
          }
          else
          {
            std::cout << "9" << std::endl;
            auto fractCompensation = ((32 - fract) > 16) ? ((32 - fract) + 32) : (32 - fract);
            cropRect.width += fractCompensation;
            auto fractCompensationHalf = fractCompensation / 2;
            if (cropRect.x >= fractCompensationHalf)
            {
              cropRect.x -= fractCompensationHalf;
            }
            else
            {
              cropRect.width += (fractCompensationHalf - cropRect.x);
              cropRect.x = 0;
            }
          }
          fs::create_directories(params["--output"][0] + "/images/");
          fs::create_directories(params["--output"][0] + "/masks/");
          //std::filesystem::create_directories(params["--output"][0] + "/edge/");
          //cv::imwrite(params["--output"][0] + "/images/" + fileName, frame(cropRect));
        }
      }
#endif
#if 0
      resultImg = frame;
      inputImg = gray;
      namedWindow("trackbar", cv::WINDOW_AUTOSIZE);
      cv::createTrackbar( "treshold1", "trackbar", &treshold1, treshold2, on_trackbar );
      cv::createTrackbar( "treshold2", "trackbar", &treshold2, treshold2, on_trackbar );
      cv::createTrackbar( "houghTreshold", "trackbar", &houghTreshold, houghTresholdMax, on_trackbar );
      cv::createTrackbar( "houghMinLineLength", "trackbar", &houghMinLineLength, houghMinLineLengthMax, on_trackbar );
      cv::createTrackbar( "houghMaxLineGap", "trackbar", &houghMaxLineGap, houghMaxLineGapMax, on_trackbar );
      cv::createTrackbar( "claheClipLimit", "trackbar", &claheClipLimit, claheClipLimitMax, on_trackbar );
      on_trackbar( treshold1, 0 );
      cv::waitKey(0);
#endif
#if 0
      //cv::line(frame, cv::Point(700, 0), cv::Point(700, 1280), cv::Scalar{0xFF, 0xFF, 0xFF}, 5);
      cv::Mat detected_edges = gray(cv::Rect(0, 0, 700, 1280)).clone();
      //cv::blur(detected_edges, detected_edges, cv::Size(3,3));
      cv::Canny(detected_edges, detected_edges, 300, 625, 5, true);
      std::vector<cv::Vec4i> linesP;
      cv::HoughLinesP(detected_edges, linesP, 1, CV_PI/180, 170, 300, 100);
      std::sort(linesP.begin(), linesP.end(), [](auto& a,auto& b) {
        return std::fabs(a[0] + a[2])/2.0f < std::fabs(b[0] + b[2])/2.0f;
      });
      cv::Vec4f line1;
      cv::Vec4f line2;
      for (auto const& line : linesP) {
        //y1 = kx1 + b; y2 = kx2 + b; b = y1 - kx1; b = y2 - kx2; (y1 - y2)/(x1 - x2) = k;
        //x = (y - b)/k
        auto k = static_cast<float>(line[1] - line[3])/static_cast<float>(line[0] - line[2]);
        auto b = static_cast<float>(line[1]) - k*line[0];
        line1[0] = (static_cast<float>(line[1]) - b)/k;
        line1[1] = 0;
        line1[2] = (static_cast<float>(line[3]) - b)/k;
        line1[3] = 1280;
        //cv::line(frame, cv::Point(line[0], line[1]), cv::Point(line[2], line[3]), cv::Scalar{0xFF, 0, 0xFF}, 4);
        cv::line(frame, cv::Point(line1[0], line1[1]), cv::Point(line1[2], line1[3]), cv::Scalar{0xFF, 0, 0xFF}, 4);
        break;
      }
      auto const& line = linesP.back();
      //y1 = kx1 + b; y2 = kx2 + b; b = y1 - kx1; b = y2 - kx2; (y1 - y2)/(x1 - x2) = k;
      //x = (y - b)/k
      auto k = static_cast<float>(line[1] - line[3])/static_cast<float>(line[0] - line[2]);
      auto b = static_cast<float>(line[1]) - k*line[0];
      line2[0] = (static_cast<float>(line[1]) - b)/k;
      line2[1] = 0;
      line2[2] = (static_cast<float>(line[3]) - b)/k;
      line2[3] = 1280;
      //cv::line(frame, cv::Point(line[0], line[1]), cv::Point(line[2], line[3]), cv::Scalar{0xFF, 0, 0xFF}, 4);
      cv::line(frame, cv::Point(line2[0], line2[1]), cv::Point(line2[2], line2[3]), cv::Scalar{0xFF, 0, 0xFF}, 4);
      cv::imwrite(params["--output"][0] + "/edge/" + fileName, detected_edges);
#endif
#if 0
      auto fileAnnotation = std::ofstream(std::string("data/18.09.2020_Cable/18.09.2020_Data") + "/" + fileName.substr(0, fileName.size() - 3) + "json");
      fileAnnotation << "{\n"
              "  \"version\": \"4.5.6\",\n"
              "  \"flags\": {},\n"
              "  \"shapes\": [\n"
              "    {\n"
              "      \"label\": \"cable\",\n"
              "      \"points\": [\n"
              "        [\n"
              << line1[0] << ",\n"
              << line1[1] << "\n"
              "        ],\n"
              "        [\n"
              << line1[2] << ",\n"
              << line1[3] << "\n"
              "        ],\n"
              "        [\n"
              << line2[2] << ",\n"
              << line2[3] << "\n"
              "        ],\n"
              "        [\n"
              << line2[0] << ",\n"
              << line2[1] << "\n"
              "        ]\n"
              "      ],\n"
              "      \"group_id\": null,\n"
              "      \"shape_type\": \"polygon\",\n"
              "      \"flags\": {}\n"
              "    }\n"
              "  ],\n"
              "  \"imagePath\": \"../18.09.2020_Photo/" << file.path().filename().string() << "\",\n"
              "  \"imageData\": null,\n"
              "  \"imageHeight\": 1280,\n"
              "  \"imageWidth\": 1024\n"
              "}";
#endif
#if 0
      auto clahe = cv::createCLAHE();
      cv::Mat clacheResult;
      clahe->apply(gray, clacheResult);
      cv::cvtColor(clacheResult, clacheResult, cv::COLOR_GRAY2BGR);
      cv::imwrite(params["--output"][0] + "/" + fileName, clacheResult);
#endif
      //cv::imwrite(params["--output"][0] + "/masks/" + fileName, mask(cropRect));
      //cv::rectangle(frame, *biggestIt, cv::Scalar{0xFF, 0xFF, 0x00}, 2);
      cv::addWeighted(frame, 1.0, toColorMask(masks, std::vector<cv::Scalar>{cv::Scalar(0x00, 0xFF, 00)/*, cv::Scalar(0x00, 0xFF, 0x00), cv::Scalar(0x00, 0x00, 0xFF)*/}), 0.5, 0.0, frame);
      cv::addWeighted(frame, 1.0, mask, 0.5, 0.0, frame);
      cv::imwrite(params["--output"][0] + fileName, frame);
      //cv::resize(frame, frame, cv::Size(, 1408));
      //video.write(frame);
      imshow(kWinName, frame);
    }
    for(auto const& skippedFile :skippedFiles)
    {
      std::cout << skippedFile << std::endl;
    }
    video.release();
    cv::destroyAllWindows();
    return;
  }

  cv::VideoCapture cap;
  cap.open(params["--input"][0]);

  cv::VideoWriter video(params["--video-output-path"][0],
                        cv::VideoWriter::fourcc('M','J','P','G'),
                        10,
                        cv::Size(static_cast<int32_t>(cap.get(cv::CAP_PROP_FRAME_WIDTH)),
                                 static_cast<int32_t>(cap.get(cv::CAP_PROP_FRAME_HEIGHT))));

  cv::Mat frame;
  while (cv::waitKey(1) < 0)
  {
    cap >> frame;
    if (frame.empty() || (cv::waitKey(1) == 27))
    {
      break;
    }
    TAKEN_TIME();
    auto masks = unet.performPrediction(frame);
    cv::addWeighted(frame, 1.0, toColorMask(masks, std::vector<cv::Scalar>{cv::Scalar(0xFF, 00, 00),
                                                                           cv::Scalar(0x00, 0xFF, 0x00),
                                                                           cv::Scalar(0x00, 0x00, 0xFF),
                                                                           cv::Scalar(0xFF, 00, 00),
                                                                           cv::Scalar(0xFF, 00, 00),
                                                                           cv::Scalar(0xFF, 00, 00),
                                                                           cv::Scalar(0xFF, 00, 00)}), 0.5, 0.0, frame);
    video.write(frame);
    imshow(kWinName, frame);
  }
  video.release();
  cv::destroyAllWindows();
}

auto main(int argc, char** argv) -> int32_t
{
  auto opts = ParseOptions(argc, argv);
  runOpts(opts);
  return 0;
}
#else
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include <iostream>
#include <opencv2/imgproc.hpp>

using namespace cv;
using std::cout;

#if 1
auto treshold1 = 0;
auto treshold2 = 650;
cv::Mat resultImg;
cv::Mat inputImg;
static void on_trackbar( int, void* )
{
  cv::Mat detected_edges;
  cv::blur(inputImg, detected_edges, cv::Size(3,3));
  cv::Canny(detected_edges, detected_edges, treshold1, treshold2, 5, true);
  std::vector<cv::Vec4i> linesP;
  cv::HoughLinesP(detected_edges, linesP, 1, CV_PI/180, 170, 300, 100);
  std::sort(linesP.begin(), linesP.end(), [](auto& a,auto& b) {
    return std::fabs(a[0] + a[2])/2.0f < std::fabs(b[0] + b[2])/2.0f;
  });
  for (auto const& line : linesP) {
    cv::line(resultImg, cv::Point(line[0], line[1]), cv::Point(line[2], line[3]), cv::Scalar{0xFF, 0, 0xFF}, 4);
    break;
  }
  auto const& line = linesP.back();
  cv::line(resultImg, cv::Point(line[0], line[1]), cv::Point(line[2], line[3]), cv::Scalar{0xFF, 0, 0xFF}, 4);

  //alpha = (double) alpha_slider/alpha_slider_max ;
  //beta = ( 1.0 - alpha );
  //addWeighted( src1, alpha, src2, beta, 0.0, dst);
  cv::imshow( "Linear Blend", resultImg);
  cv::imshow( "Detected edges", detected_edges);
}
#else
const int alpha_slider_max = 100;
int alpha_slider;
double alpha;
double beta;
Mat src1;
Mat src2;
Mat dst;
static void on_trackbar( int, void* )
{
  alpha = (double) alpha_slider/alpha_slider_max ;
  beta = ( 1.0 - alpha );
  addWeighted( src1, alpha, src2, beta, 0.0, dst);
  imshow( "Linear Blend", dst );
}
#endif
int main( void )
{
#if 0
  src1 = imread("data/Kolosov_18.09.2020/18.09.2020_Photo/18.09.2020_2.jpg");
  src2 = imread( "data/Kolosov_18.09.2020/18.09.2020_Photo/18.09.2020_11.jpg");
  alpha_slider = 0;
  namedWindow("Linear Blend", WINDOW_AUTOSIZE); // Create Window
  char TrackbarName[50];
  sprintf( TrackbarName, "Alpha x %d", alpha_slider_max );
  createTrackbar( TrackbarName, "Linear Blend", &alpha_slider, alpha_slider_max, on_trackbar );
  on_trackbar( alpha_slider, 0 );
#else
#endif
  waitKey(0);
  return 0;
}
#endif