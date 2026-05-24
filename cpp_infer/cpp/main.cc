#include <onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>

#include <algorithm>
#include <array>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr int kImageSize = 28;
constexpr int kClassCount = 10;

struct Prediction {
    int prediction = -1;
    std::array<float, kClassCount> logits {};
};

std::vector<float> PreprocessImage(const cv::Mat& image, bool invert) {
    if (image.empty()) {
        throw std::runtime_error("empty image");
    }

    cv::Mat gray;
    if (image.channels() == 1) {
        gray = image;
    } else {
        cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
    }

    cv::Mat resized;
    cv::resize(gray, resized, cv::Size(kImageSize, kImageSize), 0, 0, cv::INTER_AREA);

    std::vector<float> input;
    input.reserve(kImageSize * kImageSize);

    for (int row = 0; row < resized.rows; ++row) {
        for (int col = 0; col < resized.cols; ++col) {
            const float value = static_cast<float>(resized.at<std::uint8_t>(row, col)) / 255.0f;
            input.push_back(invert ? 1.0f - value : value);
        }
    }

    return input;
}

cv::Mat ReadFrameFromCamera(int camera_id) {
    cv::VideoCapture camera(camera_id);
    if (!camera.isOpened()) {
        throw std::runtime_error("failed to open camera");
    }

    cv::Mat frame;
    camera >> frame;
    if (frame.empty()) {
        throw std::runtime_error("failed to read camera frame");
    }

    return frame;
}

class MnistOnnxInfer {
public:
    explicit MnistOnnxInfer(const char* model_path)
        : env_(ORT_LOGGING_LEVEL_WARNING, "mnist"),
          session_options_(),
          session_(nullptr) {
        session_options_.SetIntraOpNumThreads(1);
        session_options_.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
        session_ = Ort::Session(env_, model_path, session_options_);
    }

    Prediction Predict(std::vector<float>& input_values) {
        if (input_values.size() != kImageSize * kImageSize) {
            throw std::runtime_error("input must contain 784 float values");
        }

        std::array<int64_t, 4> input_shape {1, 1, kImageSize, kImageSize};
        Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(
            OrtArenaAllocator,
            OrtMemTypeDefault);

        Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
            memory_info,
            input_values.data(),
            input_values.size(),
            input_shape.data(),
            input_shape.size());

        const char* input_names[] = {"input"};
        const char* output_names[] = {"logits"};

        std::vector<Ort::Value> outputs = session_.Run(
            Ort::RunOptions {nullptr},
            input_names,
            &input_tensor,
            1,
            output_names,
            1);

        const float* logits = outputs[0].GetTensorData<float>();

        Prediction result;
        std::copy(logits, logits + kClassCount, result.logits.begin());
        result.prediction = static_cast<int>(
            std::max_element(result.logits.begin(), result.logits.end()) - result.logits.begin());
        return result;
    }

private:
    Ort::Env env_;
    Ort::SessionOptions session_options_;
    Ort::Session session_;
};

void PrintUsage(const char* program) {
    std::cout
        << "Usage:\n"
        << "  " << program << " --image <path> [--invert]\n"
        << "  " << program << " --camera [camera_id] [--invert]\n"
        << "  " << program << " --dummy\n\n"
        << "--invert is useful for black digits on a white background.\n";
}

bool HasFlag(int argc, char** argv, const std::string& flag) {
    for (int i = 1; i < argc; ++i) {
        if (argv[i] == flag) {
            return true;
        }
    }
    return false;
}

} // namespace

int main(int argc, char** argv) {
    try {
        if (argc == 1 || HasFlag(argc, argv, "--help")) {
            PrintUsage(argv[0]);
            return 0;
        }

        const bool invert = HasFlag(argc, argv, "--invert");
        const char* model_path = CPP_INFER_PROJECT_ROOT "/model.onnx";

        std::vector<float> input_values;

        if (std::string(argv[1]) == "--image") {
            if (argc < 3) {
                throw std::runtime_error("--image requires a path");
            }

            const cv::Mat image = cv::imread(argv[2], cv::IMREAD_COLOR);
            if (image.empty()) {
                throw std::runtime_error("failed to read image: " + std::string(argv[2]));
            }
            input_values = PreprocessImage(image, invert);
        } else if (std::string(argv[1]) == "--camera") {
            const int camera_id = argc >= 3 && argv[2][0] != '-' ? std::stoi(argv[2]) : 0;
            input_values = PreprocessImage(ReadFrameFromCamera(camera_id), invert);
        } else if (std::string(argv[1]) == "--dummy") {
            input_values.assign(kImageSize * kImageSize, 0.0f);
        } else {
            PrintUsage(argv[0]);
            return 1;
        }

        MnistOnnxInfer infer(model_path);
        const Prediction result = infer.Predict(input_values);

        std::cout << "prediction: " << result.prediction << '\n';
        std::cout << "logits:";
        for (const float logit : result.logits) {
            std::cout << ' ' << logit;
        }
        std::cout << '\n';

        return 0;
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        return 1;
    }
}
