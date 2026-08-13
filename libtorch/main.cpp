#include <torch/torch.h>
#include <torch/script.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;

// ============================================================
// CONFIGURATION
// ============================================================

constexpr int IMAGE_SIZE = 28;
constexpr int INPUT_SIZE = 784;
constexpr int HIDDEN_1 = 128;
constexpr int HIDDEN_2 = 64;
constexpr int NUM_CLASSES = 10;

constexpr int EPOCHS = 5;
constexpr int BATCH_SIZE = 64;

constexpr int NUM_BENCHMARK_IMAGES = 10;
constexpr int WARMUP_RUNS = 10;
constexpr int BENCHMARK_RUNS = 100;


// ============================================================
// MNIST DATASET
// ============================================================

struct MNISTData {
    torch::Tensor images;
    torch::Tensor labels;
};


// ------------------------------------------------------------
// Read big-endian 32-bit integer
// ------------------------------------------------------------

uint32_t readBigEndianUInt32(std::ifstream& file) {

    unsigned char bytes[4];

    file.read(
        reinterpret_cast<char*>(bytes),
        4
    );

    return
        (static_cast<uint32_t>(bytes[0]) << 24)
        |
        (static_cast<uint32_t>(bytes[1]) << 16)
        |
        (static_cast<uint32_t>(bytes[2]) << 8)
        |
        static_cast<uint32_t>(bytes[3]);
}


// ------------------------------------------------------------
// Load MNIST images
// ------------------------------------------------------------

torch::Tensor loadMNISTImages(
    const std::string& filename
) {

    std::ifstream file(
        filename,
        std::ios::binary
    );

    if (!file) {
        throw std::runtime_error(
            "Could not open MNIST image file:\n"
            + filename
        );
    }

    uint32_t magic = readBigEndianUInt32(file);
    uint32_t count = readBigEndianUInt32(file);
    uint32_t rows = readBigEndianUInt32(file);
    uint32_t cols = readBigEndianUInt32(file);

    if (magic != 2051) {
        throw std::runtime_error(
            "Invalid MNIST image file: "
            + filename
        );
    }

    std::cout
        << "Loading "
        << filename
        << "\n";

    std::vector<unsigned char> buffer(
        static_cast<size_t>(count)
        * rows
        * cols
    );

    file.read(
        reinterpret_cast<char*>(buffer.data()),
        static_cast<std::streamsize>(buffer.size())
    );

    if (!file) {
        throw std::runtime_error(
            "Failed to read MNIST image data."
        );
    }

    torch::Tensor tensor =
        torch::from_blob(
            buffer.data(),
            {
                static_cast<int64_t>(count),
                static_cast<int64_t>(rows * cols)
            },
            torch::TensorOptions()
                .dtype(torch::kUInt8)
        ).clone();

    // Convert uint8 [0,255] -> float32 [0,1]
    tensor =
        tensor.to(torch::kFloat32)
              .div(255.0);

    return tensor;
}


// ------------------------------------------------------------
// Load MNIST labels
// ------------------------------------------------------------

torch::Tensor loadMNISTLabels(
    const std::string& filename
) {

    std::ifstream file(
        filename,
        std::ios::binary
    );

    if (!file) {
        throw std::runtime_error(
            "Could not open MNIST label file:\n"
            + filename
        );
    }

    uint32_t magic = readBigEndianUInt32(file);
    uint32_t count = readBigEndianUInt32(file);

    if (magic != 2049) {
        throw std::runtime_error(
            "Invalid MNIST label file: "
            + filename
        );
    }

    std::vector<unsigned char> buffer(
        count
    );

    file.read(
        reinterpret_cast<char*>(buffer.data()),
        static_cast<std::streamsize>(buffer.size())
    );

    if (!file) {
        throw std::runtime_error(
            "Failed to read MNIST label data."
        );
    }

    torch::Tensor tensor =
        torch::from_blob(
            buffer.data(),
            {
                static_cast<int64_t>(count)
            },
            torch::TensorOptions()
                .dtype(torch::kUInt8)
        ).clone();

    return tensor.to(torch::kLong);
}


// ============================================================
// FIND PROJECT ROOT
// ============================================================

fs::path findProjectRoot() {

    fs::path current =
        fs::current_path();

    while (true) {

        fs::path datasetDir =
            current / "dataset";

        fs::path trainImages =
            datasetDir / "train-images-idx3-ubyte";

        fs::path testImages =
            datasetDir / "test-images-idx3-ubyte";

        if (
            fs::exists(trainImages)
            &&
            fs::exists(testImages)
        ) {
            return current;
        }

        if (
            current == current.root_path()
        ) {
            break;
        }

        current =
            current.parent_path();
    }

    throw std::runtime_error(
        "Could not find project root containing dataset/."
    );
}


// ============================================================
// MODEL
// ============================================================

struct MNISTModelImpl
    : torch::nn::Module {

    torch::nn::Linear fc1{nullptr};
    torch::nn::Linear fc2{nullptr};
    torch::nn::Linear fc3{nullptr};

    MNISTModelImpl() {

        fc1 =
            register_module(
                "fc1",
                torch::nn::Linear(
                    INPUT_SIZE,
                    HIDDEN_1
                )
            );

        fc2 =
            register_module(
                "fc2",
                torch::nn::Linear(
                    HIDDEN_1,
                    HIDDEN_2
                )
            );

        fc3 =
            register_module(
                "fc3",
                torch::nn::Linear(
                    HIDDEN_2,
                    NUM_CLASSES
                )
            );
    }


    torch::Tensor forward(
        torch::Tensor x
    ) {

        x =
            torch::relu(
                fc1->forward(x)
            );

        x =
            torch::relu(
                fc2->forward(x)
            );

        x =
            fc3->forward(x);

        return x;
    }
};

TORCH_MODULE(MNISTModel);


// ============================================================
// TEST ACCURACY
// ============================================================

double evaluateAccuracy(
    MNISTModel& model,
    const torch::Tensor& images,
    const torch::Tensor& labels
) {

    torch::NoGradGuard no_grad;

    model->eval();

    int64_t correct = 0;
    int64_t total = labels.size(0);

    const int64_t datasetSize =
        images.size(0);

    for (
        int64_t start = 0;
        start < datasetSize;
        start += BATCH_SIZE
    ) {

        int64_t end =
            std::min(
                start + static_cast<int64_t>(BATCH_SIZE),
                datasetSize
            );

        int64_t batchSize =
            end - start;

        auto batchImages =
            images.narrow(
                0,
                start,
                batchSize
            );

        auto batchLabels =
            labels.narrow(
                0,
                start,
                batchSize
            );

        auto output =
            model->forward(
                batchImages
            );

        auto predictions =
            output.argmax(
                1
            );

        correct +=
            predictions
                .eq(batchLabels)
                .sum()
                .item<int64_t>();
    }

    return
        100.0
        *
        static_cast<double>(correct)
        /
        static_cast<double>(total);
}


// ============================================================
// SAVE TORCHSCRIPT MODEL
// ============================================================

void saveTorchScriptModel(
    MNISTModel& model,
    const fs::path& outputPath
) {

    model->eval();

    torch::Tensor exampleInput =
        torch::randn(
            {
                1,
                INPUT_SIZE
            }
        );

    std::vector<torch::jit::IValue> inputs;

    inputs.push_back(
        exampleInput
    );

    auto traced =
        torch::jit::trace(
            model,
            inputs
        );

    traced.save(
        outputPath.string()
    );
}


// ============================================================
// BENCHMARK STATISTICS
// ============================================================

double calculateAverage(
    const std::vector<double>& values
) {

    return
        std::accumulate(
            values.begin(),
            values.end(),
            0.0
        )
        /
        static_cast<double>(
            values.size()
        );
}


double calculateMedian(
    std::vector<double> values
) {

    std::sort(
        values.begin(),
        values.end()
    );

    size_t middle =
        values.size() / 2;

    if (
        values.size() % 2 == 0
    ) {

        return
            (
                values[middle - 1]
                +
                values[middle]
            )
            /
            2.0;
    }

    return values[middle];
}


double calculateStdDev(
    const std::vector<double>& values,
    double mean
) {

    double sum = 0.0;

    for (
        double value : values
    ) {

        double difference =
            value - mean;

        sum +=
            difference
            *
            difference;
    }

    return std::sqrt(
        sum
        /
        static_cast<double>(
            values.size()
        )
    );
}


// ============================================================
// MAIN
// ============================================================

int main() {

    try {

        // ====================================================
        // PROJECT PATHS
        // ====================================================

        fs::path projectRoot =
            findProjectRoot();

        fs::path datasetDir =
            projectRoot / "dataset";

        fs::path trainImagesPath =
            datasetDir
            /
            "train-images-idx3-ubyte";

        fs::path trainLabelsPath =
            datasetDir
            /
            "train-labels-idx1-ubyte";

        fs::path testImagesPath =
            datasetDir
            /
            "test-images-idx3-ubyte";

        fs::path testLabelsPath =
            datasetDir
            /
            "test-labels-idx1-ubyte";

        fs::path benchmarkPath =
            datasetDir
            /
            "benchmark_images"
            /
            "benchmark_images.npy";

        fs::path modelDir =
            projectRoot
            /
            "libtorch"
            /
            "model";

        fs::path modelPath =
            modelDir
            /
            "mnist_libtorch.pt";


        // ====================================================
        // HEADER
        // ====================================================

        std::cout
            << "\n============================================================\n";

        std::cout
            << "LIBTORCH MNIST TRAINING + BENCHMARK\n";

        std::cout
            << "============================================================\n\n";

        std::cout
            << "LibTorch version : "
            << TORCH_VERSION
            << "\n";

        std::cout
            << "Backend          : LibTorch / PyTorch\n";

        std::cout
            << "Device           : CPU\n";

        std::cout
            << "Model            : "
            << "784 -> 128 -> ReLU -> 64 -> ReLU -> 10\n";

        std::cout
            << "Epochs           : "
            << EPOCHS
            << "\n";

        std::cout
            << "Batch size       : "
            << BATCH_SIZE
            << "\n";

        std::cout
            << "Benchmark images : "
            << NUM_BENCHMARK_IMAGES
            << "\n";

        std::cout
            << "Warm-up runs     : "
            << WARMUP_RUNS
            << "\n";

        std::cout
            << "Benchmark runs   : "
            << BENCHMARK_RUNS
            << "\n";


        // ====================================================
        // LOAD DATASET
        // ====================================================

        std::cout
            << "\nLoading MNIST IDX dataset...\n";


        auto trainImages =
            loadMNISTImages(
                trainImagesPath.string()
            );

        auto trainLabels =
            loadMNISTLabels(
                trainLabelsPath.string()
            );

        auto testImages =
            loadMNISTImages(
                testImagesPath.string()
            );

        auto testLabels =
            loadMNISTLabels(
                testLabelsPath.string()
            );


        std::cout
            << "\nTraining images : "
            << trainImages.size(0)
            << "\n";

        std::cout
            << "Test images     : "
            << testImages.size(0)
            << "\n";


        // ====================================================
        // CREATE MODEL
        // ====================================================

        MNISTModel model;


        // ====================================================
        // OPTIMIZER
        // ====================================================

        torch::optim::Adam optimizer(
            model->parameters(),
            torch::optim::AdamOptions(
                0.001
            )
        );


        // ====================================================
        // TRAINING
        // ====================================================

        std::cout
            << "\n============================================================\n";

        std::cout
            << "TRAINING\n";

        std::cout
            << "============================================================\n";


        auto trainingStart =
            std::chrono::high_resolution_clock::now();


        const int64_t trainSize =
            trainImages.size(0);


        for (
            int epoch = 0;
            epoch < EPOCHS;
            ++epoch
        ) {

            model->train();

            // Shuffle indices
            auto permutation =
                torch::randperm(
                    trainSize,
                    torch::kLong
                );


            double epochLoss = 0.0;

            int64_t batches = 0;


            for (
                int64_t start = 0;
                start < trainSize;
                start += BATCH_SIZE
            ) {

                int64_t end =
                    std::min(
                        start
                        +
                        static_cast<int64_t>(
                            BATCH_SIZE
                        ),
                        trainSize
                    );

                int64_t currentBatchSize =
                    end - start;


                auto indices =
                    permutation.narrow(
                        0,
                        start,
                        currentBatchSize
                    );


                auto batchImages =
                    trainImages.index_select(
                        0,
                        indices
                    );


                auto batchLabels =
                    trainLabels.index_select(
                        0,
                        indices
                    );


                optimizer.zero_grad();


                auto output =
                    model->forward(
                        batchImages
                    );


                auto loss =
                    torch::nn::functional::cross_entropy(
                        output,
                        batchLabels
                    );


                loss.backward();

                optimizer.step();


                epochLoss +=
                    loss.item<double>();

                ++batches;
            }


            double averageLoss =
                epochLoss
                /
                static_cast<double>(
                    batches
                );


            std::cout
                << "Epoch "
                << (epoch + 1)
                << "/"
                << EPOCHS
                << " | Loss: "
                << std::fixed
                << std::setprecision(4)
                << averageLoss
                << "\n";
        }


        auto trainingEnd =
            std::chrono::high_resolution_clock::now();


        double trainingTime =
            std::chrono::duration<double>(
                trainingEnd
                -
                trainingStart
            ).count();


        // ====================================================
        // TEST ACCURACY
        // ====================================================

        std::cout
            << "\nEvaluating model...\n";


        double testAccuracy =
            evaluateAccuracy(
                model,
                testImages,
                testLabels
            );


        std::cout
            << "\n============================================================\n";

        std::cout
            << "TRAINING COMPLETE\n";

        std::cout
            << "============================================================\n";

        std::cout
            << std::fixed
            << std::setprecision(2);

        std::cout
            << "Training time : "
            << trainingTime
            << " seconds\n";

        std::cout
            << "Test accuracy : "
            << testAccuracy
            << "%\n";


        // ====================================================
        // SAVE MODEL
        // ====================================================

        fs::create_directories(
            modelDir
        );


        saveTorchScriptModel(
            model,
            modelPath
        );


        std::cout
            << "\nModel saved to:\n"
            << modelPath
            << "\n";


        // ====================================================
        // LOAD BENCHMARK IMAGES
        // ====================================================

        std::cout
            << "\nLoading benchmark images...\n";


        // The benchmark_images.npy file contains
        // 10 x 28 x 28 uint8 MNIST images.
        //
        // To avoid adding another external C++ dependency,
        // the same raw benchmark images are loaded directly
        // using a small NPY reader.


        std::ifstream npyFile(
            benchmarkPath,
            std::ios::binary
        );


        if (!npyFile) {

            throw std::runtime_error(
                "Could not open benchmark_images.npy:\n"
                +
                benchmarkPath.string()
            );
        }


        char magic[6];

        npyFile.read(
            magic,
            6
        );


        if (
            magic[0] != '\x93'
            ||
            magic[1] != 'N'
            ||
            magic[2] != 'U'
            ||
            magic[3] != 'M'
            ||
            magic[4] != 'P'
            ||
            magic[5] != 'Y'
        ) {

            throw std::runtime_error(
                "Invalid NPY benchmark file."
            );
        }


        unsigned char major;
        unsigned char minor;

        npyFile.read(
            reinterpret_cast<char*>(&major),
            1
        );

        npyFile.read(
            reinterpret_cast<char*>(&minor),
            1
        );


        uint32_t headerLength;


        if (major == 1) {

            uint16_t length16;

            npyFile.read(
                reinterpret_cast<char*>(&length16),
                sizeof(length16)
            );

            headerLength =
                length16;

        } else {

            npyFile.read(
                reinterpret_cast<char*>(&headerLength),
                sizeof(headerLength)
            );
        }


        std::string header(
            headerLength,
            '\0'
        );


        npyFile.read(
            header.data(),
            headerLength
        );


        bool isUint8 =
            header.find(
                "'|u1'"
            )
            !=
            std::string::npos;


        if (!isUint8) {

            throw std::runtime_error(
                "Expected uint8 benchmark NPY file."
            );
        }


        std::vector<unsigned char>
            benchmarkRaw(
                NUM_BENCHMARK_IMAGES
                *
                IMAGE_SIZE
                *
                IMAGE_SIZE
            );


        npyFile.read(
            reinterpret_cast<char*>(
                benchmarkRaw.data()
            ),
            static_cast<std::streamsize>(
                benchmarkRaw.size()
            )
        );


        torch::Tensor benchmarkInput =
            torch::from_blob(
                benchmarkRaw.data(),
                {
                    NUM_BENCHMARK_IMAGES,
                    INPUT_SIZE
                },
                torch::TensorOptions()
                    .dtype(torch::kUInt8)
            )
            .clone()
            .to(torch::kFloat32)
            .div(255.0);


        std::cout
            << "Benchmark input shape: "
            << benchmarkInput.sizes()
            << "\n";


        // ====================================================
        // BENCHMARK WARM-UP
        // ====================================================

        model->eval();


        std::cout
            << "\nRunning warm-up...\n";


        {
            torch::NoGradGuard no_grad;


            for (
                int i = 0;
                i < WARMUP_RUNS;
                ++i
            ) {

                auto output =
                    model->forward(
                        benchmarkInput
                    );


                // Force output access
                auto prediction =
                    output.argmax(
                        1
                    );


                (void)prediction;
            }
        }


        std::cout
            << "Warm-up complete.\n";


        // ====================================================
        // BENCHMARK
        // ====================================================

        std::cout
            << "Running benchmark...\n";


        std::vector<double> times;

        times.reserve(
            BENCHMARK_RUNS
        );


        torch::Tensor finalOutput;


        {
            torch::NoGradGuard no_grad;


            for (
                int run = 0;
                run < BENCHMARK_RUNS;
                ++run
            ) {

                auto start =
                    std::chrono::high_resolution_clock::now();


                finalOutput =
                    model->forward(
                        benchmarkInput
                    );


                // Force computation
                auto prediction =
                    finalOutput.argmax(
                        1
                    );


                (void)prediction;


                auto end =
                    std::chrono::high_resolution_clock::now();


                double elapsed =
                    std::chrono::duration<double, std::milli>(
                        end - start
                    ).count();


                times.push_back(
                    elapsed
                );
            }
        }


        // ====================================================
        // PREDICTIONS
        // ====================================================

        auto predictions =
            finalOutput
                .argmax(1)
                .to(torch::kCPU);


        const int benchmarkLabels[
            NUM_BENCHMARK_IMAGES
        ] = {
            7, 2, 1, 0, 4,
            1, 4, 9, 5, 9
        };


        int correct =
            0;


        std::cout
            << "\n============================================================\n";

        std::cout
            << "PREDICTIONS\n";

        std::cout
            << "============================================================\n";


        for (
            int i = 0;
            i < NUM_BENCHMARK_IMAGES;
            ++i
        ) {

            int64_t predicted =
                predictions[i]
                    .item<int64_t>();


            int actual =
                benchmarkLabels[i];


            bool isCorrect =
                predicted == actual;


            if (isCorrect) {
                ++correct;
            }


            std::cout
                << "image_"
                << std::setw(2)
                << std::setfill('0')
                << (i + 1)
                << ".png"
                << " | True: "
                << actual
                << " | Predicted: "
                << predicted
                << " | "
                << (
                    isCorrect
                        ? "CORRECT"
                        : "WRONG"
                )
                << "\n";
        }


        // ====================================================
        // STATISTICS
        // ====================================================

        double average =
            calculateAverage(
                times
            );


        double median =
            calculateMedian(
                times
            );


        double minimum =
            *std::min_element(
                times.begin(),
                times.end()
            );


        double maximum =
            *std::max_element(
                times.begin(),
                times.end()
            );


        double stdDev =
            calculateStdDev(
                times,
                average
            );


        double timePerImage =
            average
            /
            NUM_BENCHMARK_IMAGES;


        double benchmarkAccuracy =
            100.0
            *
            static_cast<double>(correct)
            /
            NUM_BENCHMARK_IMAGES;


        // ====================================================
        // FINAL RESULTS
        // ====================================================

        std::cout
            << "\n============================================================\n";

        std::cout
            << "BENCHMARK RESULTS\n";

        std::cout
            << "============================================================\n";


        std::cout
            << std::fixed
            << std::setprecision(4);


        std::cout
            << "Number of images       : "
            << NUM_BENCHMARK_IMAGES
            << "\n";


        std::cout
            << "Warm-up runs           : "
            << WARMUP_RUNS
            << "\n";


        std::cout
            << "Benchmark runs         : "
            << BENCHMARK_RUNS
            << "\n";


        std::cout
            << "Average batch time     : "
            << average
            << " ms\n";


        std::cout
            << "Median batch time      : "
            << median
            << " ms\n";


        std::cout
            << "Minimum batch time     : "
            << minimum
            << " ms\n";


        std::cout
            << "Maximum batch time     : "
            << maximum
            << " ms\n";


        std::cout
            << "Std deviation          : "
            << stdDev
            << " ms\n";


        std::cout
            << "Average time / image   : "
            << timePerImage
            << " ms\n";


        std::cout
            << "Inference accuracy     : "
            << benchmarkAccuracy
            << "%\n";


        std::cout
            << "============================================================\n";


        return 0;

    }

    catch (
        const c10::Error& error
    ) {

        std::cerr
            << "\nLibTorch error:\n"
            << error.what()
            << "\n";

        return 1;
    }

    catch (
        const std::exception& error
    ) {

        std::cerr
            << "\nError:\n"
            << error.what()
            << "\n";

        return 1;
    }
}