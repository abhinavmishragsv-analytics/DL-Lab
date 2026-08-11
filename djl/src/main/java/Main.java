import ai.djl.Device;
import ai.djl.Model;
import ai.djl.engine.Engine;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.Activation;
import ai.djl.nn.Block;
import ai.djl.nn.Blocks;
import ai.djl.nn.SequentialBlock;
import ai.djl.nn.core.Linear;
import ai.djl.training.DefaultTrainingConfig;
import ai.djl.training.EasyTrain;
import ai.djl.training.Trainer;
import ai.djl.training.evaluator.Accuracy;
import ai.djl.training.initializer.XavierInitializer;
import ai.djl.training.loss.Loss;
import ai.djl.training.optimizer.Adam;
import ai.djl.training.tracker.Tracker;
import ai.djl.training.dataset.ArrayDataset;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;

import java.io.DataInputStream;
import java.io.EOFException;
import java.io.File;
import java.io.IOException;

import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;


/**
 * DJL MNIST Training + Inference Benchmark
 *
 * Framework:
 *     Deep Java Library (DJL)
 *
 * Engine:
 *     PyTorch
 *
 * Device:
 *     CPU
 *
 * Model:
 *     784 -> 128 -> ReLU -> 64 -> ReLU -> 10
 *
 * Benchmark:
 *     10 images
 *     10 warm-up runs
 *     100 timed runs
 */
public class Main {


    // ============================================================
    // CONFIGURATION
    // ============================================================

    private static final int IMAGE_SIZE = 28;

    private static final int INPUT_SIZE =
            IMAGE_SIZE * IMAGE_SIZE;

    private static final int HIDDEN_1 = 128;

    private static final int HIDDEN_2 = 64;

    private static final int NUM_CLASSES = 10;

    private static final int BATCH_SIZE = 64;

    private static final int EPOCHS = 5;

    private static final int WARMUP_RUNS = 10;

    private static final int BENCHMARK_RUNS = 100;

    private static final int NUM_BENCHMARK_IMAGES = 10;


    // ============================================================
    // PATHS
    // ============================================================

    private static final Path PROJECT_DIR =
            Paths.get("..");

    private static final Path DATASET_DIR =
            PROJECT_DIR.resolve("dataset");

    private static final Path MODEL_DIR =
            Paths.get("model");

    private static final Path TRAIN_IMAGES =
            DATASET_DIR.resolve(
                    "train-images-idx3-ubyte"
            );

    private static final Path TRAIN_LABELS =
            DATASET_DIR.resolve(
                    "train-labels-idx1-ubyte"
            );

    private static final Path TEST_IMAGES =
            DATASET_DIR.resolve(
                    "test-images-idx3-ubyte"
            );

    private static final Path TEST_LABELS =
            DATASET_DIR.resolve(
                    "test-labels-idx1-ubyte"
            );

    private static final Path BENCHMARK_IMAGE_DIR =
            DATASET_DIR.resolve(
                    "benchmark_images"
            );


    // ============================================================
    // FIXED BENCHMARK LABELS
    // ============================================================

    private static final int[] TRUE_LABELS = {
            7, 2, 1, 0, 4,
            1, 4, 9, 5, 9
    };


    // ============================================================
    // MAIN
    // ============================================================

    public static void main(String[] args)
            throws Exception {


        printHeader();


        verifyFiles();


        System.out.println(
                "Backend     : "
                        + Engine.getDefaultEngineName()
        );

        System.out.println(
                "Device      : CPU"
        );

        System.out.println(
                "Model       : "
                        + INPUT_SIZE
                        + " -> "
                        + HIDDEN_1
                        + " -> "
                        + HIDDEN_2
                        + " -> "
                        + NUM_CLASSES
        );

        System.out.println(
                "Epochs      : "
                        + EPOCHS
        );

        System.out.println(
                "Batch size  : "
                        + BATCH_SIZE
        );


        // ========================================================
        // LOAD DATASET
        // ========================================================

        System.out.println();

        System.out.println(
                "Loading MNIST IDX dataset..."
        );


        MNISTData trainData =
                loadMNIST(
                        TRAIN_IMAGES,
                        TRAIN_LABELS
                );


        MNISTData testData =
                loadMNIST(
                        TEST_IMAGES,
                        TEST_LABELS
                );


        System.out.println(
                "Training images : "
                        + trainData.size
        );

        System.out.println(
                "Test images     : "
                        + testData.size
        );


        // ========================================================
        // DJL MANAGER
        // ========================================================

        try (
                NDManager manager =
                        NDManager.newBaseManager()
        ) {


            // ====================================================
            // CREATE NDARRAYS
            // ====================================================

            NDArray trainImages =
                    manager.create(
                            trainData.images,
                            new Shape(
                                    trainData.size,
                                    IMAGE_SIZE,
                                    IMAGE_SIZE
                            )
                    );


            NDArray trainLabels =
                    manager.create(
                            trainData.labels
                    )
                    .toType(
                            DataType.INT64,
                            false
                    );


            NDArray testImages =
                    manager.create(
                            testData.images,
                            new Shape(
                                    testData.size,
                                    IMAGE_SIZE,
                                    IMAGE_SIZE
                            )
                    );


            NDArray testLabels =
                    manager.create(
                            testData.labels
                    )
                    .toType(
                            DataType.INT64,
                            false
                    );


            // ====================================================
            // DATASETS
            // ====================================================

            ArrayDataset trainDataset =
                    new ArrayDataset.Builder()
                            .setData(
                                    trainImages
                            )
                            .optLabels(
                                    trainLabels
                            )
                            .setSampling(
                                    BATCH_SIZE,
                                    true
                            )
                            .build();


            ArrayDataset testDataset =
                    new ArrayDataset.Builder()
                            .setData(
                                    testImages
                            )
                            .optLabels(
                                    testLabels
                            )
                            .setSampling(
                                    BATCH_SIZE,
                                    false
                            )
                            .build();


            // ====================================================
            // MODEL
            // ====================================================

            Model model =
                    Model.newInstance(
                            "mnist_djl"
                    );


            Block block =
                    buildModel();


            model.setBlock(
                    block
            );


            // ====================================================
            // TRAINING CONFIG
            // ====================================================

            Adam optimizer =
                    Adam.builder()
                            .optLearningRateTracker(
                                    Tracker.fixed(
                                            0.001f
                                    )
                            )
                            .build();


            DefaultTrainingConfig config =
                    new DefaultTrainingConfig(
                            Loss.softmaxCrossEntropyLoss()
                    )
                    .optOptimizer(
                            optimizer
                    )
                    .optInitializer(
                            new XavierInitializer(),
                            ai.djl.nn.Parameter.Type.WEIGHT
                    )
                    .addEvaluator(
                            new Accuracy()
                    );


            // ====================================================
            // TRAIN
            // ====================================================

            System.out.println();

            System.out.println(
                    "============================================================"
            );

            System.out.println(
                    "TRAINING"
            );

            System.out.println(
                    "============================================================"
            );


            long trainingStart =
                    System.nanoTime();


            try (
                    Trainer trainer =
                            model.newTrainer(
                                    config
                            )
            ) {


                trainer.initialize(
                        new Shape(
                                BATCH_SIZE,
                                IMAGE_SIZE,
                                IMAGE_SIZE
                        )
                );


                EasyTrain.fit(
                        trainer,
                        EPOCHS,
                        trainDataset,
                        testDataset
                );
            }


            long trainingEnd =
                    System.nanoTime();


            double trainingTime =
                    (
                            trainingEnd
                                    - trainingStart
                    )
                            / 1_000_000_000.0;


            // ====================================================
            // TEST ACCURACY
            // ====================================================

            double testAccuracy =
                    evaluateAccuracy(
                            model,
                            testImages,
                            testLabels,
                            manager
                    );


            System.out.println();

            System.out.println(
                    "============================================================"
            );

            System.out.println(
                    "TRAINING COMPLETE"
            );

            System.out.println(
                    "============================================================"
            );

            System.out.printf(
                    "Training time : %.2f seconds%n",
                    trainingTime
            );

            System.out.printf(
                    "Test accuracy : %.2f%%%n",
                    testAccuracy
            );


            // ====================================================
            // SAVE MODEL PARAMETERS
            // ====================================================

            Files.createDirectories(
                    MODEL_DIR
            );


            model.save(
                    MODEL_DIR,
                    "mnist_djl"
            );


            System.out.println();

            System.out.println(
                    "Model saved to:"
            );

            System.out.println(
                    MODEL_DIR.toAbsolutePath()
            );


            // ====================================================
            // LOAD BENCHMARK IMAGES
            // ====================================================

            NDArray benchmarkBatch =
                    loadBenchmarkImages(
                            manager
                    );


            System.out.println();

            System.out.println(
                    "Benchmark images loaded."
            );

            System.out.println(
                    "Benchmark input shape: "
                            + benchmarkBatch.getShape()
            );


            // ====================================================
            // BENCHMARK
            // ====================================================

            runBenchmark(
                    model,
                    benchmarkBatch,
                    manager
            );


            model.close();
        }
    }


    // ============================================================
    // MODEL ARCHITECTURE
    // ============================================================

    private static Block buildModel() {


        SequentialBlock block =
                new SequentialBlock();


        // 28 x 28 -> 784

        block.add(
                Blocks.batchFlattenBlock()
        );


        // 784 -> 128

        block.add(
                Linear.builder()
                        .setUnits(
                                HIDDEN_1
                        )
                        .build()
        );


        // ReLU

        block.add(
                Activation.reluBlock()
        );


        // 128 -> 64

        block.add(
                Linear.builder()
                        .setUnits(
                                HIDDEN_2
                        )
                        .build()
        );


        // ReLU

        block.add(
                Activation.reluBlock()
        );


        // 64 -> 10

        block.add(
                Linear.builder()
                        .setUnits(
                                NUM_CLASSES
                        )
                        .build()
        );


        return block;
    }


    // ============================================================
    // EVALUATE TEST ACCURACY
    // ============================================================

    private static double evaluateAccuracy(
            Model model,
            NDArray images,
            NDArray labels,
            NDManager manager
    ) {


        int total =
                (int)
                        images.getShape()
                                .get(0);


        int correct =
                0;


        ai.djl.training.ParameterStore parameterStore =
                new ai.djl.training.ParameterStore(
                        manager,
                        false
                );


        int evaluationBatch =
                256;


        for (
                int start = 0;
                start < total;
                start += evaluationBatch
        ) {


            int end =
                    Math.min(
                            start
                                    + evaluationBatch,
                            total
                    );


            NDArray batch =
                    images.get(
                            new ai.djl.ndarray.index.NDIndex(
                                    start
                                            + ":"
                                            + end
                            )
                    );


            NDArray target =
                    labels.get(
                            new ai.djl.ndarray.index.NDIndex(
                                    start
                                            + ":"
                                            + end
                            )
                    );


            NDList output =
                    model.getBlock()
                            .forward(
                                    parameterStore,
                                    new NDList(
                                            batch
                                    ),
                                    false
                            );


            NDArray prediction =
                    output.singletonOrThrow()
                            .argMax(
                                    1
                            );


            NDArray matches =
                    prediction
                            .eq(
                                    target
                            );


            correct +=
                    (int)
                            matches.sum()
                                    .getLong();
        }


        return
                100.0
                        * correct
                        / total;
    }


    // ============================================================
    // BENCHMARK
    // ============================================================

    private static void runBenchmark(
            Model model,
            NDArray benchmarkBatch,
            NDManager manager
    ) {


        ai.djl.training.ParameterStore parameterStore =
                new ai.djl.training.ParameterStore(
                        manager,
                        false
                );


        System.out.println();

        System.out.println(
                "============================================================"
        );

        System.out.println(
                "INFERENCE BENCHMARK"
        );

        System.out.println(
                "============================================================"
        );


        // ========================================================
        // WARM-UP
        // ========================================================

        System.out.println();

        System.out.println(
                "Running warm-up..."
        );


        for (
                int i = 0;
                i < WARMUP_RUNS;
                i++
        ) {


            model.getBlock()
                    .forward(
                            parameterStore,
                            new NDList(
                                    benchmarkBatch
                            ),
                            false
                    );
        }


        System.out.println(
                "Warm-up complete."
        );


        // ========================================================
        // BENCHMARK
        // ========================================================

        List<Double> times =
                new ArrayList<>();


        NDList output =
                null;


        System.out.println();

        System.out.println(
                "Running benchmark..."
        );


        for (
                int run = 0;
                run < BENCHMARK_RUNS;
                run++
        ) {


            long start =
                    System.nanoTime();


            output =
                    model.getBlock()
                            .forward(
                                    parameterStore,
                                    new NDList(
                                            benchmarkBatch
                                    ),
                                    false
                            );


            long end =
                    System.nanoTime();


            double elapsed =
                    (
                            end - start
                    )
                            / 1_000_000.0;


            times.add(
                    elapsed
            );
        }


        // ========================================================
        // PREDICTIONS
        // ========================================================

        NDArray logits =
                output.singletonOrThrow();


        NDArray predictions =
                logits.argMax(
                        1
                );


        long[] predictedLabels =
                predictions.toLongArray();


        int correct =
                0;


        System.out.println();

        System.out.println(
                "============================================================"
        );

        System.out.println(
                "PREDICTIONS"
        );

        System.out.println(
                "============================================================"
        );


        for (
                int i = 0;
                i < NUM_BENCHMARK_IMAGES;
                i++
        ) {


            int predicted =
                    (int)
                            predictedLabels[i];


            int actual =
                    TRUE_LABELS[i];


            boolean correctPrediction =
                    predicted == actual;


            if (
                    correctPrediction
            ) {

                correct++;
            }


            System.out.printf(
                    "image_%02d.png | True: %d | Predicted: %d | %s%n",
                    i + 1,
                    actual,
                    predicted,
                    correctPrediction
                            ? "CORRECT"
                            : "WRONG"
            );
        }


        double accuracy =
                100.0
                        * correct
                        / NUM_BENCHMARK_IMAGES;


        // ========================================================
        // STATISTICS
        // ========================================================

        double average =
                calculateAverage(
                        times
                );


        double median =
                calculateMedian(
                        times
                );


        double minimum =
                Collections.min(
                        times
                );


        double maximum =
                Collections.max(
                        times
                );


        double stdDev =
                calculateStdDev(
                        times,
                        average
                );


        double perImage =
                average
                        / NUM_BENCHMARK_IMAGES;


        // ========================================================
        // FINAL RESULTS
        // ========================================================

        System.out.println();

        System.out.println(
                "============================================================"
        );

        System.out.println(
                "BENCHMARK RESULTS"
        );

        System.out.println(
                "============================================================"
        );


        System.out.printf(
                "Number of images       : %d%n",
                NUM_BENCHMARK_IMAGES
        );


        System.out.printf(
                "Warm-up runs           : %d%n",
                WARMUP_RUNS
        );


        System.out.printf(
                "Benchmark runs         : %d%n",
                BENCHMARK_RUNS
        );


        System.out.printf(
                "Average batch time     : %.4f ms%n",
                average
        );


        System.out.printf(
                "Median batch time      : %.4f ms%n",
                median
        );


        System.out.printf(
                "Minimum batch time     : %.4f ms%n",
                minimum
        );


        System.out.printf(
                "Maximum batch time     : %.4f ms%n",
                maximum
        );


        System.out.printf(
                "Std deviation          : %.4f ms%n",
                stdDev
        );


        System.out.printf(
                "Average time / image   : %.4f ms%n",
                perImage
        );


        System.out.printf(
                "Inference accuracy     : %.2f%%%n",
                accuracy
        );


        System.out.println(
                "============================================================"
        );
    }


    // ============================================================
    // LOAD BENCHMARK PNG IMAGES
    // ============================================================

    private static NDArray loadBenchmarkImages(
            NDManager manager
    ) throws IOException {


        float[] data =
                new float[
                        NUM_BENCHMARK_IMAGES
                                * IMAGE_SIZE
                                * IMAGE_SIZE
                ];


        for (
                int i = 0;
                i < NUM_BENCHMARK_IMAGES;
                i++
        ) {


            Path imagePath =
                    BENCHMARK_IMAGE_DIR.resolve(
                            String.format(
                                    "image_%02d.png",
                                    i + 1
                            )
                    );


            BufferedImage image =
                    ImageIO.read(
                            imagePath.toFile()
                    );


            if (
                    image == null
            ) {

                throw new IOException(
                        "Could not read: "
                                + imagePath
                );
            }


            for (
                    int y = 0;
                    y < IMAGE_SIZE;
                    y++
            ) {


                for (
                        int x = 0;
                        x < IMAGE_SIZE;
                        x++
                ) {


                    int rgb =
                            image.getRGB(
                                    x,
                                    y
                            );


                    int gray =
                            rgb & 0xFF;


                    data[
                            i * INPUT_SIZE
                                    +
                            y * IMAGE_SIZE
                                    +
                            x
                    ] =
                            gray
                                    / 255.0f;
                }
            }
        }


        return manager.create(
                data,
                new Shape(
                        NUM_BENCHMARK_IMAGES,
                        IMAGE_SIZE,
                        IMAGE_SIZE
                )
        );
    }


    // ============================================================
    // LOAD MNIST IDX DATA
    // ============================================================

    private static MNISTData loadMNIST(
            Path imagePath,
            Path labelPath
    ) throws IOException {


        try (
                DataInputStream images =
                        new DataInputStream(
                                Files.newInputStream(
                                        imagePath
                                )
                        );

                DataInputStream labels =
                        new DataInputStream(
                                Files.newInputStream(
                                        labelPath
                                )
                        )
        ) {


            int imageMagic =
                    images.readInt();


            int labelMagic =
                    labels.readInt();


            if (
                    imageMagic != 2051
            ) {

                throw new IOException(
                        "Invalid MNIST image file: "
                                + imagePath
                );
            }


            if (
                    labelMagic != 2049
            ) {

                throw new IOException(
                        "Invalid MNIST label file: "
                                + labelPath
                );
            }


            int count =
                    images.readInt();


            int rows =
                    images.readInt();


            int columns =
                    images.readInt();


            int labelCount =
                    labels.readInt();


            if (
                    count != labelCount
            ) {

                throw new IOException(
                        "Image/label count mismatch."
                );
            }


            if (
                    rows != IMAGE_SIZE
                            ||
                    columns != IMAGE_SIZE
            ) {

                throw new IOException(
                        "Unexpected MNIST image size."
                );
            }


            float[] imageData =
                    new float[
                            count
                                    * INPUT_SIZE
                    ];


            float[] labelData =
                    new float[
                            count
                    ];


            for (
                    int i = 0;
                    i < count;
                    i++
            ) {


                for (
                        int pixel = 0;
                        pixel < INPUT_SIZE;
                        pixel++
                ) {


                    int value =
                            images.readUnsignedByte();


                    imageData[
                            i * INPUT_SIZE
                                    +
                            pixel
                    ] =
                            value
                                    / 255.0f;
                }


                labelData[i] =
                        labels.readUnsignedByte();
            }


            return new MNISTData(
                    imageData,
                    labelData,
                    count
            );
        }
    }


    // ============================================================
    // VERIFY FILES
    // ============================================================

    private static void verifyFiles()
            throws IOException {


        Path[] requiredFiles = {

                TRAIN_IMAGES,

                TRAIN_LABELS,

                TEST_IMAGES,

                TEST_LABELS,

                BENCHMARK_IMAGE_DIR.resolve(
                        "image_01.png"
                ),

                BENCHMARK_IMAGE_DIR.resolve(
                        "labels.csv"
                )
        };


        for (
                Path file :
                requiredFiles
        ) {


            if (
                    !Files.exists(file)
            ) {

                throw new IOException(
                        "Required file not found: "
                                + file.toAbsolutePath()
                );
            }
        }
    }


    // ============================================================
    // HEADER
    // ============================================================

    private static void printHeader() {


        System.out.println();

        System.out.println(
                "============================================================"
        );

        System.out.println(
                "DJL MNIST TRAINING + INFERENCE BENCHMARK"
        );

        System.out.println(
                "============================================================"
        );

        System.out.println();

        System.out.println(
                "DJL version : 0.36.0"
        );
    }


    // ============================================================
    // AVERAGE
    // ============================================================

    private static double calculateAverage(
            List<Double> values
    ) {


        double sum =
                0.0;


        for (
                double value :
                values
        ) {

            sum += value;
        }


        return sum /
                values.size();
    }


    // ============================================================
    // MEDIAN
    // ============================================================

    private static double calculateMedian(
            List<Double> values
    ) {


        List<Double> sorted =
                new ArrayList<>(
                        values
                );


        Collections.sort(
                sorted
        );


        int middle =
                sorted.size()
                        / 2;


        if (
                sorted.size() % 2 == 0
        ) {


            return (
                    sorted.get(
                            middle - 1
                    )
                            +
                    sorted.get(
                            middle
                    )
            ) / 2.0;


        } else {


            return sorted.get(
                    middle
            );
        }
    }


    // ============================================================
    // STANDARD DEVIATION
    // ============================================================

    private static double calculateStdDev(
            List<Double> values,
            double mean
    ) {


        double sum =
                0.0;


        for (
                double value :
                values
        ) {


            double difference =
                    value - mean;


            sum +=
                    difference
                            * difference;
        }


        return Math.sqrt(
                sum /
                        values.size()
        );
    }


    // ============================================================
    // MNIST DATA HOLDER
    // ============================================================

    private static class MNISTData {


        final float[] images;

        final float[] labels;

        final int size;


        MNISTData(
                float[] images,
                float[] labels,
                int size
        ) {

            this.images =
                    images;

            this.labels =
                    labels;

            this.size =
                    size;
        }
    }
}