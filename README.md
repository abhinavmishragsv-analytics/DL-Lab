# Deep Learning Framework Comparison



A comparative benchmarking project evaluating multiple deep learning frameworks using the MNIST handwritten digit classification dataset.



The project compares three Python-based frameworks and two non-Python frameworks under a common neural network architecture and CPU-based inference benchmark.



\---



## Frameworks Compared



### Python Frameworks



1\. TensorFlow

2\. Keras

3\. PyTorch



### Non-Python Frameworks



4\. DJL (Deep Java Library)

5\. LibTorch (C++ API for PyTorch)



\---



## Objective



The objective of this project is to implement the same MNIST classification task across multiple deep learning frameworks and compare their:



\- Training performance

\- Classification accuracy

\- Model inference performance

\- Benchmark inference accuracy

\- Ease of implementation across different programming ecosystems



\---



## Dataset



The project uses the MNIST handwritten digit dataset.



MNIST contains:



\- 60,000 training images

\- 10,000 test images

\- Image dimensions: 28 × 28 pixels

\- 10 classes representing digits 0–9



Images are normalized from the original `\[0, 255]` pixel range to `\[0, 1]`.



\---



\## Neural Network Architecture



The implementations use a common fully connected neural network architecture:



&#x20;   784 → 128 → ReLU → 64 → ReLU → 10



Where:



\- Input layer: 784 neurons (28 × 28)

\- Hidden layer 1: 128 neurons

\- Activation: ReLU

\- Hidden layer 2: 64 neurons

\- Activation: ReLU

\- Output layer: 10 neurons



The output corresponds to the ten MNIST digit classes.



\---



\## Benchmark Methodology



A common set of 10 MNIST images is used for inference benchmarking.



The benchmark configuration is:



\- Benchmark images: 10

\- Warm-up runs: 10

\- Benchmark runs: 100

\- Device: CPU



For each framework, predictions are recorded for the same benchmark images.



The benchmark reports:



\- Average batch inference time

\- Median inference time

\- Minimum inference time

\- Maximum inference time

\- Standard deviation

\- Average inference time per image

\- Benchmark classification accuracy



\---



\## Benchmark Images



The benchmark images are stored in:



&#x20;   dataset/benchmark\_images/



The directory contains:



\- `image\_01.png`

\- `image\_02.png`

\- ...

\- `image\_10.png`

\- `labels.csv`

\- `benchmark\_images.npy`



The benchmark labels are stored in `labels.csv`.



\---



\## Framework Implementations



\### TensorFlow



The TensorFlow implementation provides:



\- MNIST model training

\- Test-set evaluation

\- Model saving

\- Benchmark inference

\- Inference timing



\### Keras



The Keras implementation provides:



\- MNIST model training

\- Test-set evaluation

\- Model saving

\- Benchmark inference

\- Inference timing



\### PyTorch



The PyTorch implementation provides:



\- MNIST model training

\- Test-set evaluation

\- Model saving

\- Benchmark inference

\- Inference timing



\### DJL



DJL provides the Java-based implementation.



The project uses:



\- DJL 0.36.0

\- PyTorch engine

\- CPU backend

\- Maven for dependency management and compilation



The implementation performs MNIST training and benchmark inference using Java.



\### LibTorch



LibTorch provides the C++ implementation using the PyTorch C++ API.



The project uses:



\- LibTorch 2.7.1

\- CPU backend

\- CMake

\- Microsoft Visual C++ compiler



The implementation loads the common MNIST benchmark images and performs TorchScript inference.



\---



\## Results



The currently recorded results are:



| Framework | Language | Training Time (s) | Test Accuracy (%) | Benchmark Accuracy (%) | Avg. Time/Image (ms) |

|---|---|---:|---:|---:|---:|

| TensorFlow | Python | 16.28 | 97.04 | 90.00 | 7.2410 |

| Keras | Python | 8.77 | 96.88 | 90.00 | 7.8613 |

| PyTorch | Python | 7.48 | 97.29 | 90.00 | 0.0100 |

| DJL | Java | 18.63 | 97.78 | 90.00 | — |

| LibTorch | C++ | — | — | 90.00 | 0.0286 |



The values represent the outputs obtained during the project runs.



\---



\## Project Structure



```text

DL-Framework-Comparison/

│

├── dataset/

│   ├── download\_mnist.py

│   └── benchmark\_images/

│

├── tensorflow/

│

├── keras/

│

├── pytorch/

│

├── djl/

│   ├── pom.xml

│   ├── src/

│   └── model/

│

├── libtorch/

│   ├── CMakeLists.txt

│   ├── main.cpp

│   └── model/

│

├── results/

│   └── results.csv

│

├── README.md

├── requirements.txt

└── .gitignore

