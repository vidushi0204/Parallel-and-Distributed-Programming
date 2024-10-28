# LeNet-5 forward propagation using CUDA (Parallel Programming)

## Objective

The objective of this assignment is to implement an image processing library for recognizing MNIST digits by constructing a neural network architecture (LeNet-5) and optimizing its performance using CUDA. The assignment is divided into four subtasks:

1. Implement essential image processing functions in C++.
2. Utilize CUDA to parallelize computationally intensive operations.
3. Build the neural network to recognize digits from the MNIST dataset.
4. Use CUDA streams to improve throughput by processing multiple images concurrently.

## Subtask 1: Image Processing Functions (C++)

We implemented the following core image processing functions:

- **Convolution (With/Without Padding)**: Performs a convolution operation on an input matrix using a kernel matrix.  
  Time Complexity: **O(n²k²)** for an input of size `n` and a kernel of size `k`.

- **ReLU / Tanh Activations**: Applies Rectified Linear Unit (ReLU) or Tanh to each element.  
  Time Complexity: **O(n²)**.

- **Max/Average Pooling**: Reduces spatial dimensions by taking the maximum or average value in each pooling window.  
  Time Complexity: **O(n²p²)** for pooling window of size `p`.

- **Softmax / Sigmoid**: Converts a vector of scores into probabilities.  
  Time Complexity: **O(n)**.

## Subtask 2: Parallelization with CUDA

We converted the image processing functions into CUDA kernels:

- **CUDA Kernels**: Each thread computes a single element of the output matrix using `blockIdx` and `threadIdx` for indexing.
- **Memory Management**: Global memory is used for accessing input and kernel matrices, and results are stored in output matrices.
- **Performance**: Functions such as **convolution**, **pooling**, **Tanh**, and **ReLU** benefit from GPU acceleration, whereas **softmax** and **sigmoid** exhibit little improvement due to the overhead of transferring data between the CPU and GPU.

## Subtask 3: Neural Network Construction (LeNet-5)

We built a feedforward neural network that processes MNIST images using convolutional, pooling, and fully connected layers:

1. **Input and Weights**: Loaded from external files (`conv1.txt`, `conv2.txt`, etc.) and transferred to the GPU.
2. **Layer-by-Layer Execution**:
    - Convolution -> Pooling -> Convolution -> Pooling -> Fully Connected Layers.
    - CUDA is used to speed up each operation, particularly in convolution and pooling.
3. **Softmax**: The logits from the output layer are converted into class probabilities using the softmax function.
4. **Top-5 Predictions**: The network outputs the top-5 predictions for each input image.

## Subtask 4: Throughput Optimization using CUDA Streams

- **CUDA Streams**: We used multiple CUDA streams to process different images in parallel, effectively utilizing GPU resources.
- **Dynamic Memory Allocation**: Each stream has its own memory allocation for inputs, weights, and outputs, ensuring independent processing of each image.
- **Stream Synchronization**: All streams are synchronized to ensure that computations are completed before moving to the next step.

## Optimizations

- **Modular CUDA Kernels**: Each layer of the neural network is implemented as a separate CUDA kernel, promoting modularity and encapsulation.
- **Efficient Memory Management**: We optimized memory allocation and data transfer between CPU and GPU using `cudaMalloc` and `cudaMemcpy`.
- **Memory Coalescing**: Interleaving partitioning is used instead of block partitioning to maximize cache hits and reduce memory latency.
- **Parallel Execution**: Grid and block dimensions are adjusted to parallelize computations effectively, leveraging all available GPU cores.

## Observations

- **Accuracy**: The neural network achieved high accuracy on the MNIST dataset.
- **Performance Scaling**: Increasing the number of threads reduces runtime, but data transfer overhead between CPU and GPU increases, reducing overall efficiency. This behavior aligns with Amdahl’s Law.
