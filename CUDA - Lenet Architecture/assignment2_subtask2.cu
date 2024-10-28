//s2 code                           
        
#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <iomanip>
using namespace std;
#define BLOCK_SIZE 256

// CUDA kernel to perform convolution without padding

// CUDA kernel to perform convolution without padding

// CUDA kernel for convolution without padding


__global__ void reluKernel(float* input, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x; // Calculate index in 1D array
    if (idx < size * size) {
        output[idx] = fmaxf(0.0f, input[idx]); // Apply ReLU activation
    }
}

float** ReLU(float** matrix, int size) {
    // Calculate total number of elements in the matrix
    int totalElements = size * size;

    // Allocate memory for input and output matrices on the GPU
    float *d_input, *d_output;
    cudaMalloc((void**)&d_input, totalElements * sizeof(float));
    cudaMalloc((void**)&d_output, totalElements * sizeof(float));

    // Flatten the input matrix to a 1D array for GPU processing
    float* flattenedInput = new float[totalElements];
    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
            flattenedInput[i * size + j] = matrix[i][j];
        }
    }

    // Copy input matrix from host to device
    cudaMemcpy(d_input, flattenedInput, totalElements * sizeof(float), cudaMemcpyHostToDevice);

    // Define block and grid dimensions
    int blockSize = 256;
    int numBlocks = (totalElements + blockSize - 1) / blockSize;

    // Launch CUDA kernel
    reluKernel<<<numBlocks, blockSize>>>(d_input, d_output, size);

    // Allocate memory for output matrix on host
    float** output = new float*[size];
    float* outputData = new float[totalElements];
    cudaMemcpy(outputData, d_output, totalElements * sizeof(float), cudaMemcpyDeviceToHost);

    // Convert output data back to 2D matrix
    for (int i = 0; i < size; ++i) {
        output[i] = new float[size];
        for (int j = 0; j < size; ++j) {
            output[i][j] = outputData[i * size + j];
        }
    }

    // Free memory on the GPU and host
    cudaFree(d_input);
    cudaFree(d_output);
    delete[] flattenedInput;
    delete[] outputData;

    return output;
}

__global__ void tanhKernel(float* input, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x; // Calculate index in 1D array
    if (idx < size * size) {
        output[idx] = tanh(input[idx]); // Apply Tanh activation
    }
}

float** Tanh(float** matrix, int size) {
    // Calculate total number of elements in the matrix
    int totalElements = size * size;

    // Allocate memory for input and output matrices on the GPU
    float *d_input, *d_output;
    cudaMalloc((void**)&d_input, totalElements * sizeof(float));
    cudaMalloc((void**)&d_output, totalElements * sizeof(float));

    // Flatten the input matrix to a 1D array for GPU processing
    float* flattenedInput = new float[totalElements];
    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
            flattenedInput[i * size + j] = matrix[i][j];
        }
    }

    // Copy input matrix from host to device
    cudaMemcpy(d_input, flattenedInput, totalElements * sizeof(float), cudaMemcpyHostToDevice);

    // Define block and grid dimensions
    int blockSize = 256;
    int numBlocks = (totalElements + blockSize - 1) / blockSize;

    // Launch CUDA kernel
    tanhKernel<<<numBlocks, blockSize>>>(d_input, d_output, size);

    // Allocate memory for output matrix on host
    float** output = new float*[size];
    float* outputData = new float[totalElements];
    cudaMemcpy(outputData, d_output, totalElements * sizeof(float), cudaMemcpyDeviceToHost);

    // Convert output data back to 2D matrix
    for (int i = 0; i < size; ++i) {
        output[i] = new float[size];
        for (int j = 0; j < size; ++j) {
            output[i][j] = outputData[i * size + j];
        }
    }

    // Free memory on the GPU and host
    cudaFree(d_input);
    cudaFree(d_output);
    delete[] flattenedInput;
    delete[] outputData;

    return output;
}
void softmax(float* input, int size) {
    float sum = 0;
    for (int i = 0; i < size; i++) {
        input[i] = exp(input[i]);
        sum += input[i];
    }

    for (int i = 0; i < size; i++) {
        input[i] /= sum;
    }
}

// Sigmoid function for converting a vector of scores to probabilities
void sigmoid(float* input, int size) {
    for (int i = 0; i < size; ++i) {
        input[i] = 1.0 / (1.0 + exp(-input[i]));
    }
}
// __global__ void softmaxKernel(float* input, float* output, int size) {
//     int idx = blockIdx.x * blockDim.x + threadIdx.x;

//     if (idx < size) {
//         float exp_val = expf(input[idx]);
//         atomicAdd(&output[0], exp_val); // Accumulate sum of exponentials in output[0]
//         output[idx] = exp_val;
//     }
    
//     __syncthreads();

//     // Compute final softmax values using the accumulated sum
//     if (idx < size) {
//         output[idx] /= output[0];
//     }
// }

// // Function to perform softmax using CUDA
// void softmax(float* input, int size) {
//     float* d_input;
//     float* d_output;

//     // Allocate device memory for input and output arrays
//     cudaMalloc((void**)&d_input, size * sizeof(float));
//     cudaMalloc((void**)&d_output, (size + 1) * sizeof(float));

//     // Copy input array to device memory
//     cudaMemcpy(d_input, input, size * sizeof(float), cudaMemcpyHostToDevice);

//     // Initialize output[0] to 0 on device
//     cudaMemset(d_output, 0, sizeof(float));

//     // Launch the CUDA kernel for softmax computation
//     int numBlocks = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;
//     softmaxKernel<<<numBlocks, BLOCK_SIZE>>>(d_input, d_output, size);

//     // Copy the result back from device memory to host
//     cudaMemcpy(input, d_output + 1, size * sizeof(float), cudaMemcpyDeviceToHost);

//     // Free device memory
//     cudaFree(d_input);
//     cudaFree(d_output);
// }
// __global__ void sigmoidKernel(float* input, float* output, int size) {
//     int idx = blockIdx.x * blockDim.x + threadIdx.x;

//     if (idx < size) {
//         output[idx] = 1.0 / (1.0 + expf(-input[idx]));
//     }
// }

// // Function to perform sigmoid using CUDA
// void sigmoid(float* input, int size) {
//     float* d_input;
//     float* d_output;

//     // Allocate device memory for input and output arrays
//     cudaMalloc((void**)&d_input, size * sizeof(float));
//     cudaMalloc((void**)&d_output, size * sizeof(float));

//     // Copy input array to device memory
//     cudaMemcpy(d_input, input, size * sizeof(float), cudaMemcpyHostToDevice);

//     // Launch the CUDA kernel for sigmoid computation
//     int numBlocks = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;
//     sigmoidKernel<<<numBlocks, BLOCK_SIZE>>>(d_input, d_output, size);

//     // Copy the result back from device memory to host
//     cudaMemcpy(input, d_output, size * sizeof(float), cudaMemcpyDeviceToHost);

//     // Free device memory
//     cudaFree(d_input);
//     cudaFree(d_output);
// }
__global__ void averagePoolingKernel(float* input, int inputSize, int poolingSize, float* output) {
    int outputSize = inputSize / poolingSize;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < outputSize && j < outputSize) {
        float sum = 0.0f;

        // Iterate over each element in the pooling region
        for (int k = 0; k < poolingSize; ++k) {
            for (int l = 0; l < poolingSize; ++l) {
                int inputRow = i * poolingSize + k;
                int inputCol = j * poolingSize + l;
                int idx = inputRow * inputSize + inputCol;

                // Add element to sum if within input bounds
                if (inputRow < inputSize && inputCol < inputSize) {
                    sum += input[idx];
                }
            }
        }

        // Calculate average for the pooling region
        float avg = sum / (poolingSize * poolingSize);
        int outputIdx = i * outputSize + j;
        output[outputIdx] = avg;
    }
}

// Function to perform average pooling on the GPU
float* averagePooling(float* input, int inputSize, int poolingSize) {
    int outputSize = inputSize / poolingSize;
    int inputSizeBytes = inputSize * inputSize * sizeof(float);
    int outputSizeBytes = outputSize * outputSize * sizeof(float);
    float* d_input, *d_output;
    float* output = new float[outputSize * outputSize];

    // Allocate device memory for input and output arrays
    cudaMalloc((void**)&d_input, inputSizeBytes);
    cudaMalloc((void**)&d_output, outputSizeBytes);

    // Copy input array from host to device
    cudaMemcpy(d_input, input, inputSizeBytes, cudaMemcpyHostToDevice);

    // Define block size and grid size for CUDA kernel
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((outputSize + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (outputSize + threadsPerBlock.y - 1) / threadsPerBlock.y);

    // Launch CUDA kernel for average pooling
    averagePoolingKernel<<<numBlocks, threadsPerBlock>>>(d_input, inputSize, poolingSize, d_output);

    // Copy output array from device to host
    cudaMemcpy(output, d_output, outputSizeBytes, cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_output);

    return output;
}
// CUDA kernel for average pooling
__global__ void maxPoolingKernel(float* input, int inputSize, int poolingSize, float* output) {
    int outputSize = inputSize / poolingSize;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < outputSize && j < outputSize) {
        float maxi = input[i * poolingSize * inputSize + j * poolingSize];

        // Iterate over each element in the pooling region
        for (int k = 0; k < poolingSize; ++k) {
            for (int l = 0; l < poolingSize; ++l) {
                int inputRow = i * poolingSize + k;
                int inputCol = j * poolingSize + l;
                int idx = inputRow * inputSize + inputCol;

                // Add element to sum if within input bounds
                if (inputRow < inputSize && inputCol < inputSize) {
                    maxi = max(maxi,input[idx]);
                }
            }
        }

        // Calculate average for the pooling region
        
        int outputIdx = i * outputSize + j;
        output[outputIdx] = maxi;
    }
    
}

// Function to perform average pooling on the GPU
float* maxPooling(float* input, int inputSize, int poolingSize) {
    int outputSize = inputSize / poolingSize;
    int inputSizeBytes = inputSize * inputSize * sizeof(float);
    int outputSizeBytes = outputSize * outputSize * sizeof(float);
    float* d_input, *d_output;
    float* output = new float[outputSize * outputSize];

    // Allocate device memory for input and output arrays
    cudaMalloc((void**)&d_input, inputSizeBytes);
    cudaMalloc((void**)&d_output, outputSizeBytes);

    // Copy input array from host to device
    cudaMemcpy(d_input, input, inputSizeBytes, cudaMemcpyHostToDevice);

    // Define block size and grid size for CUDA kernel
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((outputSize + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (outputSize + threadsPerBlock.y - 1) / threadsPerBlock.y);

    // Launch CUDA kernel for average pooling
    maxPoolingKernel<<<numBlocks, threadsPerBlock>>>(d_input, inputSize, poolingSize, d_output);

    // Copy output array from device to host
    cudaMemcpy(output, d_output, outputSizeBytes, cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_output);

    return output;
}
// CUDA kernel for convolution without padding
__global__ void convolutionWithoutPaddingKernel(float* input, int inputSize, float* kernel, int kernelSize, float* output, int bias) {
    int outputSize = inputSize - kernelSize + 1;
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < outputSize && col < outputSize) {
        float sum = 0.0f;
        for (int k = 0; k < kernelSize; ++k) {
            for (int l = 0; l < kernelSize; ++l) {
                sum += input[(row + k) * inputSize + (col + l)] * kernel[k * kernelSize + l];
            }
        }
        output[row * outputSize + col] = sum + bias;
    }
}
void printArray(float* arr, int size) {
    for (int i = 0; i < size; ++i) {
        std::cout << arr[i] << " ";
    }
    std::cout << std::endl;
}
void convolutionWithoutPadding(float* input, int inputSize, float* kernel, int kernelSize, float* output, int bias) {
    int outputSize = inputSize - kernelSize + 1;
    int inputMem = inputSize * inputSize * sizeof(float);
    int kernelMem = kernelSize * kernelSize * sizeof(float);
    int outputMem = outputSize * outputSize * sizeof(float);

    float *d_input, *d_kernel, *d_output;

    // Allocate memory on the device
    cudaMalloc((void**)&d_input, inputMem);
    cudaMalloc((void**)&d_kernel, kernelMem);
    cudaMalloc((void**)&d_output, outputMem);

    // Copy input matrix from host to device
    cudaMemcpy(d_input, input, inputMem, cudaMemcpyHostToDevice);
    // Copy kernel matrix from host to device
    cudaMemcpy(d_kernel, kernel, kernelMem, cudaMemcpyHostToDevice);

    // Define grid and block dimensions
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((outputSize + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (outputSize + threadsPerBlock.y - 1) / threadsPerBlock.y);

    // Launch kernel
    convolutionWithoutPaddingKernel<<<numBlocks, threadsPerBlock>>>(d_input, inputSize, d_kernel, kernelSize, d_output, bias);

    // Copy result back to host
    cudaMemcpy(output, d_output, outputMem, cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_kernel);
    cudaFree(d_output);
}

// __global__ void convolutionWithPaddingKernel(float* input_matrix, float* kernel, float* output_matrix, int inputSize, int kernel_size, int outputSize, int padding) {
//     int row =  threadIdx.x;
//     int col = threadIdx.y;
//     if (row < outputSize && col < outputSize) {
//         int output_index = row * outputSize + col;
//         float sum = 0.0f;

//         for (int i = 0; i < kernel_size; i++) {
//             for (int j = 0; j < kernel_size; j++) {
//                 int input_row = row + i - padding;
//                 int input_col = col + j - padding;

//                 if (input_row >= 0 && input_row < inputSize && input_col >= 0 && input_col < inputSize) {
//                     int input_index = input_row * inputSize + input_col;
//                     int weight_index = i * kernel_size + j;
//                     sum += input_matrix[input_index] * kernel[i * kernel_size + j];
//                 }
//             }
//         }
//         output_matrix[output_index] = sum;
//     }
// // }

// void convolutionWithPadding(float* input_matrix, float* kernel, float* output_matrix, int inputSize, int kernel_size, int outputSize, int outputSize, int padding) {
    

//     float* d_input_matrix; // allocate and fill input matrix in cuda
//     float* d_conv_weights; // allocate and fill weights in cuda
//     float* d_output_matrix; // allocate output buffer in cuda



//     // Allocate memory on GPU
//     cudaMalloc((void **)&d_input_matrix, sizeof(float) * inputSize * inputSize);
//     cudaMalloc((void **)&d_conv_weights, sizeof(float) * kernel_size * kernel_size );
//     cudaMalloc((void **)&d_output_matrix, sizeof(float) * outputSize * outputSize);

//     cudaMemcpy(d_input_matrix, input_matrix,sizeof(float) * inputSize * inputSize,cudaMemcpyHostToDevice);
//     cudaMemcpy(d_conv_weights, kernel,sizeof(float) * kernel_size * kernel_size ,cudaMemcpyHostToDevice);

//     dim3 threadsPerBlock(outputSize, outputSize);
//     dim3 numBlocks(1,1);

//     // Call the CUDA function
    
//     convolutionWithPaddingKernel<<<numBlocks, threadsPerBlock>>>(input_matrix, kernel, output_matrix, inputSize, kernel_size, outputSize, padding);
//     cudaDeviceSynchronize();
//     cudaMemcpy(output_matrix, d_output_matrix,sizeof(float) * outputSize * outputSize,cudaMemcpyDeviceToHost);
//     // Free allocated memory


//     cudaFree(d_input_matrix);
//     cudaFree(d_conv_weights);
//     cudaFree(d_output_matrix);
// }
    
// int main() {
// }


int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <task> [<args>]" << std::endl;
        return 1;
    }

    int task = std::atoi(argv[1]);

    if (task == 1 && argc >= 7) {
        // Convolution task
        // int padding = std::atoi(argv[2]);
        int N = std::atoi(argv[2]);
        int M = std::atoi(argv[3]);
        int P = std::atoi(argv[4]);
        float* matrix = new float[N*N];
        float* kernel = new float[M*M];
        int ct = 5;
        for (int i = 0; i < N*N; i++) {
           
                matrix[i] = std::atof(argv[ct]);
                ct++;
            
        }
        for (int i = 0; i < M*M; i++) {
            
                kernel[i] = std::atof(argv[ct]);
                ct++;
            
        }
        int outputSize = N - M + 1 + 2*P;
        float*  output = new float[outputSize*outputSize]  ;
        convolutionWithoutPadding(matrix, kernel, output, N,M,outputSize,P);
        for(int i=0;i<outputSize*outputSize;i++){
            
                cout<<output[i]<<" ";
            
        }
    } else if (task == 2 && argc >= 5) {
        // Non-linear activations task
        int activationType = std::atoi(argv[2]);
        int N = std::atoi(argv[3]);
        int M = std::atoi(argv[4]);
        float** matrix=createMatrix1(N,M);
        int ct=5;
        for(int i=0;i<N;i++){
            for(int j=0;j<M;j++){
                matrix[i][j] = std::atof(argv[ct]);
                ct++;
            }
        }
        float** output;
        if(activationType==0){
            output=ReLU(matrix, N,M);
            
        }
        
        else if(activationType==1){
            output = Tanh(matrix, N,M);
            
        }
        }
        else{
            cout<<"Invalid activation type"<<endl;
            return 1;
        }
        for(int i=0;i<N;i++){
            for(int j=0;j<M;j++){
                cout<<output[i][j]<<" ";
            }
    } }else if (task == 3 && argc >= 5) {
        // Subsampling task
        int pool = std::atoi(argv[2]);
        int pools = std::atoi(argv[3]);
        int N = std::atoi(argv[4]);
        float* matrix=new float[N*N];
        int ct=5;
        for(int i=0;i<N*N;i++){
            
                matrix[i] = std::atof(argv[ct]);
                ct++;
            
        }
        float* output;
        if(pool==0){
            output = maxPooling(matrix, N, pools);
        }
        else if(pool==1){
            output = averagePooling(matrix, N, pools);
        }
        else{
            cout<<"Invalid pooling type"<<endl;
            return 1;
        }
        for(int i=0;i<N*N/(pools*pools);i++){
            
                cout<<output[i]<<" ";
            
        }
    } else if (task == 4 && argc >= 5) {
        // Subsampling task
        int s_type = std::atoi(argv[2]);
        vector<float> inp;
        int s=3;
        while(argv[s]!=NULL){
            inp.push_back(std::atof(argv[s]));
            s++;
        }
        if(s_type==1){
            softmax(inp,inp.size());
        }
        else if(s_type==0){
            sigmoid(inp,inp.size());
        }
        else{
            cout<<"Invalid function type"<<endl;
            return 1;
        }
        for(int i=0;i<inp.size();i++){
            cout<<inp[i]<<" ";
        }
    }
    else {
        std::cerr << "Invalid task or insufficient arguments." << std::endl;
        return 1;
    }

    return 0;
}