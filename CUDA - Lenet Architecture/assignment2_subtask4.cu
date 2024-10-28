#include <iostream>
#include <dirent.h>
#include <cuda_runtime.h>
#include <fstream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <iomanip>
#include <unistd.h>
using namespace std;
// namespace fs = std::filesystem; 
#define BLOCK_SIZE 256

int returnAlloctatedMem(int input){
	return input * input * sizeof(float);
}

int countFilesInDirectory(const std::string& directoryPath) {
    int fileCount = 0;
    DIR* dir;
   	struct dirent* entry;

   	if ((dir = opendir(directoryPath.c_str())) != NULL){
   		while ((entry = readdir(dir)) != NULL){
   			if (entry->d_type == DT_REG) fileCount++;
   		}
   		closedir(dir);
   	}

    return fileCount;
}

float* loadWeights(string filename,int size){
    float* weights = new float[size];
    ifstream file(filename);
    if(file.is_open()){
        for(int i = 0; i < size; i++){
            file >> weights[i];
        }
        file.close();
    }
    return weights;
}

__global__ void RELU(float* input, int size, float* output){
	int x = blockIdx.x * blockDim.x + threadIdx.x;

	if (x < size){
		output[x] = fmaxf(0.0f, input[x]);
	}
}

__global__ void convolutionWithoutPadding4D(float* input, int inputSize, int inputChannels, float* kernels, int kernelSize, int outputChannels, float* output, float* bias){
	// Calculate row. 
	int row = blockIdx.x * blockDim.x + threadIdx.x;
	// Calculate column. 
	int column = blockIdx.y * blockDim.y + threadIdx.y;
	// Calculate outChannel. 
	int outChannel = blockIdx.z;
	// Calculate size of output. 
	int outputSize = 1 + inputSize - kernelSize;

	if (outChannel < outputChannels && row < outputSize && column < outputSize){
		float sum = bias[outChannel]; 
		for (int inputChannel = 0; inputChannel < inputChannels; inputChannel++){
			for (int x = 0; x < kernelSize; x++){
				for (int y = 0; y < kernelSize; y++){
					int inputID = inputChannel * inputSize * inputSize + (x+row) * inputSize + (y+column);
					int kernelID = outChannel * inputChannels * kernelSize * kernelSize + inputChannel * kernelSize * kernelSize + (x)* kernelSize + (y);
					sum += input[inputID] * kernels[kernelID];
			}
		}
	}
	output[outChannel * outputSize * outputSize + row * outputSize + column] = sum; 

	}

	
}


__global__ void maxPooling3D(float* input, int inputSize, int inputChannels, int poolingSize, float* output){
	int outputSize = inputSize/poolingSize;
	int channel = blockIdx.z;
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	if (channel * outputSize * outputSize + x * outputSize + y < outputSize*outputSize*inputChannels) {
		float maxVal = input[channel*inputSize*inputSize + x*inputSize*poolingSize + y*poolingSize];
		for (int k = 0; k < poolingSize; k++){
			for (int l = 0; l < poolingSize; l++){
				float comparator = input[channel*inputSize*inputSize + (x*poolingSize+k)*inputSize + (y*poolingSize+l)];
				maxVal = max(maxVal, comparator);
			}
		}

	output[channel*outputSize*outputSize + x*outputSize + y] = maxVal; 
	}
}

__global__ void softmaxKernel(float* input, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size){
    	float sum = 0.0f;

    	for (int i=0; i<size; i++){
    		sum += expf(input[i]);
    	}

    	output[idx] = expf(input[idx])/sum;
    }
}

int main(int argc, char* argv[]){
	// ###### Input kernels and biases, send to GPU #####
	// Allocate on CPU & GPU 
	// Read from file into CPU
	// CudaMemcpy to GPU
	bool use_stream = (argv[1][0] == '1');

	char cwd[1024];
	if (getcwd(cwd, sizeof(cwd))==NULL){
		std::cerr << "Cannot get current directory" << endl; 
		return -1; 
	}
	string currentDirectory = string(cwd);

	float* kernels[4];
	int kernelSizes[4] = {5, 5, 4, 1};
	int kernelInputChannels[4] = {1, 20, 50, 500};
	int kernelOutputChannels[4] = {20, 50, 500, 10};
	float* biases[4];

	float* d_kernels[4];
	float* d_biases[4];

	string ker_file_names[4] = {"conv1.txt", "conv2.txt", "fc1.txt", "fc2.txt"};
	
	for (int l=0; l<4; l++){
		size_t sz = kernelSizes[l] * kernelSizes[l] * kernelInputChannels[l] * kernelOutputChannels[l]*sizeof(float);
		kernels[l] = (float*)malloc(sz);
		biases[l] = (float*)malloc(kernelOutputChannels[l]*sizeof(float));

		ifstream inpFile(currentDirectory + "/trained_weights/" + ker_file_names[l]);

		for (int out=0; out<kernelOutputChannels[l]; out++){
			for (int in = 0; in < kernelInputChannels[l]; in++)
				for (int i=0; i<kernelSizes[l]; i++){
					for (int j=0; j<kernelSizes[l]; j++){
						int idx = out * kernelInputChannels[l] * kernelSizes[l] * kernelSizes[l] + in * kernelSizes[l] * kernelSizes[l]+ i * kernelSizes[l] + j;
						inpFile >> kernels[l][idx];
					}
				}
			}

		for (int out=0; out < kernelOutputChannels[l]; out++){
			inpFile >> biases[l][out];
		}

		cudaMalloc(&d_kernels[l], sz);
		cudaMalloc(&d_biases[l], kernelOutputChannels[l]*sizeof(float));
		cudaMemcpy(d_kernels[l], kernels[l], sz, cudaMemcpyHostToDevice);
		cudaMemcpy(d_biases[l], biases[l], kernelOutputChannels[l] * sizeof(float), cudaMemcpyHostToDevice);
		} 

	// ######## Input images ###########

    string adr = "pre-proc-imgs";
    currentDirectory = currentDirectory + "/" + adr;
    int n = countFilesInDirectory(currentDirectory);
    float** images = new float*[n];
    vector<string> filenames(n);
    int img = 0;

    // Iterate over each entry (file or directory) in the current directory
    float* output[n];
    DIR* dir;
    struct dirent* entry; 
    if ((dir = opendir(currentDirectory.c_str())) != NULL){
    	while ((entry = readdir(dir)) != NULL){
    		if (entry->d_name[0]!='.'){
    			float* image = loadWeights(currentDirectory + "/" + string(entry->d_name), 784);
    			filenames[img] = entry->d_name;
    			images[img] = image;
    			img++; 
    		}
    	}
    }

    float* d_inputs[n], *d_intermediates1[n], *d_intermediates2[n], *d_intermediates3[n];
    float* d_intermediates4[n], *d_intermediates5[n], *d_intermediates6[n], *d_intermediates7[n], *d_outputs[n];
    for (int img = 0; img < n; img++){
    	float* d_input;
		cudaMalloc(&d_input, 784 * sizeof(float));
		d_inputs[img] = d_input;
		float* d_intermediate1;
		cudaMalloc(&d_intermediate1, 20*24*24*sizeof(float)); 
		d_intermediates1[img] = d_intermediate1;
		float* d_intermediate2;
		cudaMalloc(&d_intermediate2, 20*12*12*sizeof(float));
		d_intermediates2[img] = d_intermediate2;
		float* d_intermediate3;
		cudaMalloc(&d_intermediate3, 50*8*8*sizeof(float));
		d_intermediates3[img] = d_intermediate3;
		float* d_intermediate4;
		cudaMalloc(&d_intermediate4, 50*4*4*sizeof(float));
		d_intermediates4[img] = d_intermediate4;
		float* d_intermediate5;
		cudaMalloc(&d_intermediate5, 500*1*1*sizeof(float));
		d_intermediates5[img] = d_intermediate5;
		float* d_intermediate6;
		cudaMalloc(&d_intermediate6, 500*sizeof(float));
		d_intermediates6[img] = d_intermediate6;
		float* d_intermediate7; 
		cudaMalloc(&d_intermediate7, 10*1*1*sizeof(float));
		d_intermediates7[img] = d_intermediate7;
		float* d_output; 
		cudaMalloc(&d_output, 10*1*1*sizeof(float));
		d_outputs[img] = d_output; 
		output[img] = (float*)malloc(10*sizeof(float));

    }

	// CREATE STREAMS. 
		dim3 BS1(1,1,20);
		dim3 TPB1(24,24);
		dim3 BS2(1,1,20);
		dim3 TPB2(12,12);
		dim3 BS3(1,1,50);
		dim3 TPB3(8,8);
		dim3 BS4(1,1,50);
		dim3 TPB4(4,4);
		dim3 BS5(1,1,500);
		dim3 TPB5(1,1); 
		dim3 BS6(1);
		dim3 TPB6(500);
		dim3 BS7(1,1,10);
		dim3 TPB7(1);
		dim3 BS8(1);
		dim3 TPB8(10); 

	for (int img = 0; img < n; img++){
		cudaStream_t stream;
		if (use_stream){	
			cudaStreamCreate(&stream);
		} 
		else stream = 0;
		// float* d_output = d_outputs[img]; 
		// float* d_intermediate1 = d_intermediates1[img];
		// float* d_intermediate2 = d_intermediates2[img];
		// float* d_intermediate3 = d_intermediates3[img];
		// float* d_intermediate4 = d_intermediates4[img];
		// float* d_intermediate5 = d_intermediates5[img];
		// float* d_intermediate6 = d_intermediates6[img];
		// float* d_intermediate7 = d_intermediates7[img];

		// float* d_input;
		// cudaMalloc(&d_input, 784 * sizeof(float));
		cudaMemcpyAsync(d_inputs[img], images[img], 784 * sizeof(float), cudaMemcpyHostToDevice, stream);

		// CONV2D_LAYER1
		// dim3 BS1(1,1,20);
		// dim3 TPB1(24,24);
		// float* d_intermediate1;
		// cudaMalloc(&d_intermediate1, 20*24*24*sizeof(float));
		convolutionWithoutPadding4D<<<BS1, TPB1, 0, stream>>>(d_inputs[img], 28, kernelInputChannels[0], d_kernels[0], kernelSizes[0],kernelOutputChannels[0], d_intermediates1[img], d_biases[0]);


		// POOLING1
		// dim3 BS2(1,1,20);
		// dim3 TPB2(12,12);
		// float* d_intermediate2;
		// cudaMalloc(&d_intermediate2, 20*12*12*sizeof(float));
		maxPooling3D<<<BS2, TPB2, 0, stream>>>(d_intermediates1[img], 24, 20, 2, d_intermediates2[img]);


		// CONV2D_LAYER2
		// dim3 BS3(1,1,50);
		// dim3 TPB3(8,8);
		// float* d_intermediate3;
		// cudaMalloc(&d_intermediate3, 50*8*8*sizeof(float));
		convolutionWithoutPadding4D<<<BS3, TPB3, 0, stream>>>(d_intermediates2[img], 12, kernelInputChannels[1], d_kernels[1], kernelSizes[1], kernelOutputChannels[1], d_intermediates3[img], d_biases[1]);


		// POOLING2
		// dim3 BS4(1,1,50);
		// dim3 TPB4(4,4);
		// float* d_intermediate4;
		// cudaMalloc(&d_intermediate4, 50*4*4*sizeof(float));
		maxPooling3D<<<BS4,TPB4, 0, stream>>>(d_intermediates3[img], 8, 50, 2, d_intermediates4[img]);


		// FC1
		// dim3 BS5(1,1,500);
		// dim3 TPB5(1,1); 
		// float* d_intermediate5;
		// cudaMalloc(&d_intermediate5, 500*1*1*sizeof(float));
		convolutionWithoutPadding4D<<<BS5, TPB5, 0, stream>>>(d_intermediates4[img], 4, kernelInputChannels[2], d_kernels[2], kernelSizes[2], kernelOutputChannels[2], d_intermediates5[img], d_biases[2]);

		// APPLY RELU HERE 
		// dim3 BS6(1);
		// dim3 TPB6(500);
		// float* d_intermediate6;
		// cudaMalloc(&d_intermediate6, 500*sizeof(float));
		RELU<<<BS6, TPB6, 0, stream>>>(d_intermediates5[img], 500, d_intermediates6[img]); 

		// FC2
		// dim3 BS7(1,1,10);
		// dim3 TPB7(1);
		// float* d_intermediate7; 
		// cudaMalloc(&d_intermediate7, 10*1*1*sizeof(float));
		convolutionWithoutPadding4D<<<BS7, TPB7, 0, stream>>>(d_intermediates6[img], 1, kernelInputChannels[3], d_kernels[3], kernelSizes[3], kernelOutputChannels[3], d_intermediates7[img], d_biases[3]);

		// SOFTMAX
		// dim3 BS8(1);
		// dim3 TPB8(10); 
		// float* d_output; 
		// cudaMalloc(&d_output, 10*1*1*sizeof(float));
		softmaxKernel<<<BS8, TPB8, 0, stream>>>(d_intermediates7[img], d_outputs[img], 10); 

		cudaMemcpyAsync(output[img], d_outputs[img], 10*sizeof(float), cudaMemcpyDeviceToHost, stream);



	} 

	cudaDeviceSynchronize(); 

		// ##### Write top 10 probabilities to all output files.

	for (int img = 0; img < n ; img++){

		ofstream outFile(string(cwd)+"/output/"+filenames[img]);
		for (int i=0; i<10; i++){
			outFile << output[img][i] << " ";
		}
		outFile << endl;
		outFile.close();

	}



	// ##### Run LENET ######
	// // (float* input, int inputSize, int inputChannels, float* kernels, int kernelSize, int outputChannels, float* output, float bias)
	// // Pooling: maxPooling3D(float* input, int inputSize, int inputChannels, int poolingSize, float* output)

	// // cout << "Input before convolution" << endl; 

	// 	// float* start = images[0];
	// 	// for (int i = 0; i < 28; i++){
	// 	// 	for (int j = 0; j < 28; j++){
	// 	// 		cout << start[28*i + j] << " ";
	// 	// 	}
	// 	// 	cout << "\n"; 
	// 	// }


	// 	float* d_input;
	// 	cudaMalloc(&d_input, 784 * sizeof(float));
	// 	cudaMemcpy(d_input, images[img], 784 * sizeof(float), cudaMemcpyHostToDevice);

	// 	// CONV2D_LAYER1
	// 	dim3 BS1(1,1,20);
	// 	dim3 TPB1(24,24);
	// 	float* d_intermediate1;
	// 	cudaMalloc(&d_intermediate1, 20*24*24*sizeof(float));
	// 	convolutionWithoutPadding4D<<<BS1, TPB1>>>(d_input, 28, kernelInputChannels[0], d_kernels[0], kernelSizes[0],kernelOutputChannels[0], d_intermediate1, d_biases[0]);

	// 	// JUST TO PRINT

	// 	// cout << "Matrix after convolution1" << endl; 

	// 	// float* printer = (float*)malloc(24*24*sizeof(float)); 
	// 	// cudaMemcpy(printer, d_intermediate1, 24*24*sizeof(float), cudaMemcpyDeviceToHost);
	// 	// for (int i = 0; i < 24; i++){
	// 	// 	for (int j = 0; j < 24; j++){
	// 	// 		cout << printer[24*i + j] << " ";
	// 	// 	}
	// 	// 	cout << "\n"; 
	// 	// }

	// 	// POOLING1
	// 	dim3 BS2(1,1,20);
	// 	dim3 TPB2(12,12);
	// 	float* d_intermediate2;
	// 	cudaMalloc(&d_intermediate2, 20*12*12*sizeof(float));
	// 	maxPooling3D<<<BS2, TPB2>>>(d_intermediate1, 24, 20, 2, d_intermediate2);

	// 	// cout << "Matrix after Pooling1" << endl; 

	// 	// float* hello = (float*)malloc(12*12*sizeof(float));
	// 	// cudaMemcpy(hello, d_intermediate2, 12*12*sizeof(float), cudaMemcpyDeviceToHost);
	// 	// for (int i = 0; i < 12; i++){
	// 	// 	for (int j = 0; j < 12; j++){
	// 	// 		cout << hello[12*i + j] << " ";
	// 	// 	}
	// 	// 	cout << "\n"; 
	// 	// }


	// 	// CONV2D_LAYER2
	// 	dim3 BS3(1,1,50);
	// 	dim3 TPB3(8,8);
	// 	float* d_intermediate3;
	// 	cudaMalloc(&d_intermediate3, 50*8*8*sizeof(float));
	// 	convolutionWithoutPadding4D<<<BS3, TPB3>>>(d_intermediate2, 12, kernelInputChannels[1], d_kernels[1], kernelSizes[1], kernelOutputChannels[1], d_intermediate3, d_biases[1]);

	// 	// cout << "Matrix after convolution2" << endl;
	// 	// float* mega = (float*)malloc(8*8*sizeof(float));
	// 	// cudaMemcpy(mega, d_intermediate3, 8*8*sizeof(float), cudaMemcpyDeviceToHost);
	// 	// for (int i = 0; i<8 ; i++){
	// 	// 	for (int j = 0; j < 8; j ++){
	// 	// 		cout << mega[8*i+j] << " " ;
	// 	// 	}
	// 	// 	cout << "\n"; 
	// 	// }

	// 	// POOLING2
	// 	dim3 BS4(1,1,50);
	// 	dim3 TPB4(4,4);
	// 	float* d_intermediate4;
	// 	cudaMalloc(&d_intermediate4, 50*4*4*sizeof(float));
	// 	maxPooling3D<<<BS4,TPB4>>>(d_intermediate3, 8, 50, 2, d_intermediate4);

	// 	// cout << "Matrix after pooling2" << endl;
	// 	// float* mega1 = (float*)malloc(4*4*sizeof(float));
	// 	// cudaMemcpy(mega1, d_intermediate4, 4*4*sizeof(float), cudaMemcpyDeviceToHost);
	// 	// for (int i = 0; i<4 ; i++){
	// 	// 	for (int j = 0; j < 4; j ++){
	// 	// 		cout << mega1[4*i+j] << " " ;
	// 	// 	}
	// 	// 	cout << "\n"; 
	// 	// }


	// 	// FC1
	// 	dim3 BS5(1,1,500);
	// 	dim3 TPB5(1,1); 
	// 	float* d_intermediate5;
	// 	cudaMalloc(&d_intermediate5, 500*1*1*sizeof(float));
	// 	convolutionWithoutPadding4D<<<BS5, TPB5>>>(d_intermediate4, 4, kernelInputChannels[2], d_kernels[2], kernelSizes[2], kernelOutputChannels[2], d_intermediate5, d_biases[2]);

	// 	// cout << "Matrix after FC1" << endl;
	// 	// float* mega2 = (float*)malloc(500*sizeof(float));
	// 	// cudaMemcpy(mega2, d_intermediate5, 500*sizeof(float), cudaMemcpyDeviceToHost);
	// 	// for (int i = 0; i < 500; i++){
	// 	// 	cout << mega2[i] << endl; 
	// 	// }


	// 	// APPLY RELU HERE 
	// 	dim3 BS6(1);
	// 	dim3 TPB6(500);
	// 	float* d_intermediate6;
	// 	cudaMalloc(&d_intermediate6, 500*sizeof(float));
	// 	RELU<<<BS6, TPB6>>>(d_intermediate5, 500, d_intermediate6); 

	// 	// cout << "Matrix after RELU" << endl; 


	// 	// float* mega3 = (float*)malloc(500*sizeof(float));
	// 	// cudaMemcpy(mega3, d_intermediate6, 500*sizeof(float), cudaMemcpyDeviceToHost);
	// 	// for (int i = 0; i < 500; i++){
	// 	// 	cout << mega3[i] << endl; 
	// 	// }

	// 	// FC2
	// 	dim3 BS7(1,1,10);
	// 	dim3 TPB7(1);
	// 	float* d_intermediate7; 
	// 	cudaMalloc(&d_intermediate7, 10*1*1*sizeof(float));
	// 	convolutionWithoutPadding4D<<<BS7, TPB7>>>(d_intermediate6, 1, kernelInputChannels[3], d_kernels[3], kernelSizes[3], kernelOutputChannels[3], d_intermediate7, d_biases[3]);

	// 	// cout << "Matrix after FC2" << endl; 
	// 	// float* last = (float*)malloc(10*sizeof(float));
	// 	// cudaMemcpy(last, d_intermediate7, 10*sizeof(float), cudaMemcpyDeviceToHost);
	// 	// for (int i = 0; i < 10; i++){
	// 	// 	cout << last[i] << endl; 
	// 	// }

	// 	// cout << "DONE!" << endl; 

	// 	// SOFTMAX
	// 	dim3 BS8(1);
	// 	dim3 TPB8(10); 
	// 	float* d_output; 
	// 	cudaMalloc(&d_output, 10*1*1*sizeof(float));
	// 	softmaxKernel<<<BS8, TPB8>>>(d_intermediate7, d_output, 10); 

	// 	output[img] = (float*)malloc(10*sizeof(float));

	// 	cudaMemcpy(output[img], d_output, 10*sizeof(float), cudaMemcpyDeviceToHost);

	// 	cudaFree(d_input); 
	// 	cudaFree(d_intermediate1);
	// 	cudaFree(d_intermediate2);
	// 	cudaFree(d_intermediate3);
	// 	cudaFree(d_intermediate4);
	// 	cudaFree(d_intermediate5);
	// 	cudaFree(d_intermediate6); 
	// 	cudaFree(d_intermediate7); 
	// 	cudaFree(d_output);  

	// 	// ##### Write top 5 probabilities to output file.
	// 	for (int i=0; i<10; i++){
	// 		cout << output[img][i] << " ";
	// 	}
	// 	cout << endl;

		

	for (int l = 0; l<4; l++){
		cudaFree(&d_kernels[l]);
		cudaFree(&d_biases[l]); 
	}

 
}