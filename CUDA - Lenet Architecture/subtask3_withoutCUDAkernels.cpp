#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <algorithm>
// #include "s1.h"
using namespace std;

float** createMatrix(int n) {
    float** matrix = new float*[n];
    for (int i = 0; i < n; ++i) {
        matrix[i] = new float[n];
    }
    float initialValue = 0.0f;
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            matrix[i][j] = initialValue;
        }
    }

    return matrix;
}
void convolutionWithoutPadding(float** input, int inputSize, float** kernel, int kernelSize,float** output,int bias) {
    int outputSize = inputSize - kernelSize + 1;

    for (int i = 0; i < outputSize; ++i) {
        for (int j = 0; j < outputSize; ++j) {
            for (int k = 0; k < kernelSize; ++k) {
                for (int l = 0; l < kernelSize; ++l) {
                    output[i][j] += input[i + k][j + l] * kernel[k][l]+bias;
                }
            }
        }
    }
}

// Convolution of a square input matrix and a square kernel with padding
float** convolutionWithPadding(float** input, int inputSize, float** kernel, int kernelSize) {
    int padding = (kernelSize - 1) / 2;
    int paddedSize = inputSize + 2 * padding;

    float ** output = createMatrix(paddedSize); 

    for (int i = 0; i < inputSize; ++i) {
        for (int j = 0; j < inputSize; ++j) {
            output[i][j] = 0;
            for (int k = 0; k < kernelSize; ++k) {
                for (int l = 0; l < kernelSize; ++l) {
                    int row = i + k - padding;
                    int col = j + l - padding;
                    if (row >= 0 && row < inputSize && col >= 0 && col < inputSize) {
                        output[i][j] += input[row][col] * kernel[k][l];
                    }
                }
            }
        }
    }
    return output;
}


// Apply ReLU activation to each element of the matrix
float** ReLU(float** matrix, int size) {
    float** output = createMatrix(size);
    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
            output[i][j] = max(0.0f,matrix[i][j]);
        }
    }
    return output;
}

// Apply Tanh activation to each element of the matrix
float** Tanh(float** matrix, int size) {
    float** output = createMatrix(size);
    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
            output[i][j] = tanh(matrix[i][j]);
        }
    }
    return output;
}

// Subsampling using Max Pooling
float** maxPooling(float** input, int inputSize, int poolingSize) {
    int outputSize = inputSize / poolingSize;
    float**  output = createMatrix(outputSize);

    for (int i = 0; i < outputSize; ++i) {
        for (int j = 0; j < outputSize; ++j) {
            float maxVal = input[i * poolingSize][j * poolingSize];
            for (int k = 0; k < poolingSize; ++k) {
                for (int l = 0; l < poolingSize; ++l) {
                    maxVal = max(maxVal, input[i * poolingSize + k][j * poolingSize + l]);
                }
            }
            output[i][j] = maxVal;
        }
    }
    return output;
}

// Subsampling using Average Pooling
float** averagePooling(float** input, int inputSize, int poolingSize) {
    int outputSize = inputSize / poolingSize;
    float** output = createMatrix(outputSize);

    for (int i = 0; i < outputSize; ++i) {
        for (int j = 0; j < outputSize; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < poolingSize; ++k) {
                for (int l = 0; l < poolingSize; ++l) {
                    sum += input[i * poolingSize + k][j * poolingSize + l];
                }
            }
            output[i][j] = sum / (poolingSize * poolingSize);
        }
    }
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
// float** readImage(string filename){
//     float** inputImage = new float*[28];
    
//     ifstream file(filename);
//     if(file.is_open()){
//         for(int i = 0; i < 28; i++){
//             inputImage[i] = new float[28];
//             for(int j = 0; j < 28; j++){
//                 file >> inputImage[i][j];
//             }
//         }
//         file.close();
//     }
//     return inputImage;
// }
float*** conv2d_layer1(float** inputImage, float* conv1Weights){
    // heck1";
    float* bias = new float[20];
    for(int i = 0; i < 20; i++){
        bias[i] =  conv1Weights[520 + i];
    }
    // heck2";
    float*** output = new float**[20];
    for(int i=0;i<20;i++){
        output[i]=createMatrix(24);
    }
    // heck3";
    float** kernel = new float*[5];
    for(int i=0;i<5;i++){
        kernel[i] = new float[5];
    }
    int offset=0;
    // heck4";
    for(int i=0;i<20;i++){
        for(int k=0;k<5;k++){
            for(int l=0;l<5;l++){
                kernel[k][l] = conv1Weights[offset];
                offset++;
            }
        }
        // nter"<<i<<endl;
        convolutionWithoutPadding(inputImage,28,kernel,5,output[i],bias[i]);   
        // xit"<<i<<endl;
    }

    return output;
}
float*** pooling_layer(float*** inputImage,int numFilters,int inputSize,int stride){
    float*** output = new float**[numFilters];
    for(int i=0;i<numFilters;i++){
        output[i] = maxPooling(inputImage[i],inputSize,stride);
    }
    return output;
}
float*** conv2d_layer2(float*** inputImage, float* conv2Weights){
    float* bias = new float[50];
    for(int i = 0; i < 50; i++){
        bias[i] =  conv2Weights[25000 + i];
    }
    float*** output = new float**[50];
    for(int i=0;i<50;i++){
        output[i]=createMatrix(8);
    }
    float** kernel = new float*[5];
    for(int i=0;i<5;i++){
        kernel[i] = new float[5];
    }
    int offset=0;
    
    for(int i=0;i<50;i++){
        for(int j=0;j<20;j++){
            for(int k=0;k<5;k++){
                for(int l=0;l<5;l++){
                    kernel[k][l] = conv2Weights[offset];
                    offset++;
                }
            }
            
            convolutionWithoutPadding(inputImage[j],12,kernel,5,output[i],bias[i]);
        }   
    }
    return output;
}
float*** fc_layer1(float*** inputImage, float* fc1Weights){
    float* bias = new float[500];
    for(int i = 0; i < 500; i++){
        bias[i] =  fc1Weights[400000 + i];
    }
    float*** output = new float**[500];
    for(int i=0;i<500;i++){
        output[i]=createMatrix(1);
    }
    float** kernel = new float*[4];
    for(int i=0;i<4;i++){
        kernel[i] = new float[4];
    }
    int offset=0;
    for(int i=0;i<500;i++){
        for(int j=0;j<50;j++){
            for(int k=0;k<4;k++){
                for(int l=0;l<4;l++){
                    kernel[k][l] = fc1Weights[offset];
                    offset++;
                }
            }
            convolutionWithoutPadding(inputImage[j],4,kernel,4,output[i],bias[i]);
        }  
        ReLU(output[i],1);
    }
    
    return output;
}
float*** fc_layer2(float*** inputImage, float* fc2Weights){
    float* bias = new float[10];
    for(int i = 0; i < 10; i++){
        bias[i] =  fc2Weights[5000 + i];
    }
    float*** output = new float**[10];
    for(int i=0;i<10;i++){
        output[i]=createMatrix(1);
    }
    float** kernel = new float*[1];
    for(int i=0;i<1;i++){
        kernel[i] = new float[1];
    }
    int offset=0;
    for(int i=0;i<10;i++){
        for(int j=0;j<500;j++){
            for(int k=0;k<1;k++){
                for(int l=0;l<1;l++){
                    kernel[k][l] = fc2Weights[offset];
                    offset++;
                }
            }
            convolutionWithoutPadding(inputImage[j],1,kernel,1,output[i],bias[i]);
        }
    }    
    return output;
}
// void PrintMatrix(float*** matrix,int num, int size){
//     ofstream file("3d.txt");
//     for(int i=0;i<num;i++){
//         for(int j=0;j<size;j++){
//             for(int k=0;k<size;k++){
//                 file<<matrix[i][j][k]<<" ";
//             }
//             file<<endl;
//         }
//         file<<endl;
//     }
//     file.close();
// }
// void Print2dMatrix(float** matrix,int size){
//     ofstream file("2d.txt");
//     for(int i=0;i<size;i++){
//         for(int j=0;j<size;j++){
//             file<<matrix[i][j]<<" ";
//         }
//         file<<endl;
        
//     }
//     file.close();
// }
int main(){
    int inputSize = 28;

    float* image = loadWeights("0001.txt",784);
    float** inputImage = new float*[28];
    for(int i=0;i<28;i++){
        inputImage[i] = new float[28];
        for(int j=0;j<28;j++){
            inputImage[i][j] = image[i*28+j];
        }
    }
    
    float* conv1Weights = loadWeights("conv1.txt",520);
    
    float* conv2Weights = loadWeights("conv2.txt",25050);
    
    float* fc1Weights = loadWeights("fc1.txt",400500);
    
    float* fc2Weights = loadWeights("fc2.txt",5010);
    

    // Use details.txt to get the details of all layers
    // Print2dMatrix(inputImage,28);
    float*** output = conv2d_layer1(inputImage,conv1Weights); 
      
    float*** output2 = pooling_layer(output,20,24,2);    
    float*** output3 = conv2d_layer2(output2,conv2Weights);    
    float*** output4 = pooling_layer(output3,50,8,2);    
    // PrintMatrix(output4,50,4);
    
    float*** output5 = fc_layer1(output4,fc1Weights);
    
    float*** final_output = fc_layer2(output5,fc2Weights);
    float* pdp = new float[10];
    for(int i=0;i<10;i++){
        pdp[i] = final_output[i][0][0];
    }
    
    softmax(pdp,10);
    for(int i=0;i<10;i++){
        cout<<pdp[i]<<endl;
    }

}