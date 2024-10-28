#include <iostream>
#include <cmath>
#include <vector>
#include <iomanip>
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
float** createMatrix1(int n,int m) {
    float** matrix = new float*[n];
    for (int i = 0; i < n; ++i) {
        matrix[i] = new float[m];
    }
    float initialValue = 0.0f;
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < m; ++j) {
            matrix[i][j] = initialValue;
        }
    }

    return matrix;
}

// Convolution of a square input matrix and a square kernel without padding
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

void convolutionWithPadding(float* input_matrix, float* kernel, float* output_matrix, int input_size, int kernel_size, int output_size, int padding) {
   for(int row=0;row<output_size;row++){
    for(int col=0;col<output_size;col++){
   if (row < output_size && col < output_size) {
        int output_index = row * output_size + col;
        float sum = 0.0f;

        for (int i = 0; i < kernel_size; i++) {
            for (int j = 0; j < kernel_size; j++) {
                int input_row = row + i - padding;
                int input_col = col + j - padding;

                if (input_row >= 0 && input_row < input_size && input_col >= 0 && input_col < input_size) {
                    int input_index = input_row * input_size + input_col;
                    int weight_index = i * kernel_size + j;
                    sum += input_matrix[input_index] * kernel[i * kernel_size + j];
                }
            }
        }
        output_matrix[output_index] = sum;
    }
    }

   }
}


// Apply ReLU activation to each element of the matrix
void ReLU(float** matrix, int size, int m) {
    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < m; ++j) {
            matrix[i][j] = max(0.0f,matrix[i][j]);
        }
    }
    
}

// Apply Tanh activation to each element of the matrix
float** Tanh(float** matrix, int size,int m) {
    float** output = createMatrix(size);
    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < m; ++j) {
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

void softmax(vector<float> input, int size) {
    float maxVal = input[0];
    for (int i = 1; i < size; ++i) {
        maxVal = max(maxVal, input[i]);
    }

    float sum = 0;
    for (int i = 0; i < size; ++i) {
        input[i] = exp(input[i] - maxVal);
        sum += input[i];
    }

    for (int i = 0; i < size; ++i) {
        input[i] /= sum;
    }
}

// Sigmoid function for converting a vector of scores to probabilities
void sigmoid(vector<float> input, int size) {
    for (int i = 0; i < size; ++i) {
        input[i] = 1.0 / (1.0 + exp(-input[i]));
    }
}


// void printMatrix(float** matrix, int size) {
//     for (int i = 0; i < size; ++i) {
//         for (int j = 0; j < size; ++j) {
//             cout << setw(8) << matrix[i][j] << " ";
//         }
//         cout << endl;
//     }
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
        convolutionWithPadding(matrix, kernel, output, N,M,outputSize,P);
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
            ReLU(matrix, N,M);
            for(int i=0;i<N;i++){
            for(int j=0;j<M;j++){
                cout<<matrix[i][j]<<" ";
            }
            cout<<endl;
        }
        }
        else if(activationType==1){
            output = Tanh(matrix, N,M);
            for(int i=0;i<N;i++){
            for(int j=0;j<M;j++){
                cout<<output[i][j]<<" ";
            }
            cout<<endl;
        }
        }
        else{
            cout<<"Invalid activation type"<<endl;
            return 1;
        }
        
    } else if (task == 3 && argc >= 5) {
        // Subsampling task
        int pool = std::atoi(argv[2]);
        int pools = std::atoi(argv[3]);
        int N = std::atoi(argv[4]);
        float** matrix=createMatrix(N);
        int ct=5;
        for(int i=0;i<N;i++){
            for(int j=0;j<N;j++){
                matrix[i][j] = std::atof(argv[ct]);
                ct++;
            }
        }
        float** output;
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
        for(int i=0;i<N/pools;i++){
            for(int j=0;j<N/pools;j++){
                cout<<output[i][j]<<" ";
            }
            cout<<endl;
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