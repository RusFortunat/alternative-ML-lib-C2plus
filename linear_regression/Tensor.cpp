// File with Tensor class constructor, destructor, and methods
// Authors: 
#include <iostream>
#include <random>
#include <cmath>
#include <vector>
#include <ctime>
#include "Tensor.h"

using namespace std;

// A constructor, that generates the neural network structure; 
// To create a network with 1 hidden layer, i need to initialize 2 adjacency matrices and 2 bias vectors
Tensor::Tensor(int input_size, int output_size, double learning_rate){

    _input_size = input_size;
    _output_size = output_size;
    _learning_rate = learning_rate;

    // For a fully-connected network, the W matrix woud have (output_size x input_size) dimensions. 
    // Here I assume that output vector z is obtained by (W * input vector) + biases, or 
    // z_i = \sum_{i=0}^{input_size} W_ij * input_j + b_i. 
    _W.resize(output_size, vector<double>(input_size));  // weights or adjacency matrix
    _B.resize(output_size); // biases; 0 by default
        
    random_device rd{}; //doesn't work with my MinGW compiler, gives the same number... should work with other compilers
    //srand(std::time(0));
    //const int random_seed2 = rand();
    //mt19937 RNG2{ random_seed2 };
    mt19937 RNG2{rd()};
    // I will go with the uniform distribution, that ranges from -(1/sqrt(input_size)):(1/sqrt(input_size))
    double range_w = 1.0/sqrt(1.0*input_size);
    uniform_real_distribution<double> w1_weights{ -range_w, range_w };
    for(auto i = 0; i < output_size; i++){
        for(auto j = 0; j < input_size; j++){
            _W[i][j] = w1_weights(RNG2);
        }
    }
}

// destructor
Tensor::~Tensor(){
    // since all objects of the class are vectors, I am not sure if I should explicitly delete them here or not...
}

// forward propagation method with Linear
vector<double> Tensor::forward(vector<double> &input_vector){
    // forwardprop is very simple really; just do the following:
    // 1. compute [z] = [W][input] + [biases]
    // 2. obtain the activations [y] by applying to [z] some activation function f([z]). 
    // Here we will use Linear activation f(x) = x for all x
    vector<double> predicted(_output_size);

    for(auto i = 0; i < _output_size; i++){
        double sum = 0;
        for(auto j = 0; j < _input_size; j++){
            sum += _W[i][j]* input_vector[j] + _B[i];
        }
        predicted[i] = sum;
    }

    return predicted;
}

// compute gradients here
void Tensor::compute_gradients(vector<double>& input_vector, vector<double>& predicted_vector, vector<double>& target_vector,
    vector<vector<double>>& w_gradients, vector<double>& b_gradients, int batch_size) {
    
    // d(loss_i)/dw_ij = 2*(y_i-t_i)*\theta(z_i)*\sum_j y_j^l-1, where loss_i = (y_i - t_i)^2,
    // y_i=f(z_i) is the output value, t_i is the target value, y_j^l-1 is the activation from the previous layer

    for (auto i = 0; i < _output_size; i++) {
        // if the i-th activation of output layer y_i = 0, none of the w_ij will be updated due to ReLU activation
        for (auto j = 0; j < _input_size; j++) {
            w_gradients[i][j] += (1.0 / (1.0 * batch_size)) * 2 * (predicted_vector[i] - target_vector[i]) * input_vector[j];
        }
        b_gradients[i] += (1.0 / (1.0 * batch_size)) * 2 * (predicted_vector[i] - target_vector[i]);
    }
}

// update network parameters
void Tensor::optimizer_step(vector<vector<double>>& w_gradients, vector<double>& b_gradients){

    for (auto i = 0; i < _output_size; i++) {
        for (auto j = 0; j < _input_size; j++) {
            _W[i][j] = _W[i][j] - _learning_rate * w_gradients[i][j];
        }
        _B[i] = _B[i] - _learning_rate * b_gradients[i];
    }
}

tuple<vector<vector<double>>, vector<double>> Tensor::model_parameters(){

    return make_tuple(_W, _B);
}