// File with Tensor class constructor, destructor, and methods
// Authors: 
#include <iostream>
#include <random>
#include <cmath>
#include <vector>
#include "Tensor.h"

using namespace std;

// A constructor, that generates the neural network structure; 
// To create a network with 1 hidden layer, i need to initialize 2 adjacency matrices and 2 bias vectors
Tensor::Tensor(int input_size, int hidden_size, int output_size, double learning_rate, int mode){

    _input_size = input_size;
    _hidden_size = hidden_size;
    _output_size = output_size;
    _learning_rate = learning_rate;
    _input_vector.resize(input_size); // not sure if this is needed; prolly yes, for backprop
    _hidden_vector.resize(hidden_size);
    _output_vector.resize(output_size);

    // For a fully-connected network, the dimensions of W1 matrix woud have (hidden_size x input_size) dimensions. 
    // Here I assume that output vector z is obtained by (W * input vector) + biases, or 
    // z_i = \sum_{i=0}^{input_size} W_ij * input_j + b_i. 
    if( mode == 0){ // RelU activation
        _W1.resize(hidden_size, vector<double>(input_size));  // weights or adjacency matrix
        _W2.resize(output_size, vector<double>(hidden_size)); // output_size x hidden_size
        _B1.resize(hidden_size); // biases; 0 by default
        _B2.resize(output_size);
        
        // Choosing the initial values of weights and biases can be a work of art, and I will follow the 
        // guidlines in this answer by Ashunigion user:  https://stackoverflow.com/a/55546528/16639275
        random_device rd{};
        mt19937 RNG{ rd() };
        // I will go with the uniform distribution, that ranges from -(1/sqrt(input_size)):(1/sqrt(input_size))
        double range_w1 = 1.0/sqrt(1.0*input_size);
        uniform_real_distribution<double> w1_weights{ -range_w1, range_w1 }; 
        for(auto i = 0; i < hidden_size; i++){
            for(auto j = 0; j < input_size; j++){
                _W1[i][j] = w1_weights(RNG);
            }
        }
        double range_w2 = 1.0/sqrt(1.0*hidden_size);
        uniform_real_distribution<double> w2_weights{ -range_w2, range_w2 }; 
        for(auto i = 0; i < output_size; i++){
            for(auto j = 0; j < hidden_size; j++){
                _W2[i][j] = w2_weights(RNG);
            }
        }
    }
    else{ // Linear activation only; no need for hidden layers
        _W1.resize(output_size, vector<double>(input_size));  // weights or adjacency matrix
        _B1.resize(output_size); // biases; 0 by default
        
        random_device rd{};
        mt19937 RNG{ rd() };
        // I will go with the uniform distribution, that ranges from -(1/sqrt(input_size)):(1/sqrt(input_size))
        double range_w1 = 1.0/sqrt(1.0*input_size);
        uniform_real_distribution<double> w1_weights{ -range_w1, range_w1 }; 
        for(auto i = 0; i < hidden_size; i++){
            for(auto j = 0; j < input_size; j++){
                _W1[i][j] = w1_weights(RNG);
            }
        }
    }
}

// destructor
Tensor::~Tensor(){
    // since all objects of the class are vectors, I am not sure if I should explicitly delete them here or not...
}

// forward propagation method with ReLU
void Tensor::forward_ReLU(vector<double> &input_vector){
    // forwardprop is very simple really; just do the following:
    // 1. compute [z] = [W][input] + [biases]
    // 2. obtain the activations [y] by applying to [z] some activation function f([z]). 
    // Here we will use ReLU activation f(x) = x if x > 0, or 0 if x < 0 

    _input_vector = input_vector;
    
    // compute hidden layer activations
    for(auto i = 0; i < _hidden_size; i++){
        double sum = 0;
        for(auto j = 0; j < _input_size; j++){
            double z = _W1[i][j]*_input_vector[j] + _B1[i];
            sum += z > 0 ? z : 0; // consize inline if-statement for RELU
        }
        _hidden_vector[i] = sum;
    }

    // compute output layer activations
    for(auto i = 0; i < _output_size; i++){
        double sum = 0;
        for(auto j = 0; j < _hidden_size; j++){
            double z = _W2[i][j]*_hidden_vector[j] + _B2[i];
            sum += z > 0 ? z : 0; // consize inline if-statement for RELU
        }
        _output_vector[i] = sum;
    }
}

// compute loss and do the backprop for ReLU
double Tensor::optimizer_ReLU(vector<double> &target_vector){
    // Now here comes the complicated part: we will compute loss here and then backpropagate 
    // it to update the model parameters. At the moment, I will use the knowledge I have about
    // how the derivatives of the loss are being computed, but maybe later some automatic differenntiation
    // package can be utilized.

    // compute the loss
    vector<double> loss(_output_size);
    for(auto i = 0; i < _output_size; i++){
        loss[i] = (_output_vector[i] - target_vector[i])*(_output_vector[i] - target_vector[i]);
    }

    // backpropagate the loss

    // d(loss_i)/dw_ij = 2*(y_i-t_i)*\theta(z_i)*\sum_j y_j^l-1, where loss_i = (y_i - t_i)^2,
    // y_i=f(z_i) is the output value, t_i is the target value, y_j^l-1 is the activation from the previous layer
    for(auto i = 0; i < _output_size; i++){
        // if the i-th activation of output layer y_i = 0, none of the w_ij will be updated due to ReLU activation
        if(_output_vector[i] != 0){ 
            for(auto j = 0; j < _hidden_size; j++){
                _W2[i][j] = _W2[i][j] - _learning_rate * 2 * (_output_vector[i] - target_vector[i]) * _hidden_vector[j];
            }
            _B2[i] = _B2[i] - _learning_rate * 2 * (_output_vector[i] - target_vector[i]);
        }
    }

    // updating W1 and B1 will be here below
    for(auto i = 0; i < _hidden_size; i++){
        // if the i-th activation of output layer y_i = 0, none of the w_ij will be updated due to ReLU activation
        if(_hidden_vector[i] != 0){ 
            // compute first d(loss_i)/dy_i^l-1
            double dC_dy_prev = 0;
            for(auto n = 0; n < _output_size; n++){
                if(_output_vector[n] != 0){
                    dC_dy_prev += 2 * (_output_vector[n] - target_vector[n]) * _W2[n][i];
                }
            }
            // update weights and biases
            for(auto j = 0; j < _input_size; j++){
                _W1[i][j] = _W1[i][j] - _learning_rate * dC_dy_prev * _input_vector[j];
            }
            _B1[i] = _B1[i] - _learning_rate * dC_dy_prev;
        }
    }

    double total_loss = accumulate(loss.begin(), loss.end(), 0);
    return total_loss;
}

// forward propagation method with Linear
void Tensor::forward_Linear(vector<double> &input_vector){
    // forwardprop is very simple really; just do the following:
    // 1. compute [z] = [W][input] + [biases]
    // 2. obtain the activations [y] by applying to [z] some activation function f([z]). 
    // Here we will use Linear activation f(x) = x for all x

    _input_vector = input_vector;
    // no need for hidden layer for linear activation
    for(auto i = 0; i < _output_size; i++){
        double sum = 0;
        for(auto j = 0; j < _input_size; j++){
            sum += _W1[i][j]*_input_vector[j] + _B1[i];
        }
        _output_vector[i] = sum;
    }
}

// compute loss and do the backprop for Linear activation
double Tensor::optimizer_ReLU(vector<double> &target_vector){
    // compute the loss
    vector<double> loss(_output_size);
    for(auto i = 0; i < _output_size; i++){
        loss[i] = (_output_vector[i] - target_vector[i])*(_output_vector[i] - target_vector[i]);
    }

    // backpropagate the loss

    // d(loss_i)/dw_ij = 2*(y_i-t_i)*\theta(z_i)*\sum_j y_j^l-1, where loss_i = (y_i - t_i)^2,
    // y_i=f(z_i) is the output value, t_i is the target value, y_j^l-1 is the activation from the previous layer
    for(auto i = 0; i < _output_size; i++){
        // if the i-th activation of output layer y_i = 0, none of the w_ij will be updated due to ReLU activation
        if(_output_vector[i] != 0){ 
            int sign = _output_vector[i] > 0 ? 1 : -1;
            for(auto j = 0; j < _hidden_size; j++){
                _W1[i][j] = _W1[i][j] - _learning_rate * 2 * sign * (_output_vector[i] - target_vector[i]) * _input_vector[j];
            }
            _B1[i] = _B1[i] - _learning_rate * 2 * sign * (_output_vector[i] - target_vector[i]);
        }
    }

    double total_loss = accumulate(loss.begin(), loss.end(), 0);
    return total_loss;
}