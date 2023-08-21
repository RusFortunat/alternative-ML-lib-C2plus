// File with Tensor class constructor, destructor, and methods
// Authors: for now just me
#include <iostream>
#include <random>
#include <cmath>
#include "Tensor.h"

// A constructor, that generates the neural network structure; To create a network with 1 hidden layer, i need to initialize adjacency matrices and two bias vectors
Tensor::Tensor(int input_size, int hidden_size, int output_size){

    _input_size = input_size;
    _hidden_size = hidden_size;
    _output_size = output_size;

    // For a fully-connected network, the dimensions of W1 matrix woud be (hidden_size x input_size); here I assume that output vector h is obtained by (W * input vector), or h_j = \sum_{i=0}^{input_size} W_ji * input_i; similarly for W2; 
    _W1 (hidden_size, vector<double>(input_size));
    _W2 (output_size, vector<double>(hidden_size));
    // for biases vectors, their size is simply the size of the layers, for which the activations are computed, i.e.
    _B1(hidden_size, 0);
    _B2(output_size, 0);
    
    // Choosing the initial values of weights and biases can a work of art (as it turns out to be), and I will follow the guidline in this answer by Ashunigion user:
    // https://stackoverflow.com/a/55546528/16639275
    random_device rd{};
    mt19937 RNG{ rd() };
    // i will go with uniform distribution, that ranges from -(1/sqrt(input_size)):(1/sqrt(input_size))
    range_w1 = 1.0/sqrt(1.0*input_size);
    uniform_real_distribution<double> w1_weights{ -range_w1, range_w1 }; // just a dice from 0 to 1
    for(auto i=0; i < hidden_size; i++){
        for(auto j=0; j < input_size; j++){
            _W1[i][j] = w1_weights(RNG);
        }
    }
    range_w2 = 1.0/sqrt(1.0*hidden_size);
    uniform_real_distribution<double> w2_weights{ -range_w2, range_w2 }; // just a dice from 0 to 1
    for(auto i=0; i < output_size; i++){
        for(auto j=0; j < hidden_size; j++){
            _W2[i][j] = w2_weights(RNG);
        }
    }
}

// destructor
Tensor::~Tensor(){

    // since all objects of the class are vectors, I am not sure if I should explicitly delete them here or not...
}

// forward propagation method
vector<double> Tensor::forward(vector<double> &input_vector){
    // forwardprop is very simple really; just do the following:
    // 1. compute [z] = [W][input] + [biases]
    // 2. obtain the activations by applying to [z] some function f([z]). Here we will use ReLU activation f(x) = x if x > 0, or 0 if x < 0 

    vector<double> hidden_activations(_hidden_size);
    vector<double> output_activations(_output_size);
    
    // compute hidden layer activations
    for(auto i = 0; i < _hidden_size; i++){
        for(auto j = 0; j < _input_size; j++){
            double z = _W1[i][j]*input_vector[j] + _B1[i];
            hidden_activations[i] = z > 0 ? z : 0; // consize inline if-statement for RELU
        }
    }

    // compute output layer activations
    for(auto i = 0; i < _output_size; i++){
        for(auto j = 0; j < _hidden_size; j++){
            double z = _W2[i][j]*hidden_activations[j] + _B2[i];
            output_activations[i] = z > 0 ? z : 0; // consize inline if-statement for RELU
        }
    }

    return output_activations;
}