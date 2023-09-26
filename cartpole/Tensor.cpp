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
Tensor::Tensor(int input_size, int hidden_size, int output_size, double learning_rate){

    _input_size = input_size;
    _hidden_size = hidden_size;
    _output_size = output_size;
    _learning_rate = learning_rate;

    // For a fully-connected network, the W1 matrix woud have (output_size x hidden_size) dimensions. 
    // Here I assume that z values are obtained by (W1 * input vector) + B1, or 
    // z_i = \sum_{i=0}^{input_size} W_ij * input_j + b_i. The activations are then obtained by f(z), where
    // f is a ReLU here -- activation_val = f(z) > 0 ? z : 0.
    _W1.resize(hidden_size, vector<double>(input_size));  // weights or adjacency matrix
    _W2.resize(output_size, vector<double>(hidden_size));  // weights or adjacency matrix
    _B1.resize(hidden_size); // biases; 0 by default
    _B2.resize(output_size); // biases; 0 by default
    _hidden_vector.resize(hidden_size);
        
    //random_device rd{}; //doesn't work with my MinGW compiler, gives the same number... should work with other compilers
    srand(std::time(0));
    const int random_seed2 = rand();
    mt19937 RNG2{ random_seed2 };
    //mt19937 RNG2{rd()};
    // I will go with the uniform distribution, that ranges from -(1/sqrt(input_size)):(1/sqrt(input_size))
    double range_w1 = 1.0/sqrt(1.0*input_size);
    uniform_real_distribution<double> w1_weights{ -range_w1, range_w1 };
    for(auto i = 0; i < hidden_size; i++){
        for(auto j = 0; j < input_size; j++){
            _W1[i][j] = w1_weights(RNG2);
        }
    }
    double range_w2 = 1.0/sqrt(1.0*hidden_size);
    uniform_real_distribution<double> w2_weights{ -range_w2, range_w2 };
    for(auto i = 0; i < output_size; i++){
        for(auto j = 0; j < hidden_size; j++){
            _W2[i][j] = w2_weights(RNG2);
        }
    }
}

// destructor
Tensor::~Tensor(){
    // since all objects of the class are vectors, I am not sure if I should explicitly delete them here or not...
}

// forward propagation method with ReLU; 
vector<double> Tensor::forward(vector<double> &input_vector){
    // forwardprop is very simple really; just do the following:
    // 1. compute [z] = [W1][input] + [biases1]
    // 2. obtain the activations [y] by applying to [z] some activation function f([z]). 
    // Here we will use ReLU activation: f(z) > 0 ? z : 0
    // 3. repeat for the next layers with W2 and B2
    
    vector<double> predicted(_output_size);

    // compute hidden activations
    for(auto i = 0; i < _hidden_size; i++){
        double sum = 0;
        for(auto j = 0; j < _input_size; j++){
            double activation = _W1[i][j] * input_vector[j] + _B1[i];
            if (activation > 0) sum += activation; // ReLU
        }
        _hidden_vector[i] = sum;
    }
    // compute output activations
    double total_sum = 0.0; // for softmax or normalization 
    for(auto i = 0; i < _output_size; i++){
        double sum = 0.0;
        for(auto j = 0; j < _hidden_size; j++){
            double activation = _W2[i][j] * _hidden_vector[j] + _B2[i];
            if (activation > 0) sum += activation; // ReLU
        }
        predicted[i] = sum;
        total_sum += exp(sum);
    }
    /*printf("predicted vector before softmax\n");
    for(auto i = 0; i < _output_size; i++){
        cout << predicted[i] << " ";
    }*/
    // do softmax
    for(auto i = 0; i < _output_size; i++){
        double element = exp(predicted[i]) / total_sum;
        predicted[i] = element;
    }

    return predicted; // we treat the NN output as Q values
}

// select action using epsilon-greedy policy
int Tensor::select_action(vector<double>& state, double epsilon, int output_size) {
    srand(std::time(0));
    const int random_seed2 = rand();
    mt19937 RNG3{ random_seed2 };
    uniform_real_distribution<double> dice{ 0, 1 };
    uniform_int_distribution<int> action_dice{ 0, output_size - 1 };
    double dice_throw = dice(RNG3);
    int action = -1;

    if (dice_throw > epsilon) { // greedy action
        vector<double> predicted_actions = forward(state);
        const int N = sizeof(predicted_actions) / sizeof(int);
        action = distance(predicted_actions, max_element(predicted_actions, predicted_actions + N)); // max action
    }
    else {
        action = action_dice(RNG3); // random action
    }

    return action;
}

// compute gradients here
void Tensor::compute_gradients(vector<double>& input_vector, 
    vector<double>& predicted_vector, 
    vector<double>& target_vector,
    vector<vector<double>>& w_gradients1, vector<double>& b_gradients1, 
    vector<vector<double>>& w_gradients2, vector<double>& b_gradients2, int batch_size) {
    // for ReLU activation:
    // d(loss_i)/dw_ij = 2*(y_i-t_i)*\theta(z_i)*\sum_j y_j^l-1, where loss_i = (y_i - t_i)^2,
    // y_i=f(z_i) is the output value, t_i is the target value, y_j^l-1 is the activation from the previous layer

    // W2 & B2
    for (auto i = 0; i < _output_size; i++) {
        // if the i-th activation of output layer y_i = 0, none of the w_ij will be updated due to ReLU activation
        if(predicted_vector[i] > 0){ // derivative of the relu is good 
            for (auto j = 0; j < _hidden_size; j++) {
                if( _hidden_vector[j] != 0){ // y_j^l-1; if the prev activation is zero -- the contribution to grad is zero
                    w_gradients2[i][j] += (1.0 / (1.0 * batch_size)) * 2 * 
                        (predicted_vector[i] - target_vector[i]) * _hidden_vector[j];
                }
            }
            b_gradients2[i] += (1.0 / (1.0 * batch_size)) * 2 * (predicted_vector[i] - target_vector[i]);
        }
    }
    // W1 & B1
    for (auto i = 0; i < _hidden_size; i++) {
        // if the i-th activation of output layer y_i = 0, none of the w_ij will be updated due to ReLU activation
        if(_hidden_vector[i] > 0){ 
            // compute dC/dy_j^l-1
            double dC_dy_prev = 0;
            for(auto k = 0; k < _output_size; k++){
                if(predicted_vector[k] > 0){ 
                    dC_dy_prev += 2 * (predicted_vector[k] - target_vector[k]) * _W2[k][i];
                }
            }

            for (auto j = 0; j < _input_size; j++) {
                if( input_vector[j] != 0){ // y_j^l-1; if the prev activation is zero -- the whole grad is zero
                    w_gradients1[i][j] += (1.0 / (1.0 * batch_size)) * dC_dy_prev * input_vector[j];
                }
            }
            b_gradients1[i] += (1.0 / (1.0 * batch_size)) * dC_dy_prev;
        }
    }
}

// update network parameters
void Tensor::optimizer_step(vector<vector<double>>& w_gradients1, vector<double>& b_gradients1,
    vector<vector<double>>& w_gradients2, vector<double>& b_gradients2){
    // W1 & B1
    for (auto i = 0; i < _hidden_size; i++) {
        for (auto j = 0; j < _input_size; j++) {
            _W1[i][j] = _W1[i][j] - _learning_rate * w_gradients1[i][j];
        }
        _B1[i] = _B1[i] - _learning_rate * b_gradients1[i];
    }
    // W2 & B2
    for (auto i = 0; i < _output_size; i++) {
        for (auto j = 0; j < _hidden_size; j++) {
            _W2[i][j] = _W2[i][j] - _learning_rate * w_gradients2[i][j];
        }
        _B2[i] = _B2[i] - _learning_rate * b_gradients2[i];
    }
}

tuple<vector<vector<double>>, vector<double>, vector<vector<double>>, vector<double>> Tensor::get_model_parameters(){

    return make_tuple(_W1, _B1, _W2, _B2);
}


void Tensor::copy_parameters() {


}

void Tensor::soft_update(tuple<vector<vector<double>>, vector<double>,
    vector<vector<double>>, vector<double>> net_params) {

}