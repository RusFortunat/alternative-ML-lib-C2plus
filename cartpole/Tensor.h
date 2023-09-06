// File with Tensor class constructor, destructor, and methods
// Authors: 
#pragma once
#include <iostream>
#include <vector>
using namespace std;

class Tensor{
    public:
        // for now all tensors will have just a single hidden layer
        Tensor(int input_size, int hidden_size, 
        int output_size, double learning_rate); // constructor with random parameters
        
        ~Tensor();

        vector<double> forward(vector<double> &input_vector); // forwardprop with ReLU activation
        
        // compute gradients for backprop
        void compute_gradients(vector<double>& input_vector, 
            vector<double>& predicted_vector, 
            vector<double>& target_vector,
            vector<vector<double>>& w_gradients1, vector<double>& b_gradients1, 
            vector<vector<double>>& w_gradients2, vector<double>& b_gradients2, int batch_size); 

        // update network parameters
        void optimizer_step(vector<vector<double>>& w_gradients1, vector<double>& b_gradients1,
            vector<vector<double>>& w_gradients2, vector<double>& b_gradients2); 
        
        // allows accessing network parameters
        tuple<vector<vector<double>>, vector<double>, 
            vector<vector<double>>, vector<double>> model_parameters();

    private: 
        int _input_size;
        int _hidden_size;
        int _output_size;
        double _learning_rate;
        vector<vector<double>> _W1;
        vector<vector<double>> _W2;
        vector<double> _B1;
        vector<double> _B2;

        vector<double> _hidden_vector;
};