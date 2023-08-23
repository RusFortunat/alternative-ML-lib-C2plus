// File with Tensor class constructor, destructor, and methods
// Authors: 
#pragma once
#include <iostream>
#include <vector>
using namespace std;

class Tensor{
    public:
        // for now all tensors will have just a single layer
        Tensor(int input_size, int output_size, double learning_rate); // constructor with random parameters
        ~Tensor();

        vector<double> forward(vector<double> &input_vector); // forwardprop with linear activation
        void compute_gradients(vector<double>& input_vector, vector<double>& predicted_vector,
            vector<double>& target_vector, vector<vector<double>>& w_gradients, vector<double>& b_gradients, int batch_size); // compute gradients for backprop

        void optimizer_step(vector<double> &gradients); // update network parameters

    private: 
        int _input_size;
        int _output_size;
        double _learning_rate;
        vector<vector<double>> _W;
        vector<double> _B;
};