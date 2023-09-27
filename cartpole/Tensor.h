// File with Tensor class constructor, destructor, and methods
// Authors: 
#pragma once
#include <iostream>
#include <vector>
using namespace std;

class Tensor{
    public:
        // for now all tensors will have just a single hidden layer and random parameters
        Tensor(int input_size, int hidden_size, int output_size, double learning_rate); // constructor
        ~Tensor(); // destructor

        vector<double> forward(vector<double> &input_vector); // forwardprop with ReLU activation
        int select_action(vector<double>& state, double epsilon, int output_size); // e-greedy sampling
        
        void compute_gradients(vector<vector<double>>& w_gradients1, vector<double>& b_gradients1,
            vector<vector<double>>& w_gradients2, vector<double>& b_gradients2, int batch_size, int action, double loss);

        void optimizer_step(vector<vector<double>>& w_gradients1, vector<double>& b_gradients1,
            vector<vector<double>>& w_gradients2, vector<double>& b_gradients2);

        tuple<vector<vector<double>>, vector<double>, vector<vector<double>>, vector<double>> get_model_parameters();
        void copy_parameters(tuple<vector<vector<double>>, vector<double>, vector<vector<double>>, vector<double>> net_params);
        void soft_update(tuple<vector<vector<double>>, vector<double>, vector<vector<double>>, vector<double>> net_params,
            double tau);

    private: 
        int _input_size;
        int _hidden_size;
        int _output_size;
        double _learning_rate;
        vector<vector<double>> _W1;
        vector<vector<double>> _W2;
        vector<double> _B1;
        vector<double> _B2;

        vector<double> _input_vector;
        vector<double> _hidden_vector;
        vector<double> _predicted_vector;
};