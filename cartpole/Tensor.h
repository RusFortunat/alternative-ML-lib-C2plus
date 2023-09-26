// File with Tensor class constructor, destructor, and methods
// Authors: 
#pragma once
#include <iostream>
#include <vector>
using namespace std;

class Tensor{
    public:
        // for now all tensors will have just a single hidden layer and random parameters
        Tensor(int input_size, int hidden_size, int output_size, double learning_rate, double gamma); // constructor
        ~Tensor(); // destructor

        vector<double> forward(vector<double> &input_vector); // forwardprop with ReLU activation
        
        int select_action(vector<double> state, double epsilon, int output_size); // e-greedy sampling

        // update network parameters (DQN stuff here)
        void optimizer_step(vector <tuple <vector<double>, int, vector<double>, double>> ReplayBuffer, int batch_size);
        
        // allows accessing network parameters
        tuple<vector<vector<double>>, vector<double>, vector<vector<double>>, vector<double>> get_model_parameters();

        void copy_parameters(tuple<vector<vector<double>>, vector<double>,
            vector<vector<double>>, vector<double>> net_params);

        void soft_update(tuple<vector<vector<double>>, vector<double>,
            vector<vector<double>>, vector<double>> net_params, double tau);

    private: 
        int _input_size;
        int _hidden_size;
        int _output_size;
        double _learning_rate;
        double _gamma;
        vector<vector<double>> _W1;
        vector<vector<double>> _W2;
        vector<double> _B1;
        vector<double> _B2;

        vector<double> _hidden_vector;
};