// File with Tensor class constructor, destructor, and methods
// Authors: 
#pragma once
#include <iostream>
#include <vector>
using namespace std;

class Tensor{
    public:
        // for now all tensors will have just a single layer
        Tensor(int input_size, int hidden_size, int output_size, double learning_rate, int mode); // constructor with random parameters
        ~Tensor();

        void forward_Linear(vector<double> &input_vector); // forwardprop with linear activation
        void forward_ReLU(vector<double> &input_vector); // forwardprop with ReLU activation
        //vector<double> loss(vector<double> &output_vector);
        double optimizer_Linear(vector<double> &target_vector); // not sure if computing loss should be a separate function... 
        double optimizer_ReLU(vector<double> &target_vector); // not sure if computing loss should be a separate function... 

        //void model_parameters(); // would be good to have this one too, to monitor changes

        vector<double> _output_vector;

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
};