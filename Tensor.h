#pragma once
#include <iostream>

class Tensor{
    public:
        // for now all tensors will have just a single layer
        Tensor(int input_size, int hidden_size, int output_size); // constructor with random parameters
        ~Tensor();

        vector<double> forward(vector<double> &input_vector); // forwardprop with ReLU activation
        vector<double> loss(vector<double> &output_vector);
        void optimizer(); // not sure if computing loss should be a separate function... 

        void model_parameters(); // would be good to have this one too, to monitor changes

    private: 
        int _input_size;
        int _hidden_size;
        int _output_size;
        vector<vector<double>> _W1;
        vector<vector<double>> _W2;
        vector<double> _B1;
        vector<double> _B2;
}