#pragma once
#include <iostream>

class Tensor{
    public:
        // for now all tensors will have just a single layer
        Tensor(int input_size, int hidden_size, int output_size); // constructor with random parameters
        ~Tensor();

        void forward(); // ReLU activation only for now
        void loss();
        void optimizer();

        void model_parameters(); // would be good to have this one too, to monitor changes

    private: 
        vector<vector<double>> _W1;
        vector<vector<double>> _W2;
        vector<double> _B1;
        vector<double> _B2;
}