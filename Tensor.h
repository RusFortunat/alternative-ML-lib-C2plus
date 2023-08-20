#pragma once
#include <iostream>

class Tensor{
    public:
        Tensor(int input_size, int hidden_size, int output_size); // for now just one hidden layer
        ~Tensor();

        void forward(); 


    private: 



}