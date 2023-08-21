// Here we perform the linear regression with our ML Tensor class.
#include <iostream>
#include <vector>
#include <ctime>
#include <algorithm>
#include "Tensor.h"

using namespace std;

int main(){

    // NN parameters
    int data_set_size = 10;
    int input_size = data_set_size;
    int hidden_size = 2 * data_set_size;
    int output_size = data_set_size;
    double learning_rate = 0.001;
    int training_episodes = 10;

    // generate two data sets
    vector<double> input_set(data_set_size);
    vector<double> target_set(data_set_size);
    srand(unsigned(std::time(nullptr)));
    generate(input_set.begin(), input_set.end(), rand);
    generate(target_set.begin(), target_set.end(), rand);

    Tensor model(input_size, hidden_size, output_size, learning_rate);

    for(auto episode = 0; episode < training_episodes; episode++){
        model.forward(input_set); // produces output vector
        loss = model.optimize(target_set); // compares NN output with target data set
        printf("Loss = %.2f \n ", loss);
    }

    return 0;
}