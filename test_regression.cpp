// Here we perform the linear regression with our ML Tensor class.
#include <iostream>
#include <vector>
#include <random>
#include "Tensor.h"

using namespace std;

int main(){

    // NN parameters
    int data_set_size = 10;
    int input_size = data_set_size;
    int hidden_size = 2 * data_set_size;
    int output_size = data_set_size;
    double learning_rate = 0.001;
    int training_episodes = 20;

    // generate two data sets
    random_device rd{};
    mt19937 RNG{ rd() };
    uniform_real_distribution<double> data{ -5, 5 }; 
    vector<double> input_set(data_set_size);
    vector<double> target_set(data_set_size);
    for(auto i = 0; i < data_set_size; i++){
        double dice = data(RNG);
        input_set[i] = dice;
        dice = data(RNG);
        target_set[i] = dice;
    }
    printf("Input set: \n");
    for(auto i = 0; i < data_set_size; i++){
        cout << input_set[i] << "\t";
    }
    cout << endl;
    printf("Target set: \n");
    for(auto i = 0; i < data_set_size; i++){
        cout << target_set[i] << "\t";
    }
    cout << endl;

    Tensor model(input_size, hidden_size, output_size, learning_rate);

    for(auto episode = 0; episode < training_episodes; episode++){
        model.forward(input_set); // produces output vector
        double loss = model.optimizer(target_set); // compares NN output with target data set
        printf("Episode = %d; Loss = %.2f \n ", episode, loss);
    }

    return 0;
}