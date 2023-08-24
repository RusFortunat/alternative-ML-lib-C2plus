// Here we perform the linear regression test with our ML Tensor class. I'm sure this whole thing could have been
// done in a smarter way, but oh well. I need to preserve the whole matrix multiplication apparatus of Tensor forward 
// and backprop methods and at the same time test Stochastic Gradient Descent (SGD) for the linear regression problem.. 
#include <iostream>
#include <vector>
#include <random>
#include <algorithm>
#include <cstdlib>
#include "Tensor.h"

using namespace std;

int main(){

    // NN parameters
    int input_size = 1;
    int output_size = 1;
    double learning_rate = 0.001;  
    
    int data_set_size = 20;
    int batch_size = 5;
    int training_episodes = 10;

    // generate two data sets; 
    vector<vector<double>> train_set(data_set_size, vector<double>(input_size));
    vector<vector<double>> target_set(data_set_size, vector<double>(output_size));
    random_device rd{};
    mt19937 RNG{ rd() };
    uniform_real_distribution<double> data{ -5, 5 };
    for(auto i = 0; i < data_set_size; i++){
        for (auto j = 0; j < input_size; j++) {
            double dice = data(RNG);
            train_set[i][j] = dice;
        }
        for (auto j = 0; j < output_size; j++) {
            double dice = data(RNG);
            target_set[i][j] = dice;
        }
    }
    printf("Input set: \n");
    for (auto i = 0; i < data_set_size; i++) {
        for (auto j = 0; j < input_size; j++) {
            cout << train_set[i][j] << " ";
        }
        cout << " ";
    }
    cout << endl;
    printf("Target set: \n");
    for (auto i = 0; i < data_set_size; i++) {
        for (auto j = 0; j < input_size; j++) {
            cout << target_set[i][j] << " ";
        }
        cout << " ";
    }
    cout << endl;

    // define out network
    Tensor model(input_size, output_size, learning_rate);
    printf("Model initialized \n");

    // train the network
    for(auto episode = 0; episode < training_episodes; episode++){
        // split the data into N mini-batches
        vector<vector<double>> copy_train_set = train_set;
        vector<vector<double>> copy_target_set = target_set;
        double loss = 0;
        for (auto minibatch = 0; minibatch < int(data_set_size / batch_size) - 1; minibatch++) {
            printf("Minibatch number: %d \n", minibatch);
            // create a mini-batch
            vector<double> input(input_size);
            vector<double> predicted(output_size);
            vector<double> target(output_size);
            vector<vector<double>> w_gradients(input_size, vector<double>(output_size)); // accumulate computed gradient for this minibatch; for all weights and biases
            vector<double> b_gradients(output_size);
            double loss = 0;
            printf("Initialized vectors for a minibatch \n");
            for (auto i = 0; i < batch_size; i++) {
                int vec_size = copy_train_set.size()-1;
                uniform_int_distribution<int> element{ 0, vec_size};
                int index = element(RNG);
                input = copy_train_set[index];
                target = copy_target_set[index];

                predicted = model.forward(input);
                model.compute_gradients(input, predicted, target, w_gradients, b_gradients, batch_size);
                printf("forwardprop is done, gradients computed  \n");
                
                for (auto n = 0; n < predicted.size(); n++) {
                    loss += (predicted[n] - target[n])*(predicted[n] - target[n]);
                }

                iter_swap(copy_train_set.begin() + index, copy_train_set.end());
                iter_swap(copy_target_set.begin() + index, copy_target_set.end());
                copy_train_set.erase(copy_train_set.end());
                copy_target_set.erase(copy_target_set.end());
                printf("vector resized  \n");
            }
            printf("minibatch loop is over \n");

            // update network parameters
            model.optimizer_step(w_gradients, b_gradients);
            printf("network parameters are updated  \n");
        }
        printf("one episode is over  \n");

        printf("Episode = %d; Loss = %.2f \n ", episode, loss);
    }
    printf("Training ended. \n\n ");

    // test network's performance
    printf("Let's test the network's performance now: \n\n ");
    vector<vector<double>> test_set(data_set_size, vector<double>(input_size));
    for (auto i = 0; i < data_set_size; i++) {
        for (auto j = 0; j < input_size; j++) {
            double dice = data(RNG);
            test_set[i][j] = dice;
        }
        for (auto j = 0; j < output_size; j++) { // reinitialize the target data set
            double dice = data(RNG);
            target_set[i][j] = dice;
        }
    }
    int error_rate = 0;
    double epsilon = 0.0001; // if the discrepancy between predicted and target values is greater than this number, count as error
    for (auto i = 0; i < data_set_size; i++) {
        vector<double> predicted(output_size);
        vector<double> input(input_size);
        input = test_set[i];
        predicted = model.forward(input);
        for (auto j = 0; j < output_size; j++) {
            if (abs(predicted[j] - target_set[i][j]) > epsilon) error_rate++;
        }
    }
    double correct_percent = (1.0 - (1.0 * error_rate / (1.0*output_size* data_set_size)))*100;
    printf("Correctly predicted: %.2f % \n\n ", correct_percent);

    printf("End of simulation... \n");

    return 0;
}