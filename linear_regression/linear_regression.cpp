// Here we perform the linear regression test with our ML Tensor class. I'm sure this whole thing could have been
// done in a smarter way, but oh well. I need to preserve the whole matrix multiplication apparatus of Tensor forward 
// and backprop methods and at the same time test Stochastic Gradient Descent (SGD) for the linear regression problem.. 
#include <iostream>
#include <vector>
#include <random>
#include <algorithm>
#include <cstdlib>
#include <ctime>
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

    // define out network
    Tensor model(input_size, output_size, learning_rate);

    // RNG
    //random_device rd{}; // doesn't work with my MinGW C++ compiler... 
    //mt19937 RNG{ rd() }; 
    srand(std::time(0));
    int random_seed = rand();
    cout << "seed: " << random_seed << endl;
    mt19937 RNG{ random_seed };
    uniform_real_distribution<double> data{ -5, 5 };

    // train the network
    for(auto episode = 0; episode < training_episodes; episode++){
        // generate two data sets, input and target ones;
        vector<vector<double>> train_set(data_set_size, vector<double>(input_size));
        vector<vector<double>> target_set(data_set_size, vector<double>(output_size));
        for (auto i = 0; i < data_set_size; i++) {
            for (auto j = 0; j < input_size; j++) {
                double dice = data(RNG);
                train_set[i][j] = dice;
            }
            for (auto j = 0; j < output_size; j++) {
                double dice = data(RNG);
                target_set[i][j] = dice;
            }
        }

        double loss = 0;
        // determine how the elements of the training and target set will be sampled for the SGD scheme
        vector<int> element_index(data_set_size);
        for (auto i = 0; i < data_set_size; i++) {
            element_index[i] = i;
        }
        shuffle(element_index.begin(), element_index.end(), std::default_random_engine(random_seed));

        // split the data into N mini-batches
        for (auto minibatch = 0; minibatch < int(data_set_size / batch_size) - 1; minibatch++) {
            // create a mini-batch
            vector<double> input(input_size);
            vector<double> predicted(output_size);
            vector<double> target(output_size);
            vector<vector<double>> w_gradients(output_size, vector<double>(input_size)); // accumulate computed gradient for this minibatch; for all weights and biases
            vector<double> b_gradients(output_size);
            for (auto i = 0; i < batch_size; i++) {
                int id = minibatch * batch_size + i;
                int index = element_index[id];
                input = train_set[index];
                target = target_set[index];

                predicted = model.forward(input);
                model.compute_gradients(input, predicted, target, w_gradients, b_gradients, batch_size);
                
                for (auto n = 0; n < predicted.size(); n++) {
                    loss += (predicted[n] - target[n])*(predicted[n] - target[n]);
                }
            }

            // update network parameters
            model.optimizer_step(w_gradients, b_gradients);
        }

        printf("Episode = %d; Loss = %.2f \n ", episode, loss);
    }
    printf("Training ended. \n\n ");

    // test network's performance
    printf("Let's test the network's performance now: \n\n ");
    vector<vector<double>> test_set(data_set_size, vector<double>(input_size));
    vector<vector<double>> target_set(data_set_size, vector<double>(output_size));
    for (auto i = 0; i < data_set_size; i++) {
        for (auto j = 0; j < input_size; j++) {
            double dice = data(RNG);
            test_set[i][j] = dice;
        }
        for (auto j = 0; j < output_size; j++) {
            double dice = data(RNG);
            target_set[i][j] = dice;
        }
    }

    double loss = 0;
    for (auto i = 0; i < data_set_size; i++) {
        vector<double> predicted(output_size);
        vector<double> input(input_size);
        input = test_set[i];
        predicted = model.forward(input);
        for (auto j = 0; j < output_size; j++) {
            loss += (predicted[j] - target_set[i][j]) * (predicted[j] - target_set[i][j]);
        }
    }
    printf("Total loss after training: %f\n", loss);
    printf("Data is saved");

    printf("End of simulation... \n");

    return 0;
}