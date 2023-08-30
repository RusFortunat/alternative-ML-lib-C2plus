// Here we perform the linear regression test with our ML Tensor class. I'm sure this whole thing could have been
// done in a smarter way, but oh well. I need to preserve the whole matrix multiplication apparatus of Tensor forward 
// and backprop methods and at the same time test Stochastic Gradient Descent (SGD) for the linear regression problem.. 
// This is why I generate 2d vectors for a simple linear regression problem.

#include <iostream>
#include <vector>
#include <random>
#include <algorithm>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <iterator>
#include "Tensor.h"

using namespace std;

int main(){

    // NN parameters
    int input_size = 1;
    int output_size = 1;
    double learning_rate = 0.001;  
    
    int data_set_size = 15;
    int batch_size = 3;
    int training_episodes = 10;

    // define out network
    Tensor model(input_size, output_size, learning_rate);

    // RNG
    random_device rd{}; // doesn't work with my Windows MinGW C++ compiler... get the same number
    mt19937 RNG{ rd() }; 
    //srand(std::time(0));
    //const int random_seed = rand(); // 
    //cout << "seed: " << random_seed << endl;
    //mt19937 RNG{ random_seed }; // and this doesn't work on my Mac, what a joke
    uniform_real_distribution<double> data{ 0, 10 };

    // generate train and target data sets
    vector<vector<double>> x(data_set_size, vector<double>(input_size));
    vector<vector<double>> y(data_set_size, vector<double>(output_size));
    for (auto i = 0; i < data_set_size; i++) {
        for (auto j = 0; j < input_size; j++) {
            double dice = data(RNG);
            x[i][j] = dice;
            dice = data(RNG);
            y[i][j] = dice;
        }
    }

    // train the network
    printf("Train the network on a randomly generated data set:\n");
    for(auto episode = 0; episode < training_episodes; episode++){

        double loss = 0;
        // my implementation of random sampling of the elements of the input and target sets for the stochastic gradient descent (SGD) scheme
        vector<int> element_index(data_set_size);
        for (auto i = 0; i < data_set_size; i++) {
            element_index[i] = i;
        }
        shuffle(element_index.begin(), element_index.end(), std::default_random_engine(rd()));

        // split the data into N mini-batches
        for (auto minibatch = 0; minibatch < int(data_set_size / batch_size); minibatch++) {
            // accumulate computed gradient for this minibatch; for all weights and biases
            vector<vector<double>> w_gradients(output_size, vector<double>(input_size)); 
            vector<double> b_gradients(output_size);
            // create a mini-batch
            for (auto i = 0; i < batch_size; i++) {
                int index = element_index[minibatch * batch_size + i];
                vector<double> input = x[index];
                vector<double> target = y[index];

                vector<double> predicted = model.forward(input); // model prediction
                model.compute_gradients(input, predicted, target, w_gradients, b_gradients, batch_size); // accumulate gradients
                
                for (auto n = 0; n < predicted.size(); n++) {
                    loss += (1.0/(1.0*data_set_size))*(predicted[n] - target[n])*(predicted[n] - target[n]);
                }
            }

            // update the network parameters using accumulated gradients that 
            model.optimizer_step(w_gradients, b_gradients);
        }
        printf("Episode = %d; Loss = %.2f \n ", episode, loss);
    }

    // save data and parameters
    ofstream output_data("./output/random_data.txt");
    for(auto i = 0; i < data_set_size; i++){
        output_data << x[i][0] << "\t" << y[i][0] << "\n";
    }
    
    ofstream output_parameters("./output/random_data_network_parameters.txt");
    tuple<vector<vector<double>>, vector<double>> weights_and_biases = model.model_parameters();
    vector<vector<double>> weights = get<0>(weights_and_biases);
    vector<double> biases = get<1>(weights_and_biases);
    // i will be very crude here and use the knowledge that we only have 1 weight and 1 bias for the linear regression task
    output_parameters << "weight " << weights[0][0] << "; bias " << biases[0]; 



    // =========================================================================================================================
    // now let's try repeating the procedure on not-so-random target data; 
    // i will generate a noisy straight line and check if the network can learn the slope
    
    uniform_real_distribution<double> noise{ -1.0, 1.0 };
    double k = noise(RNG); // randomly chosen slope
    double b = noise(RNG); // randomly chosen shift
    for (auto i = 0; i < data_set_size; i++) {
        for (auto j = 0; j < input_size; j++) {
            double dice = data(RNG);
            x[i][j] = dice;
            double dice2 = noise(RNG);
            y[i][j] = k * dice + b + dice2; // i avoid explicetly mentioning x[i][j] here to not create the same memory reference for x[i][j] and y[i][j]
        }
    }

    Tensor model2(input_size, output_size, learning_rate);
    weights_and_biases = model2.model_parameters();
    vector<vector<double>> starting_weights = get<0>(weights_and_biases);
    vector<double> starting_biases = get<1>(weights_and_biases);

    printf("Train the network on a noisy straight line:\n");
    // train the network
    for(auto episode = 0; episode < training_episodes; episode++){

        double loss = 0;
        // my implementation of random sampling of the elements of the input and target sets for the stochastic gradient descent (SGD) scheme
        vector<int> element_index(data_set_size);
        for (auto i = 0; i < data_set_size; i++) {
            element_index[i] = i;
        }
        shuffle(element_index.begin(), element_index.end(), std::default_random_engine(rd()));

        // split the data into N mini-batches
        for (auto minibatch = 0; minibatch < int(data_set_size / batch_size); minibatch++) {
            // accumulate computed gradient for this minibatch; for all weights and biases
            vector<vector<double>> w_gradients(output_size, vector<double>(input_size)); 
            vector<double> b_gradients(output_size);
            // create a mini-batch
            for (auto i = 0; i < batch_size; i++) {
                int index = element_index[minibatch * batch_size + i];
                vector<double> input = x[index];
                vector<double> target = y[index];

                vector<double> predicted = model2.forward(input); // model prediction
                model2.compute_gradients(input, predicted, target, w_gradients, b_gradients, batch_size); // accumulate gradients
                
                for (auto n = 0; n < predicted.size(); n++) {
                    loss += (1.0/(1.0*data_set_size))*(predicted[n] - target[n])*(predicted[n] - target[n]);
                }
            }

            // update the network parameters using accumulated gradients that 
            model2.optimizer_step(w_gradients, b_gradients);
        }
        printf("Episode = %d; Loss = %.2f \n ", episode, loss);
    }

    // save data and parameters
    ofstream output_data2("./output/noisy_line_data.txt");
    for(auto i = 0; i < data_set_size; i++){
        output_data2 << x[i][0] << "\t" << y[i][0] << "\n";
    }
    
    ofstream output_parameters2("./output/noisy_line_network_parameters.txt");
    weights_and_biases = model2.model_parameters();
    weights = get<0>(weights_and_biases);
    biases = get<1>(weights_and_biases);
    // i will be very crude here and use the knowledge that we only have 1 weight and 1 bias for the linear regression task
    output_parameters2 << "True parameters: slope " << k << "; bias " << b << "\n"; 
    output_parameters2 << "Starting random parameters: weight " << starting_weights[0][0] << "; bias " << starting_biases[0] << "\n"; 
    output_parameters2 << "Parameters at the end of training: weight " << weights[0][0] << "; bias " << biases[0] << "\n"; 

    printf("End of simulation... \n");

    return 0;
}