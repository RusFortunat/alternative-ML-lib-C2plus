// Here I will use a fully-connected network with 1 hidden layer to classify images. 
// I will test my code on NIST database of handwritten number and letters.

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
    int hidden_size = 1;
    int output_size = 1;
    double learning_rate = 0.001;  
    
    int data_set_size = 15;
    int batch_size = 3;
    int training_episodes = 10;

    // define out network
    Tensor model(input_size, hidden_size, output_size, learning_rate);

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
            vector<vector<double>> w_gradients1(hidden_size, vector<double>(input_size)); 
            vector<vector<double>> w_gradients2(output_size, vector<double>(hidden_size)); 
            vector<double> b_gradients1(hidden_size);
            vector<double> b_gradients2(output_size);
            // create a mini-batch
            for (auto i = 0; i < batch_size; i++) {
                int index = element_index[minibatch * batch_size + i];
                vector<double> input = x[index];
                vector<double> target = y[index];

                vector<double> predicted = model.forward(input); // model prediction
                model.compute_gradients(input, predicted, target, 
                    w_gradients1, b_gradients1, w_gradients2, b_gradients2, batch_size); // accumulate gradients
                
                for (auto n = 0; n < predicted.size(); n++) {
                    loss += (1.0/(1.0*data_set_size))*(predicted[n] - target[n])*(predicted[n] - target[n]);
                }
            }

            // update the network parameters using accumulated gradients that 
            model.optimizer_step(w_gradients1, b_gradients1, w_gradients2, b_gradients2);
        }
        printf("Episode = %d; Loss = %.2f \n ", episode, loss);
    }

    // save data and parameters
    ofstream output_data("./output/random_data.txt");
    for(auto i = 0; i < data_set_size; i++){
        output_data << x[i][0] << "\t" << y[i][0] << "\n";
    }
    
    ofstream output_parameters("./output/random_data_network_parameters.txt");
    tuple<vector<vector<double>>, vector<double>, vector<vector<double>>, vector<double>> weights_and_biases = model.model_parameters();
    vector<vector<double>> weights1 = get<0>(weights_and_biases);
    vector<vector<double>> weights2 = get<2>(weights_and_biases);
    vector<double> biases1 = get<1>(weights_and_biases);
    vector<double> biases2 = get<3>(weights_and_biases);
    // i will be very crude here and use the knowledge that we only have 1 weight and 1 bias for the linear regression task
    output_parameters << "weights1 " << weights1[0][0] << "; biases1 " << biases1[0]
        << "; weights2 " << weights2[0][0] << "; biases2 " << biases2[0]; 

    return 0;
}