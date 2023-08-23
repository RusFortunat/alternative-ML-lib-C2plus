// Here we perform the linear regression with our ML Tensor class.
#include <iostream>
#include <vector>
#include <random>
#include "Tensor.h"

using namespace std;

int main(){

    // NN parameters
    int mode = 1; // 0 for ReLU, 1 for Linear activation; need for initialization
    int input_size = 1;
    int hidden_size = 0;
    int output_size = 1;
    double learning_rate = 0.0001;  
    
    int data_set_size = 100;
    int batch_size = 10;
    int training_episodes = 10;

    // generate two data sets
    vector<double> train_set(data_set_size);
    vector<double> target_set(data_set_size); // let's first set it as a straight line
    random_device rd{};
    mt19937 RNG{ rd() };
    uniform_real_distribution<double> data{ -5, 5 };
    for(auto i = 0; i < data_set_size; i++){
        double dice = data(RNG);
        train_set[i] = dice;
        dice = data(RNG);
        target_set[i] = dice;
    }
    printf("Input set: \n");
    for(auto i = 0; i < data_set_size; i++){ cout << train_set[i] << "\t"; }
    cout << endl;
    printf("Target set: \n");
    for(auto i = 0; i < data_set_size; i++){ cout << target_set[i] << "\t"; }
    cout << endl;

    // define out network
    Tensor model(input_size, hidden_size, output_size, learning_rate, mode);

    // train the network
    for(auto episode = 0; episode < training_episodes; episode++){
        // split the data into N mini-batches
        vector<double> copy_train_set = train_set;
        vector<double> copy_target_set = target_set;
        double loss = 0;
        for (auto minibatch = 0; minibatch < int(data_set_size / batch_size); minibatch++) {
            // create a mini-batch
            vector<double> input;
            vector<double> predicted;
            vector<double> target;
            for (auto i = 0; i < batch_size; i++) {
                int index = rand_between(0, copy_train_set.size() - i - 1);
                input.append(copy_train_set[index]);
                target.append(copy_target_set[index]);
                copy_train_set.erase(copy_train_set.begin() + index);
                copy_target_set.erase(copy_target_set.begin() + index);
            }

            // work with a minibatch
            for (auto i = 0; i < batch_size; i++) {
                model.forward_Linear(input[i]);
            }


            model.forward_Linear(input); // generates predicted values, that are accessible inside the class
            loss += (1.0 / 1.0* batch_size) * model.optimizer_Linear(predicted, target);
        }
        printf("Episode = %d; Loss = %.2f \n ", episode, loss);
    }

    printf("Output vector: \n");
    for(auto i = 0; i < output_size; i++){
        cout << model._output_vector[i] << "\t";
    }
    cout << endl;

    printf("End of simulation... \n");

    return 0;
}