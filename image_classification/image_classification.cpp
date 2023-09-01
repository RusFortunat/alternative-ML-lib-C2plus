// Here I will use a fully-connected network with 1 hidden layer to classify images. 
// I will test my code on MNIST database of handwritten number and letters.
// Download the MNIST handwritten images dataset from here: http://www.pymvpa.org/datadb/mnist.html

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

// load the dataset and the labels; I copied these functions from here: https://stackoverflow.com/a/33384846 
unsigned char** read_mnist_images(string full_path, int& number_of_images, int& image_size) {
    auto reverseInt = [](int i) {
        unsigned char c1, c2, c3, c4;
        c1 = i & 255, c2 = (i >> 8) & 255, c3 = (i >> 16) & 255, c4 = (i >> 24) & 255;
        return ((int)c1 << 24) + ((int)c2 << 16) + ((int)c3 << 8) + c4;
    };

    typedef unsigned char uchar;

    ifstream file(full_path, ios::binary);

    if (file.is_open()) {
        int magic_number = 0, n_rows = 0, n_cols = 0;

        file.read((char*)&magic_number, sizeof(magic_number));
        magic_number = reverseInt(magic_number);

        if (magic_number != 2051) throw runtime_error("Invalid MNIST image file!");

        file.read((char*)&number_of_images, sizeof(number_of_images)), number_of_images = reverseInt(number_of_images);
        file.read((char*)&n_rows, sizeof(n_rows)), n_rows = reverseInt(n_rows);
        file.read((char*)&n_cols, sizeof(n_cols)), n_cols = reverseInt(n_cols);

        image_size = n_rows * n_cols;

        uchar** _dataset = new uchar * [number_of_images];
        for (int i = 0; i < number_of_images; i++) {
            _dataset[i] = new uchar[image_size];
            file.read((char*)_dataset[i], image_size);
        }
        return _dataset;
    }
    else {
        throw runtime_error("Cannot open file `" + full_path + "`!");
    }
}

unsigned char* read_mnist_labels(string full_path, int& number_of_labels) {
    auto reverseInt = [](int i) {
        unsigned char c1, c2, c3, c4;
        c1 = i & 255, c2 = (i >> 8) & 255, c3 = (i >> 16) & 255, c4 = (i >> 24) & 255;
        return ((int)c1 << 24) + ((int)c2 << 16) + ((int)c3 << 8) + c4;
    };

    typedef unsigned char uchar;

    ifstream file(full_path, ios::binary);

    if (file.is_open()) {
        int magic_number = 0;
        file.read((char*)&magic_number, sizeof(magic_number));
        magic_number = reverseInt(magic_number);

        if (magic_number != 2049) throw runtime_error("Invalid MNIST label file!");

        file.read((char*)&number_of_labels, sizeof(number_of_labels)), number_of_labels = reverseInt(number_of_labels);

        uchar* _dataset = new uchar[number_of_labels];
        for (int i = 0; i < number_of_labels; i++) {
            file.read((char*)&_dataset[i], 1);
        }
        return _dataset;
    }
    else {
        throw runtime_error("Unable to open file `" + full_path + "`!");
    }
}

int main(){

    // NN parameters
    int input_size = 28 * 28; // image size
    int hidden_size = 128;
    int output_size = 10; // number of labels
    double learning_rate = 0.001;  
    int data_set_size = 200; // number of images
    int batch_size = 20;
    int training_episodes = 5;

    // load images
    string train_path_images = "";
    string train_path_labels = "";
    int data_set_size = 100;
    unsigned char** train_dataset = read_mnist_images(train_path_images, data_set_size, input_size); // input_size = image size
    unsigned char* train_labels = read_mnist_labels(train_path_labels, output_size); // output_size = numer of labels

    // define out network
    Tensor model(input_size, hidden_size, output_size, learning_rate);

    // train the network
    printf("Train the network on a randomly generated data set:\n");
    for(auto episode = 0; episode < training_episodes; episode++){

        double loss = 0;
        // my implementation of random sampling of the elements of the input and target sets 
        // for the stochastic gradient descent (SGD) scheme:
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
                vector<double> input = train_dataset[index];
                vector<double> target(output_size);
                int true_label = train_labels[index];
                target[true_label] = 1; // prepare the target vector in a way that only one element is non-zero

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
    printf("The training is done. Time to see how well the network can classify the images..\n");

    // test your network
    string test_path_images = "";
    string test_path_labels = "";
    int test_set_size = 100;
    unsigned char** test_dataset = read_mnist_images(test_path_images, test_set_size, input_size); // input_size = image size
    unsigned char* test_labels = read_mnist_labels(test_path_labels, output_size); // output_size = numer of labels

    int correct = 0;
    for (auto i = 0; i < test_set_size; i++) {
        vector<double> input = test_dataset[i];
        vector<double> target(output_size);
        int true_label = test_labels[i];
        target[true_label] = 1; // prepare the target vector in a way that only one element is non-zero

        vector<double> predicted = model.forward(input); // model prediction
        int max_label = *max_element(begin(predicted), end(predicted));
        if (max_label == true_label) correct++;
    }
    printf("Number of correctly predicted images: %f\n", 1.0 * correct / (1.0 * test_set_size));


    return 0;
}