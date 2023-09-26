// Will train a cartpole here with DQN algorithm
// Authors: Ruggero Malenfant 
#include <iostream>
#include <random>
#include <cmath>
#include <vector>
#include <ctime>
#include "Tensor.h"
#include "Environment.h"

using namespace std;

int main(){

    // cartpole parameters
    double M = 10;
    double m = 1;
    double L = 5;
    // network parameters
    int input_size = 4; // x, x_dot, theta, theta_dot
    int hidden_size = 32;
    int output_size = 2; // two actions - push to the left or to the right
    int batch_size = 128; // number of experiences sampled from the replay buffer
    double lr = 0.001;
    double eps_start = 0.9; // for greedy algorithm
    double eps_end = 0.01;
    double eps_decay = 1000.0;
    double tau = 0.005; // for soft update of target net parameters 
    double theta_fail = 0.1; // if theta exceeds this number, episode stops
    double x_fail = 100; // if cart leaves the simulation box, episode fails;
    int goal = 2000; // train the network until the pole can stay up for at least 10k updates

    // we need two networks for DQN algorithm
    Tensor targer_net(input_size, hidden_size, output_size, lr); // updated softly 
    Tensor policy_net(input_size, hidden_size, output_size, lr); // updated every step
    tuple<vector<vector<double>>, vector<double>, vector<vector<double>>, vector<double>> policy_net_params = policy_net.model_parameters();
    targer_net.copy_parameters(policy_net_params); // copy parameters 

    // memory buffer
    vector <tuple <vector<double>, int, vector<double>, double>> ReplayBuffer(10000); // collect experiences here

    // main simulation loop
    int episode_end = 0;
    int episode = 0;
    while (episode_end < goal) { 

        // initiate the cartpole model 
        Environment my_little_cartpole(M, m, L);
        vector<double> state(4); // x, x_dot, theta, theta_dot; all zero at the beginning of the simulation
        double current_epsilon = eps_start;

        // start balancing the pole
        for (auto steps_done = 0; steps_done < goal; steps_done++) {
            // select action
            current_epsilon = eps_end + (eps_start - eps_end) * exp(-(1.0 * steps_done) / eps_decay);
            int action = policy_net.select_action(state, current_epsilon, output_size); // 0 or 1 (left or right push)

            // get tuple of state, action, next_state, reward
            tuple<vector<double>, int, vector<double>, double> Transition = my_little_cartpole.update(action); 
            ReplayBuffer.push_back(Transition); // save it to the memory buffer


            int terminated = 0;
            if (abs(my_little_cartpole._theta) > theta_fail || 
                abs(my_little_cartpole._x) > x_fail) { // terminate episode 
                printf("Episode ended after %d steps\n", steps_done);
                terminated = 1;
                episode_end = steps_done;
                episode++;
                break;
            }


            policy_net.optimizer_step(state, action, reward);

        }
    }

    printf("The training is finished after %d episodes\n", episode);


    return 0;
}