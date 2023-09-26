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
    int input_size = 4;     // x, x_dot, theta, theta_dot
    int hidden_size = 32;
    int output_size = 2;    // two actions - push to the left or to the right
    int batch_size = 128;   // number of experiences sampled from the replay buffer
    double gamma = 0.99;    // to compute expected return
    double lr = 0.001;      // learning rate
    double eps_start = 0.9; // for greedy algorithm; at the beginning do more exploration and choose random actions
    double eps_end = 0.01;  // closer to the end we choose more greedily, relying more on network predictions
    double eps_decay = 1000.0;
    double tau = 0.005;     // for soft update of target net parameters 
    double theta_fail = 0.1; // if theta exceeds this number, episode stops
    double x_fail = 5;      // if cart leaves the simulation box, episode fails;
    int goal = 2000;        // train the network until the pole can stay up for at least 10k updates

    // we need two networks for DQN algorithm: policy_net used for choosing actions from observations, and target_net computes value functions V(s, t+1) 
    Tensor target_net(input_size, hidden_size, output_size, lr, gamma); // updated softly 
    Tensor policy_net(input_size, hidden_size, output_size, lr, gamma); // updated every step
    tuple<vector<vector<double>>, vector<double>, vector<vector<double>>, vector<double>> policy_net_params = policy_net.get_model_parameters();
    target_net.copy_parameters(policy_net_params); // copy parameters 

    // memory buffer
    vector <tuple <vector<double>, int, vector<double>, double>> ReplayBuffer(10000); // collect experiences here

    // main simulation loop
    vector<int> episode_durations;
    int episode = 1;
    while (last_step < goal) {

        // initiate the cartpole model 
        Environment my_little_cartpole(M, m, L);
        vector<double> state(4); // x, x_dot, theta, theta_dot; all zero at the beginning of the simulation
        double current_epsilon = eps_start;

        // start balancing the pole
        for (auto steps_done = 0; steps_done < goal; steps_done++) {
            // select action
            current_epsilon = eps_end + (eps_start - eps_end) * exp(-(1.0 * steps_done) / eps_decay);
            int action = policy_net.select_action(state, current_epsilon, output_size); // 0 or 1 (left or right push)
            // get tuple ( state, action, next_state, reward )
            tuple<vector<double>, int, vector<double>, double> Transition = my_little_cartpole.update(action); 
            ReplayBuffer.push_back(Transition); // save tuple to the memory buffer

            policy_net.optimizer_step(ReplayBuffer, batch_size); // update policy_net parameters
            policy_net_params = policy_net.get_model_parameters(); // copy them
            target_net.soft_update(policy_net_params, tau); // soft update of target network parameters

            // terminate episode if cart goes to far or tilts too much
            if (abs(my_little_cartpole._theta) > theta_fail || abs(my_little_cartpole._x) > x_fail) { 
                printf("Episode %d. Episode ended after %d steps\n", episode, steps_done);
                episode_durations.push_back(steps_done);
                episode++;
                break;
            }
        }
    }

    printf("The training is finished after %d episodes\n", episode);

    return 0;
}