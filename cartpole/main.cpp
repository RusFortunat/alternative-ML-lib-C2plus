// Will train a cartpole here
// Authors: 
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
    double lr = 0.001;
    double theta_fail = 0.1; // if theta exceeds this number, episode stops
    double x_fail = 100; // if cart leaves the simulation box, episode fails;
    int goal = 10000; // train the network until the pole can stay up for at least 10k updates

    // DQN network
    Tensor DQN_model(lr);

    // main simulation loop
    int episode_end = 0;
    int episode = 0;
    while (episode_end < goal) { 

        // initiate the cartpole model 
        Environment my_little_cartpole(M, m, L);
        vector<double> state(4); // x, x_dot, theta, theta_dot; all zero at the beginning of the simulation

        // start balancing the pole
        for (auto update = 0; update < goal; update++) {
            
            int action = DQN_model.forward(state);
            my_little_cartpole.update(action);

            int terminated = 0;
            if (abs(my_little_cartpole._theta) > theta_fail || 
                abs(my_little_cartpole._x) > x_fail) { // terminate episode 
                printf("Episode ended after %d updates\n", update);
                terminated = 1;
                episode_end = update;
                episode++;
                break;
            }

            double reward = compute_reward(state, update, terminated);
            DQN_model.optimizer_step(state, action, reward);
        }
    }

    printf("The training is finished after %d episodes\n", episode);


    return 0;
}