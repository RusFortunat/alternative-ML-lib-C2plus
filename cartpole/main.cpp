// Will train a cartpole here with DQN algorithm
// Authors: Ruggero Malenfant 
#include <iostream>
#include <random>
#include <cmath>
#include <vector>
#include <ctime>
#include <cassert>
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
    int goal = 2000;        // train the network until the pole can stay up for at least 10k updates
    int terminal_episode = 1000; // if the system doesn't learn to balance the pole in this N number of trials, terminate simulation

    // we need two networks for DQN algorithm: policy_net used for choosing actions from observations, and target_net computes value functions V(s, t+1) 
    Tensor target_net(input_size, hidden_size, output_size, lr, gamma); // updated softly 
    Tensor policy_net(input_size, hidden_size, output_size, lr, gamma); // updated every step
    tuple<vector<vector<double>>, vector<double>, vector<vector<double>>, vector<double>> policy_net_params = policy_net.get_model_parameters();
    target_net.copy_parameters(policy_net_params); // copy parameters 

    // memory buffer
    vector <tuple <vector<double>, int, vector<double>, double>> replay_buffer(10000); // collect experiences here

    // RNG
    //random_device rd{}; //doesn't work with my MinGW compiler, gives the same number... should work with other compilers
    //mt19937 RNG2{rd()};
    srand(std::time(0));
    const int random_seed = rand();
    mt19937 RNG{ random_seed };

    // main simulation loop
    vector<int> episode_durations;
    int episode = 1; 
    int last_step = 0;
    while (last_step < goal) {

        double current_epsilon = eps_start;

        // initiate the cartpole model 
        Environment my_little_cartpole(M, m, L);

        // start balancing the pole
        for (auto steps_done = 0; steps_done < goal; steps_done++) {
            // select action
            current_epsilon = eps_end + (eps_start - eps_end) * exp(-(1.0 * steps_done) / eps_decay);
            vector<double> state = my_little_cartpole.get_state(); // x, x_dot, theta, theta_dot; all zero at the beginning of the simulation
            int action = policy_net.select_action(state, current_epsilon, output_size); // 0 or 1 (left or right push)
            // get tuple ( state, action, next_state, reward )
            tuple<vector<double>, int, vector<double>, double> transition = my_little_cartpole.update(action); 
            replay_buffer.push_back(transition); // save tuple to the memory buffer

            // compute loss and W & B gradients; I don't know how to do it better... If you do, please let me know
            vector<double> loss(output_size);
            vector<vector<double>> w_gradients1(hidden_size, vector<double>(input_size));
            vector<vector<double>> w_gradients2(output_size, vector<double>(hidden_size));
            vector<double> b_gradients1(hidden_size);
            vector<double> b_gradients2(output_size);
            // select (batch_size) number of transitions from ReplayBuffer
            vector<tuple <vector<double>, int, vector<double>, double>> batch; // state, action, next_state, reward
            size_t m{ batch_size };
            ranges::sample(replay_buffer, std::back_inserter(batch), m, RNG);
            assert(batch.size() == m);
            // I follow this pytorch tutorial here: https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
            for (auto i = 0; i < batch_size; i++) {
                tuple <vector<double>, int, vector<double>, double> sampled_transition = batch[i];
                vector<double> state = get<0>(sampled_transition);
                int action = get<1>(sampled_transition);
                vector<double> next_state = get<2>(sampled_transition);
                double reward = get<3>(sampled_transition);
                vector<double> all_Qs = policy_net.forward(state); // the reason why i have this code block here; solution? pass class instance to a function?
                double Q_s_a = all_Qs[action];

                vector<double> all_Qs_next = target_net.forward(next_state); // the reason why i don't implement this block inside Tensor class
                double V_s_next = max_element(begin(all_Qs_next), end(all_Qs_next));

                double expected_state_action_values = V_s_next * gamma + reward;
                double delta = Q_s_a - expected_state_action_values;
                // Huber loss
                if (abs(delta) <= 1) {
                    loss[action] += 0.5 * delta * delta / (1.0 * batch_size);
                }
                else {
                    loss[action] += (abs(delta) - 0.5) / (1.0 * batch_size);
                }

                //gradients; potential problem -- hidden vectors; i think they should stay same, and i don't have to use forward() again
                policy_net.compute_gradients(w_gradients1, b_gradients1, w_gradients2, b_gradients2, batch_size, loss);
            }

            policy_net.optimizer_step(w_gradients1, b_gradients1, w_gradients2, b_gradients2); // update policy_net parameters
            policy_net_params = policy_net.get_model_parameters(); // copy them
            target_net.soft_update(policy_net_params, tau); // soft update of target network parameters

            // the cartpole failed, reward = -1
            if (transition[3] == -1) {
                printf("Episode %d. Episode ended after %d steps\n", episode, steps_done);
                last_step = steps_done;
                episode_durations.push_back(steps_done);
                episode++;
                break;
            }
            // terminate simulation if it goes for too long
            if (episode > terminal_episode) break;
        }
    }

    printf("The training is finished after %d episodes\n", episode);

    return 0;
}