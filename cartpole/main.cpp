// Will train a cartpole here with DQN algorithm
// Authors: Ruggero Malenfant 
#include <iostream>
#include <random>
#include <string>
#include <cmath>
#include <vector>
#include <ctime>
#include <algorithm>
#include <iomanip>
#include <iterator>
#include <tuple>
#include <fstream>
#include "Tensor.h"
#include "Environment.h"

using namespace std;

int main() {

    // cartpole parameters; I chosen same parameters as Sutton
    double M = 1.0;
    double m = 0.1;
    double L = 0.5; /* actually half the pole's length */
    // network parameters
    int input_size = 4;     // x, x_dot, theta, theta_dot
    int hidden_size = 64;
    int output_size = 2;    // two actions - push to the left or to the right
    int batch_size = 128;   // number of experiences sampled from the replay buffer
    double gamma = 0.99;    // to compute expected return
    double lr = 0.001;      // learning rate
    double eps_start = 0.9; // for greedy algorithm; at the beginning do more exploration and choose random actions
    double eps_end = 0.01;  // closer to the end we choose more greedily, relying more on network predictions
    double eps_decay = 1000.0;
    double tau = 0.005;     // for soft update of target net parameters 
    int goal = 1000;        // train the network until the pole can stay up for at least 10k updates
    int terminal_episode = 1000; // if the system doesn't learn to balance the pole in this N number of trials, terminate simulation

    // we need two networks for DQN algorithm: policy_net used for choosing actions from observations, and target_net computes value functions V(s, t+1) 
    // can't make random_device to work with MinGW compiler
    srand(std::time(0));
    int random_seed = rand();
    Tensor target_net(input_size, hidden_size, output_size, lr, random_seed); // updated softly 
    random_seed = rand();
    Tensor policy_net(input_size, hidden_size, output_size, lr, random_seed); // updated every step
    tuple<vector<vector<double>>, vector<double>, vector<vector<double>>, vector<double>> policy_net_params = policy_net.get_model_parameters();
    //printf("policy network parameters:\n");
    //policy_net.print_parameters();
    //printf("target network parameters before copying policy network params:\n");
    //target_net.print_parameters();
    target_net.copy_parameters(policy_net_params); // copy parameters 
    //printf("target network parameters after copying policy network params:\n");
    //target_net.print_parameters();

    // memory buffer
    vector <tuple <vector<double>, int, vector<double>, double>> replay_buffer; // collect experiences here

    // RNG
    //random_device rd{}; //doesn't work with my MinGW compiler, gives the same number... should work with other compilers
    //mt19937 RNG{rd()};
    srand(std::time(0));
    random_seed = rand();
    mt19937 RNG{ random_seed };

    // main simulation loop
    int print_stuff = 0; // 0 do not print; 1 print; for unit tests
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
            if (print_stuff == 1) printf("time step = %d\n", steps_done);
            current_epsilon = eps_end + (eps_start - eps_end) * exp(-(1.0 * steps_done) / eps_decay);
            if (print_stuff == 1) printf("current epsilon %f\n", current_epsilon);
            vector<double> state = my_little_cartpole.get_state(); // x, x_dot, theta, theta_dot; all zero at the beginning of the simulation
            if (print_stuff == 1) printf("current state: x = %f, x_dot = %f, theta = %f, theta_dot = %f\n", state[0], state[1], state[2], state[3]);
            int action = policy_net.select_action(state, current_epsilon); // 0 or 1 (left or right push)
            if (print_stuff == 1) printf("selected action: %d\n", action);
            // get tuple ( state, action, next_state, reward )
            tuple<vector<double>, int, vector<double>, double> transition = my_little_cartpole.update(action);
            vector<double> new_state = my_little_cartpole.get_state(); // x, x_dot, theta, theta_dot; all zero at the beginning of the simulation
            if (print_stuff == 1) printf("new state: x = %f, x_dot = %f, theta = %f, theta_dot = %f\n", new_state[0], new_state[1], new_state[2], new_state[3]);
            //printf("policy network parameters:\n");
            //policy_net.print_parameters();
            /*double print_saved_action = get<1>(transition);
            if (print_stuff == 1) printf("saved action: %d\n", print_saved_action);
            if (print_saved_action == 2) {
                printf("Error!\n");
            }*/
            double print_reward = get<3>(transition);
            if (print_stuff == 1) printf("reward = %f\n\n", print_reward);

            replay_buffer.push_back(transition); // save tuple to the memory buffer

            // we need to gather enough experiences to fill our table before start updating the network paramters
            if (replay_buffer.size() >= batch_size) {
                // compute loss and W & B gradients; I don't know how to do it better... If you do, please let me know
                double loss = 0.0;
                vector<vector<double>> w_gradients1(hidden_size, vector<double>(input_size));
                vector<vector<double>> w_gradients2(output_size, vector<double>(hidden_size));
                vector<double> b_gradients1(hidden_size);
                vector<double> b_gradients2(output_size);
                // create a batch of samples
                int current_bufffer_size = replay_buffer.size();
                if (print_stuff == 1) printf("current buffer size %d\n", current_bufffer_size);
                vector<int> batch(current_bufffer_size);
                //printf("batch before shuffling\n");
                for (auto i = 0; i < current_bufffer_size; i++) {
                    batch[i] = i; //cout << batch[i] << " ";
                }
                //cout << endl;
                random_shuffle(batch.begin(), batch.end());
                //printf("batch after shuffling\n");
                //for (auto i = 0; i < current_bufffer_size; i++) cout << batch[i] << " ";
                //cout << endl;

                // I follow this pytorch tutorial here: https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
                for (auto i = 0; i < batch_size; i++) {
                    int index = batch[i];
                    tuple <vector<double>, int, vector<double>, double> sampled_transition = replay_buffer[index];
                    if (print_stuff == 1) printf("sampled transition %d\n", index);
                    vector<double> sampled_state = get<0>(sampled_transition);
                    if (print_stuff == 1) printf("sampled state: x = %f, x_dot = %f, theta = %f, theta_dot = %f\n", sampled_state[0], sampled_state[1], sampled_state[2], sampled_state[3]);
                    int sampled_action = get<1>(sampled_transition);
                    if (print_stuff == 1) printf("sampled action %d\n", sampled_action);
                    vector<double> sampled_next_state = get<2>(sampled_transition);
                    if (print_stuff == 1) printf("sampled next state: x = %f, x_dot = %f, theta = %f, theta_dot = %f\n", sampled_next_state[0], sampled_next_state[1], sampled_next_state[2], sampled_next_state[3]);
                    double sampled_reward = get<3>(sampled_transition);
                    if (print_stuff == 1) printf("sampled reward %f\n", sampled_reward);
                    // Q(s,a)
                    vector<double> all_Q_s_a = policy_net.forward(sampled_state); // the reason why i have this code block here; solution? pass class instance to a function?
                    if (print_stuff == 1) printf("all Q(s,a)\n");
                    if (print_stuff == 1) {
                        for (auto a = 0; a < all_Q_s_a.size(); a++) cout << a << " " << all_Q_s_a[a] << endl;
                    }
                    double Q_s_a = all_Q_s_a[sampled_action];
                    if (print_stuff == 1) printf("selected Q(s,a) = %f\n", Q_s_a);
                    // V(s)
                    double Qs_max_next = 0.0;
                    if (sampled_reward >= 0) {
                        vector<double> all_Qs_next = target_net.forward(sampled_next_state); // the reason why i don't implement this block inside Tensor class
                        if (print_stuff == 1) printf("all Q(s+1,a)\n");
                        if (print_stuff == 1) {
                            for (auto a = 0; a < all_Qs_next.size(); a++) cout << a << " " << all_Qs_next[a] << endl;
                        }
                        vector<double>::iterator result = max_element(all_Qs_next.begin(), all_Qs_next.end());
                        Qs_max_next = *result;
                    }
                    if (print_stuff == 1) printf("max Q(s+1,a) = %f\n", Qs_max_next);

                    double expected_state_action_values = Qs_max_next * gamma + sampled_reward;
                    //printf("expected_state_action_values = %f\n", expected_state_action_values);
                    double delta = Q_s_a - expected_state_action_values;
                    if (print_stuff == 1) printf("delta = %f\n", delta);

                    // Huber loss
                    double loss = 0;
                    if (abs(delta) <= 1) {
                        loss = 0.5 * delta * delta;
                    }
                    else {
                        loss = (abs(delta) - 0.5);
                    }
                    if (print_stuff == 1) printf("loss = %f\n", loss);

                    //gradients; potential problem -- hidden vectors; i think they should stay same, and i don't have to use forward() again
                    // for now, let us pass TD difference, not loss
                    policy_net.compute_gradients(w_gradients1, b_gradients1, w_gradients2, b_gradients2, batch_size, sampled_action, loss);
                    if (print_stuff == 1) printf("gradients are computed!\n");
                }

                policy_net.optimizer_step(w_gradients1, b_gradients1, w_gradients2, b_gradients2); // update policy_net parameters
                if (print_stuff == 1) printf("policy_net.optimizer_step -- done\n");
                policy_net_params = policy_net.get_model_parameters(); // copy them
                target_net.soft_update(policy_net_params, tau); // soft update of target network parameters
                if (print_stuff == 1) printf("target_net.soft_update -- done\n");
            }

            // the cartpole failed, reward = -1
            double reward = get<3>(transition);
            if (reward < 0) {
                printf("Episode %d. Episode ended after %d steps\n", episode, steps_done);
                last_step = steps_done;
                episode_durations.push_back(steps_done);
                episode++;
                break;
            }
            // 
        }

        if (episode > terminal_episode) {
            printf("We haven't achieved our goal after %d episodes. Terminating training...", episode - 1);
            break;
        }
    }

    if (episode < terminal_episode) {
        printf("The training goal has been achieved after %d episodes\n", episode - 1);
    }
	
	string filename = "training_output.txt";
    ofstream output(filename.c_str(), std::ofstream::out);
    for (auto ep = 0; ep < episode - 1; ep++) {
        output << ep + 1 << "\t" << episode_durations[ep] << endl;
    }
    output.close();

    return 0;
}