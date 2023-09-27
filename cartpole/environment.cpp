// Here the cartpole state will be updated
// Authors: me ^^
#include <iostream>
#include <cmath>
#include <vector>
#include <ctime>
#include "Environment.h"

using namespace std;

Environment::Environment(double M, double m, double L){

    _M = M; // mass of the block that can be pushed
    _m = m; // mass of the rod
    _L = L; // length of the rod

    // i will be always starting with x=0, x_dot = 0, and \theta = 0, \theta_dot = 0
    _x = 0; // cart position
    _x_dot = 0; // cart velocity
    _theta = 0; // the angle that the rod makes from the vertical axis
    _theta_dot = 0; // rod's angular velocity
}

Environment::~Environment(){
    // i'll just put it here
}

// the function receives the action to push cartpole to the left or to the right
tuple<vector<double>, int, vector<double>, double> Environment::update(int action){

    // Will copy this implementation from R. Sutton's book: http://incompleteideas.net/sutton/book/code/pole.c 

    double Force = action == 0 ? -10.0 : 10.0; // push the cart to the left for action = 0; otherwise to the right
    double TAU = 0.02; // time increment
    double g = 9.8; // gravity
    double theta_fail = 0.1; // if theta exceeds this number, episode stops
    double x_fail = 5;      // if cart leaves the simulation box, episode fails;
    double costheta = cos(_theta);
    double sintheta = sin(_theta);

    double temp = (Force + _L * _theta_dot * _theta_dot * sintheta) / (_M + _m);
    double thetaacc = (g * sintheta - costheta * temp) / (_L * (4.0/3.0 - _m * costheta * costheta / (_M + _m)));
    double xacc = temp - _m * _L * thetaacc * costheta / (_M + _m);

    vector<double> current_state = { _x, _x_dot, _theta, _theta_dot };

    /*** Update the four state variables, using Euler's method. ***/
    _x += TAU * _x_dot;
    _x_dot += TAU * xacc;
    _theta += TAU * _theta_dot;
    _theta_dot += TAU * thetaacc;

    vector<double> next_state = { _x, _x_dot, _theta, _theta_dot };

    double reward = 0.0;
    // terminate episode if cart goes to far or tilts too much
    if (abs(_theta) > theta_fail || abs(_x) > x_fail) {
        reward = -1.0;
        next_state.erase(next_state.begin(), next_state.begin() + next_state.size()); // empty vector, equivalent to python's next_state=None
    }
    else {
        reward = 1.0;
    }

    return make_tuple(current_state, action, next_state, reward);
}

vector<double> Environment::get_state() {

    return _x, _x_dot, _theta, _theta_dot;
}