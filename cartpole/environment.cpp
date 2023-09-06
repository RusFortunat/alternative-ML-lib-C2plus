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
vector<double> Environment::update(int action){

    // 







    // return x, x_dot, theta, and theta_dot 
    return _x, _x_dot, _theta, _theta_dot;
}