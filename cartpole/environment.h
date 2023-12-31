// File with Environment class methods and variables
// Authors: 
#pragma once
#include <iostream>
#include <vector>
using namespace std;

class Environment{
    public:
        double _x;
        double _x_dot;
        double _theta;
        double _theta_dot;

        // not sure if anything else is needed 
        Environment(double M, double m, double L); // constructor with random parameters
        ~Environment();

        tuple<vector<double>, int, vector<double>, double> update(int& action); // forwardprop with ReLU activation

        vector<double> get_state();

    private:
        double _M;
        double _m;
        double _L;
};