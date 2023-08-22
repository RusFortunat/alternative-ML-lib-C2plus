# alternative-ML-lib-C2plus
Here we will try creating our own C++ library for our Machine Learning research. Specifically, we will introduce a new type of data structure that we will be calling Tensors as in PyTorch, and write a few methods for it.


The skeleton of the library is in "Tensor.h" and "Tensor.cpp" files.


Open problems:

1. At the moment the structure of the network defined in Tensor class is very rigid: it allows only for a single hidden layer and forwardprop and backprop only use ReLU activation. It would be imperative to improve that at some point and allow initializing more complex networks with various activations.

2. We need to implement Stochastic Gradient descent. At the moment the network takes the whole input data and does the gradient descent for the whole data batch. We need to change that. 
   
3. I have tried supplying the input vector to a network and adjusting the network parameters for the output to fit target values, and the program seems to work great when the target values are positive. However, if the target set has negative elements, the network will be facing "dying ReLU" problem, when certain neurons turn to 0 and get stuch there because of ReLU activation function. Not sure at the moment if this is something we should concern ourselves with or not.






Once the class and its methods are written:

1. Test the data structure and its methods for regression tasks
2. Test the data structure and its methods for classification tasks
3. Test the data structure and its methods for "gymnasium" Reinforcement Learning tasks
