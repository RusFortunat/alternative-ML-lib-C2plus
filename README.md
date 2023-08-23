# alternative-ML-lib-C2plus
Here we will try creating our own C++ library for our Machine Learning research. Specifically, we will introduce a new type of data structure that we will be calling Tensors as in PyTorch, and write a few methods for it.


The skeleton of the library is in "Tensor.h" and "Tensor.cpp" files.


Open problems:

1. At the moment the structure of the network defined in Tensor class is very rigid: it allows only for a single hidden layer and forwardprop and backprop only use ReLU activation. It would be imperative to improve that at some point and allow initializing more complex networks with various activations.

2. Stochastic Gradient Descent (SGD) is implemented, but I have realized that generalizing the class so it can do both classification tasks and regression analysis is hard... To proceed right now, I will have separate folders with different implementations of the network class in each of them. Perhaps, some day I will find time to generalize the method for it to be able to deal with any of these tasks.






Once the class and its methods are written:

1. Test the data structure and its methods for regression tasks
2. Test the data structure and its methods for classification tasks
3. Test the data structure and its methods for "gymnasium" Reinforcement Learning tasks
