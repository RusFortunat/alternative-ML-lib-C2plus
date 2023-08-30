# alternative-ML-lib-C2plus
Here we will try creating our own C++ library for our Machine Learning research. Specifically, we will introduce a new type of data structure that we will be calling Tensors as in PyTorch, and write a few methods for it.


The skeleton of the library is in "Tensor.h" and "Tensor.cpp" files.


Open problems:

1. Rigid, case-specific network structure and methods. The Stochastic Gradient Descent (SGD) method is implemented, but I have realized that generalizing the Tensor class so it can do both classification tasks and regression analysis is hard... To proceed right now, I have separate folders and put different implementations of the network class in each of them. Perhaps, some day I will find time to generalize the method for it to be able to deal with any of these tasks.






Once the class and its methods are written:

1. Test the data structure and its methods for regression tasks -- done!
2. Test the data structure and its methods for classification tasks
3. Test the data structure and its methods for "gymnasium" Reinforcement Learning tasks
