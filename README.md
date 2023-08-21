# alternative-ML-lib-C2plus
Here we will try creating our own C++ library for our Machine Learning research. Specifically, we will introduce a new type of data structure that we will be calling Tensors as in PyTorch, and write a few methods for it.


The skeleton of the library is in "Tensor.h" and "Tensor.cpp" files. At the moment, the simpliest possbile variant is written, which allows for only a single hidden layer and ReLU activations. 


Open problems:

1. Something is wrong with backprop method. The network cannot approach the negative target values.





Once the class and its methods are written:

1. Test the data structure and its methods for regression tasks
2. Test the data structure and its methods for classification tasks
3. Test the data structure and its methods for "gymnasium" Reinforcement Learning tasks
