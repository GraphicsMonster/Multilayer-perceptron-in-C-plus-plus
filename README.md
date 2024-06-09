# A feedforward Neural network written from scratch in C++.

To once again strengthen my hold over C++ and to keep the weekend busy, I decided to write a feedforward neural network from scratch mimicking the implementation strategies we usually employ in Python, except using C++. This is a simple implementation and is not optimized for performance. The model consists of 2 hidden layers and 1 output layer. To test the network, I trained it on the XOR problem and had it predict the output for the same. The network was able to predict the output with 100% accuracy. The code is written in a way that it can be easily extended to include more hidden layers and neurons.

![XOR Problem predictions]("xor_set_results.png")

To further extend the testing of the network, one can easily make some modifications in the hyperparameters such that the dimensions fo the input and the weights align. The network can then be trained on any desired dataset and will likely produce good results. 

Suitable for classification(more than one neuron in the output layer) and regression problems(only one neuron in the output layer).

# File Structure
- `include` directory contains all the header files. This is where you will find the math functions, activation functions and their derivatives.
    - `activations.h` contains the declaration of the activation functions and their derivatives.
    - `math.h` contains the declaration of the math functions.
    - `activations.cpp` contains the implementation of the activation functions and their derivatives.
    - `math.cpp` contains the implementation of the math functions.

- `src` directory contains all the source files. This is where you will find the implementation of the neural network and the layers.
    - `main.cpp` is the driver code that trains the network on the XOR problem and predicts the output for the same.
    - `mlp.cpp` contains the implementation of the neural network.
    - `layers.cpp` contains the implementation of the layers.

# How to run the code?
1. Clone the repository.
2. Run the following commands in the terminal:
```bash
g++ -std=c++11 -Iinclude src/main.cpp src/mlp.cpp src/layers.cpp include/activations.cpp include/math.cpp -o neural_net
./neural_net
```
3. The code will train the network on the XOR problem and will predict the output for the same. The output will be displayed on the terminal.

# Dependencies
Absolutely none. The code is written from scratch and does not depend on any external libraries. All the required hardcore math, from hadamard product to derivatives of various activation functions, is written from scratch. You can find it all in the `include` directory. All the linear algebra operations are written in the `math.cpp` file and can be accessed by including the `math.h` header file in your code.

# License
This project is licensed under the MIT License - see the LICENSE file for details.