#include <iostream>
#include "mlp.cpp"

using namespace std;

int main(){

    // Hyperparams
    int input_dim = 2;
    int num_neurons = 14;
    int output_dim = 1;
    double lr = 0.01;
    string activation = "Tanh";

    // Instantiating the model
    MLP model = MLP(input_dim, num_neurons, output_dim, lr, activation);

    // generating the XOR set to trainn it on
    vector<vector<double>> x = {{0.0, 0.0}, {0.0, 1.0}, {1.0, 0.0}, {1.0, 1.0}};
    vector<double> y = {0.0, 1.0, 1.0, 0.0};

    model.train(1000, 0.03, x, y);

    vector<vector<double>> test_set_x = {{1.0, 1.0}, {1.0, 1.0}, {0.0, 1.0}};
    vector<double> test_set_y = {0, 0, 1};

    // Evaluation
    vector<vector<double>> output = model.forward(test_set_x);

    for(int i = 0; i<output.size(); i++){
        cout<< "Here are the outputs: " << output[i][0] <<"\n";
    }

    return 0;
}