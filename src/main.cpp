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

    model.train(10000, 0.03, x, y);

    vector<vector<double>> test_set_x = {{1.0, 1.0}, {1.0, 0.0}, {0.0, 1.0}, {0.0, 0.0}};
    vector<double> test_set_y = {0, 1, 1, 0};

    // Evaluation
    vector<vector<double>> output = model.forward(test_set_x);

    cout << "Test set: {";
    for(int i = 0; i < test_set_x.size(); i++){
        cout << "{";
        for(int j = 0; j<test_set_x[0].size(); j++){
            cout << test_set_x[i][j] << ", ";
        }
        cout << "}";
    }
    cout << "} \n";

    cout << "Generated set: {";

    for(int i = 0; i<output.size(); i++){
        cout << output[i][0] << ", ";
    }

    cout << "}";

    return 0;
}

// XOR training set -- Works perfectly fine.
// Image recognition -- Working on it.