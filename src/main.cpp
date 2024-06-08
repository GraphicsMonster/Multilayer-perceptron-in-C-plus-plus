#include <iostream>
#include "mlp.cpp"

using namespace std;

int main(){

    // Instantiating the model
    MLP model = MLP();

    // generating the XOR set to tarin it on
    vector<vector<double>> x = {{0.0, 0.0}, {0.0, 1.0}, {1.0, 0.0}, {1.0, 1.0}};
    vector<double> y = {0.0, 1.0, 1.0, 0.0};

    model.train(100, 0.001, x, y);

    vector<vector<double>> test_set_x = {{1.0, 1.0}, {1.0, 1.0}, {0.0, 1.0}};
    vector<double> test_set_y = {0, 0, 1};

    // Evaluation
    vector<vector<double>> output = model.forward(test_set_x);

    for(int i = 0; i<output.size(); i++){
        cout<< "Here are the outputs: " << output[i][0] <<"\n";
    }

    return 0;
}