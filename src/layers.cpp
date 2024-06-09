#include <iostream>
#include <vector>
#include "../include/math.hpp"
#include "../include/activations.hpp"

using namespace std;

class FCLayer
{
    // A fully connected layer

    private:

        vector<vector<double>> weights;
        vector<double> biases;
        string activation;
        vector<vector<double>> raw;
        vector<vector<double>> weight_der;
        vector<double> bias_der;
    
    public:

        vector<vector<double>> input_data;

        FCLayer(int input_dim, int num_neurons, const string& activation){
            this->weights = initializeMatrix(input_dim, num_neurons, -0.1, 0.1);
            this->biases = initializeBias(num_neurons, -0.1, 0.1);
            this->activation = activation;
        }


        vector<vector<double>> forward(vector<vector<double>> data)
        {
            this->input_data = data;
            cout << "Data columns: " << data[0].size() << "\n"; // Debugging step -- Works
            vector<vector<double>> output = Multiply_matrices(data, this->weights);
            output = Add_biases(output, this->biases);
            this->raw = output;
            cout << "raw.shape: (" << raw.size() << ", "<< raw[0].size() << "\n"; // Debugging step -- Works
            output = applyActivation(output, this->activation);
            cout << "Output 1st element after activation: " << output[0][0] << "\n"; // Debugging step -- Works
            return output;
        }

        vector<vector<double>> backprop(vector<vector<double>> dlda){

            cout << "dlda shape: (" << dlda.size() << ", " << dlda[0].size() << ") \n";

            cout << "activated raw outputs shape: (" << applyActivation_derivative(this->raw, this->activation).size() << ", " << applyActivation_derivative(this->raw, this->activation)[0].size() << ") \n";

            vector<vector<double>> activated_derivative = applyActivation_derivative(this->raw, this->activation);
            vector<vector<double>> dldz = hammard(transposeMatrix(dlda), activated_derivative);

            cout << "Printing the raw values: \n"; // Debugging step -- Raw values have a shape of 4, 1. But only for the last layer. Expected.

            printMatrix(this->raw);

            cout << "\n Printing the activated values : \n";

            printMatrix(applyActivation_derivative(this->raw, this->activation));

            cout << "dldz shape: (" << dldz.size() << ", "<< dldz[0].size() << ")\n";
            printMatrix(dldz); // debugging step

            this->weight_der = transposeMatrix(Multiply_matrices(transposeMatrix(dldz), this->input_data));

            cout << "Printing the weight derivatives: \n";

            printMatrix(this->weight_der);

            this->bias_der = calculate_bias_derivatives(dldz);

            cout << "The bias derivative shape: (" << this->bias_der.size() << ") \n";
            
            cout << "Printing the bias derivatives: \n";
            
            for(int i = 0; i<this->bias_der.size(); i++){
                cout << this->bias_der[i] << " ";
            }


            vector<vector<double>> dldx = Multiply_matrices(dldz, transposeMatrix(this->weights));

            cout << "bias_der shape: (" << bias_der.size() << ")"; // Debugging step -- Works.

            return dldx;

        }

        void update_params(double lr){
            for(int i = 0; i<this->weights.size(); i++){
                for(int j = 0; j<this->weights[0].size(); j++){
                    this->weights[i][j] -= lr * this->weight_der[i][j];
                }
            }

            for (size_t i = 0; i < biases.size(); ++i) {
                biases[i] -= lr * this->bias_der[i];
            }

            cout << "Update params for the third layer worked! \n";
        }

};

// #TODO: The issue is definitely in the backpass. Look into the backpass for up until the second iteration of
// training. -- Mostly done.
// #TODO: Randomize the initialization of weights and biases. -- Done