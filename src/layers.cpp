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
            this->weights = vector<vector<double>>(input_dim, vector<double>(num_neurons, 0));
            this->biases = vector<double>(num_neurons, 0);
            this->activation = activation;
        }


        vector<vector<double>> forward(vector<vector<double>> data)
        {
            this->input_data = data;
            vector<vector<double>> output = Multiply_matrices(data, this->weights);
            output = Add_biases(output, this->biases);
            this->raw = output;
            output = applyActivation(output, this->activation);
            return output;
        }

        vector<vector<double>> backprop(vector<vector<double>> dlda){

            vector<vector<double>> dldz = Multiply_matrices(dlda, applyActivation_derivative(this->raw, this->activation));

            this->weight_der = vector<vector<double>>(this->weights.size(), vector<double>(this->weights[0].size()));

            this->weight_der = Multiply_matrices(dldz, this->input_data);

            for(int i = 0; i<dldz.size(); i++){
                for(int j = 0; j<dldz[0].size(); j++){
                    this->bias_der[j] += dldz[i][j];
                }
            }

            vector<vector<double>> dldx = Multiply_matrices(dldz, this->weights);
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
        }

};

// TODO: Edit everything such that vectors are represented as 2 dimensional matrices with shapes (1, something)