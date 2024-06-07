#include <iostream>
#include <vector>
#include "./header_files/math.hpp"
#include "./header_files/activations.hpp"

using namespace std;

class FCLayer
{
    // A fully connected layer

    private:

        vector<vector<double>> weights;
        vector<double> biases;
        string activation;
        vector<double> raw;
        vector<vector<double>> weight_der;
        vector<double> bias_der;
    
    public:

        vector<double> input_data;

        FCLayer(int input_dim, int num_neurons, const string& activation){
            this->weights = vector<vector<double>>(input_dim, vector<double>(num_neurons, 0));
            this->biases = vector<double>(num_neurons, 0);
            this->activation = activation;
        }


        vector<double> forward(vector<double> data)
        {
            this->input_data = data;
            vector<double> output = Multiply_matrices(data, this->weights);
            output = Add_vectors(output, this->biases);
            this->raw = output;
            output = applyActivation(output, this->activation);
            return output;
        }

        vector<double> backprop(vector<double> dlda){

            vector<double> dldz = multiply_vectors(dlda, applyActivation_derivative(this->raw, this->activation));

            this->weight_der = vector<vector<double>>(this->weights.size(), vector<double>(this->weights[0].size()));

            for(int i = 0; i<weights.size(); i++){
                for(int j = 0; j<weights[0].size(); j++){
                    this->weight_der[i][j] = this->input_data[i] * dldz[j];
                }
            }

            this->bias_der = dldz;

            vector<double> dldx(this->input_data.size(), 0.0);
            for (size_t i = 0; i < dldx.size(); ++i) {
                for (size_t j = 0; j < dldz.size(); ++j) {
                    dldx[i] += dldz[j] * weights[i][j];
                }
            }

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