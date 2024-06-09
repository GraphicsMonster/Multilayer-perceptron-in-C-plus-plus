#include <iostream>
#include "layers.cpp"
#include <vector>
#include <cmath>

class MLP 
{
    private:
        int input_dim, num_neurons, output_dim;
        string activations;
        FCLayer fc1;
        FCLayer fc2;
        FCLayer fc3;
        double learning_rate;

    public:

        MLP(int input_dim, int num_neurons, int output_dim, double learning_rate, string activations)
        {
            this->input_dim = input_dim;
            this->num_neurons = num_neurons;
            this->output_dim = output_dim;
            this->learning_rate = learning_rate;
            this->activations = activations;

            this->fc1 = FCLayer(input_dim, num_neurons, "RELU");
            this->fc2 = FCLayer(num_neurons, num_neurons, "RELU");
            this->fc3 = FCLayer(num_neurons, output_dim, "None");
        }

        vector<vector<double>> forward(vector<vector<double>> x)
        {
            // Forward pass
            vector<vector<double>> out = fc1.forward(x);
            out = fc2.forward(out);
            out = fc3.forward(out);
            return out;
        }

        double get_loss(vector<vector<double>> pred, vector<double> val)
        {
            // Going with MSE loss
            vector<double> predictions(pred.size(), 0.0);
            for(int i = 0; i<pred.size(); i++){
                predictions[i] = pred[i][0];
            }

            double loss = 0.0;
            for(int i = 0; i<pred.size(); i++){
                loss += pow((predictions[i] - val[i]), 2);
            }

            return loss/pred.size();
        }

        void backpass(vector<double> true_vals){
            vector<vector<double>> dlda(1, vector<double>(true_vals.size(), 0.0));

            vector<vector<double>> preds = this->forward(this->fc1.input_data);

            double loss = this->get_loss(preds, true_vals);

            for(int i = 0; i<preds.size(); i++){
                dlda[0][i] = (2 * (preds[i][0] - true_vals[i]))/preds.size();
            }

            // Initiating backpass
            vector<vector<double>>third_grad = this->fc3.backprop(dlda);
            this->fc3.update_params(this->learning_rate);

            vector<vector<double>>second_grad = this->fc2.backprop(transposeMatrix(third_grad));
            this->fc2.update_params(learning_rate);

            vector<vector<double>> first_grad = this->fc1.backprop(transposeMatrix(second_grad));
            this->fc1.update_params(learning_rate);
        }

        void train(int num_epochs, double learning_rate, vector<vector<double>> x, vector<double> y){
            this->learning_rate = learning_rate;

            for(int epoch = 0; epoch<num_epochs; epoch++){
                vector<vector<double>> preds = this->forward(x);
                this->backpass(y);
                double loss = this->get_loss(preds, y);;

                if((epoch+1) % 100 == 0){
                    cout << "Epoch: " << (epoch+1) << " || Loss: " << loss << "\n";
                }
            }
        }
};

