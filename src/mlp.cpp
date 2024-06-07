/* In this project I shall write a neural net in C++
    * The neural net will be a multi-layer perceptron
    * The neural net will be trained using backpropagation
    * The neural net will be trained using stochastic gradient descent
*/

#include <iostream>
#include "layers.cpp"
#include <vector>
#include <cmath>

class MLP 
{
    private:
        FCLayer fc1 = FCLayer(2, 24, "RELU");
        FCLayer fc2 = FCLayer(24, 24, "RELU");
        FCLayer fc3 = FCLayer(24, 1, "None");
        double learning_rate = 0.001;

    public:

        vector<double> forward(vector<double> x)
        {
            // Forward pass
            vector<double> out = fc1.forward(x);
            out = fc2.forward(out);
            out = fc3.forward(out);
            return out;
        }

        double get_loss(vector<double> pred, vector<double> val)
        {
            // Going with MSE loss
            double loss = 0.0;
            for(int i = 0; i<pred.size(); i++){
                loss += pow((pred[i] - val[i]), 2);
            }

            return loss/pred.size();
        }

        void backpass(vector<double> true_vals){
            vector<double> dlda = vector<double>(true_vals.size(), 0.0);

            vector<double> inputs = this->fc1.input_data;
            vector<double> preds = this->forward(inputs);

            double loss = this->get_loss(preds, true_vals);

            for(int i = 0; i<preds.size(); i++){
                dlda[i] = (2 * (preds[i] - true_vals[i]))/preds.size();
            }

            // Initiating backpass
            vector<double>second_grad = this->fc3.backprop(dlda);
            this->fc3.update_params(this->learning_rate);

            vector<double>third_grad = this->fc2.backprop(second_grad);
            this->fc3.update_params(learning_rate);

            this->fc1.backprop(third_grad);
            this->fc1.update_params(learning_rate);
        }
};

