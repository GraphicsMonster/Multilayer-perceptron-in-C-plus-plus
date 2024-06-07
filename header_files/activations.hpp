#ifndef ACTIVATIONS_H
#define ACTIVATIONS_H

#include <vector>
#include <stdexcept>
#include <cmath>
#include <algorithm>
#include <string>

std::vector<double> RELU(std::vector<double> a){
    for(int i = 0; i<a.size(); i++){
        a[i] = std::max(0.0, a[i]);
    }

    return a;
}

std::vector<double> Sigmoid(std::vector<double> a){
    for(int i = 0; i<a.size(); i++){
        a[i] = 1/(1+exp(-a[i]));
    }

    return a;
}

std::vector<double> Tanh(std::vector<double> a){
    for(int i = 0; i<a.size(); i++){
        a[i] = tanh(a[i]);
    }

    return a;
}

std::vector<double> Softmax(std::vector<double> a){
    double sum = 0;
    for(int i = 0; i<a.size(); i++){
        a[i] = exp(a[i]);
        sum += a[i];
    }

    for(int i = 0; i<a.size(); i++){
        a[i] = a[i]/sum;
    }

    return a;
}

std::vector<double> LeakyRELU(std::vector<double> a, double alpha){
    for(int i = 0; i<a.size(); i++){
        a[i] = std::max(alpha*a[i], a[i]);
    }

    return a;
}

std::vector<double> applyActivation(std::vector<double> data, std::string activation)
{
    if(activation == "RELU")
    {
        return RELU(data);
    }
    else if(activation == "Sigmoid")
    {
        return Sigmoid(data);
    }
    else if(activation == "Tanh")
    {
        return Tanh(data);
    }
    else if(activation == "Softmax")
    {
        return Softmax(data);
    }
    else if(activation == "LeakyRELU")
    {
        return LeakyRELU(data, 0.01);
    }
    else if(activation == "None")
    {
        return data;
    }
    else
    {
        throw std::invalid_argument("Invalid activation function.");
    }
}

// Now let us write the derivatives of these activation functions

std::vector<double> RELU_derivative(std::vector<double> a){
    for(int i = 0; i<a.size(); i++){
        a[i] = a[i] > 0 ? 1 : 0;
    }

    return a;
}

std::vector<double> Sigmoid_derivative(std::vector<double> a){
    for(int i = 0; i<a.size(); i++){
        a[i] = a[i] * (1 - a[i]);
    }

    return a;
}

std::vector<double> Tanh_derivative(std::vector<double> a){
    for(int i = 0; i<a.size(); i++){
        a[i] = 1 - pow(a[i], 2);
    }

    return a;
}

std::vector<double> Softmax_derivative(std::vector<double> a){
    for(int i = 0; i<a.size(); i++){
        a[i] = a[i] * (1 - a[i]);
    }

    return a;
}

std::vector<double> LeakyRELU_derivative(std::vector<double> a, double alpha){
    for(int i = 0; i<a.size(); i++){
        a[i] = a[i] > 0 ? 1 : alpha;
    }

    return a;
}

std::vector<double> applyActivation_derivative(std::vector<double> data, std::string activation)
{
    if(activation == "RELU")
    {
        return RELU_derivative(data);
    }
    else if(activation == "Sigmoid")
    {
        return Sigmoid_derivative(data);
    }
    else if(activation == "Tanh")
    {
        return Tanh_derivative(data);
    }
    else if(activation == "Softmax")
    {
        return Softmax_derivative(data);
    }
    else if(activation == "LeakyRELU")
    {
        return LeakyRELU_derivative(data, 0.01);
    }
    else if(activation == "None")
    {
        return data;
    }
    else
    {
        throw std::invalid_argument("Invalid activation function.");
    }
}

#endif