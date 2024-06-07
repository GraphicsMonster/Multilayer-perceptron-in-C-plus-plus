#ifndef MATH_H
#define Math_H

#include <vector>
#include <stdexcept>


std::vector<double> Multiply_matrices(std::vector<double>& data, std::vector<std::vector<double>>& weights)
{
    if(data.size() != weights.size())
    {
        throw std::invalid_argument("The second matrix should have the same number of rows as that of columns in the first matrix.");
    }

    std::vector<double> output(weights[0].size(), 0.0);

    for(int i = 0; i<weights[0].size(); i++)
        {
            for(int j = 0; j<data.size(); j++)
            {
                output[i] += data[j] * weights[j][i];
            }
        }
    

    return output;
}

std::vector<double> multiply_vectors(std::vector<double> a, std::vector<double> b)
{
    if(a.size() != b.size())
    {
        throw std::invalid_argument("The two vectors should have the same size.");
    }

    std::vector<double> output(a.size(), 0.0);

    for(int i = 0; i<a.size(); i++)
    {
        output[i] = a[i] * b[i];
    }

    return output;
}

std::vector<double> Add_vectors(std::vector<double> data, std::vector<double> biases)
{
    for(int i = 0; i<biases.size(); i++){
        data[i] += biases[i];
    }

    return data;
}


#endif