#include "math.hpp"

std::vector<std::vector<double>> Multiply_matrices(const std::vector<std::vector<double>>& A, const std::vector<std::vector<double>>& B) {
    if (A[0].size() != B.size()) {
        throw std::invalid_argument("The number of columns in the first matrix must be equal to the number of rows in the second matrix.");
    }

    // Dimensions of the resulting matrix
    size_t rows = A.size();
    size_t cols = B[0].size();
    size_t inner_dim = B.size();

    // Initialize the result matrix with zeros
    std::vector<std::vector<double>> result(rows, std::vector<double>(cols, 0.0));

    // Perform matrix multiplication
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            for (size_t k = 0; k < inner_dim; ++k) {
                result[i][j] += A[i][k] * B[k][j];
            }
        }
    }

    return result;
}


std::vector<double> Multiply_vector_matrix(std::vector<double>& data, std::vector<std::vector<double>>& weights)
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

std::vector<std::vector<double>> Add_biases(std::vector<std::vector<double>>& data, std::vector<double>& biases)
{
    for(int i = 0; i<data.size(); i++){
        for(int j = 0; data[0].size(); j++){
                data[j][i] += biases[j];
        }
    }

    return data;
}