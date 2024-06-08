#ifndef MATH_H
#define MATH_H

#include <vector>
#include <stdexcept>

std::vector<std::vector<double>> Multiply_matrices(const std::vector<std::vector<double>>& A, const std::vector<std::vector<double>>& B);

std::vector<double> Multiply_vector_matrix(std::vector<double>& data, std::vector<std::vector<double>>& weights);

std::vector<double> multiply_vectors(std::vector<double> a, std::vector<double> b);

std::vector<double> Add_vectors(std::vector<double> data, std::vector<double> biases);

std::vector<std::vector<double>> Add_biases(std::vector<std::vector<double>>& data, std::vector<double>& biases);

#endif
