#ifndef MATH_H
#define MATH_H

#include <vector>
#include <stdexcept>

std::vector<std::vector<double>> Multiply_matrices(const std::vector<std::vector<double>>& A, const std::vector<std::vector<double>>& B);

std::vector<double> Multiply_vector_matrix(std::vector<double>& data, std::vector<std::vector<double>>& weights);

std::vector<double> multiply_vectors(std::vector<double> a, std::vector<double> b);

std::vector<double> Add_vectors(std::vector<double> data, std::vector<double> biases);

std::vector<std::vector<double>> Add_biases(const std::vector<std::vector<double>>& matrix, const std::vector<double>& biases);

std::vector<std::vector<double>> Add_biases(const std::vector<std::vector<double>>& matrix, const std::vector<double>& biases);

double getRandomDouble(double min, double max);

std::vector<std::vector<double>> initializeMatrix(int rows, int cols, double min=-0.1, double max=0.1);

std::vector<double> initializeBias(int size, double min, double max);

void printMatrix(const std::vector<std::vector<double>>& matrix);

std::vector<std::vector<double>> transposeMatrix(const std::vector<std::vector<double>>& matrix);

std::vector<std::vector<double>> hammard(const std::vector<std::vector<double>>& matrix1, const std::vector<std::vector<double>>& matrix2);

#endif
