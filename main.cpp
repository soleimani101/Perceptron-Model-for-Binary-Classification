#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>
#include <cmath>



// This function calculates the sigmoid function for the given input
double sigmoid(double z) {
    return 1.0 / (1.0 + std::exp(-z));
}
// This function loads a CSV file and returns its contents as a vector of vector of strings
std::vector<std::vector<std::string>> load_csv(std::string filename) {
    std::vector<std::vector<std::string>> dataset;
    std::ifstream file(filename);
    std::string line;
    while (std::getline(file, line)) {
        std::vector<std::string> row;
        std::string value;
        for (char c : line) {
            if (c == ',') {
                row.push_back(value);
                value.clear();
            } else {
                value += c;
            }
        }
        row.push_back(value);
        dataset.push_back(row);
    }
    return dataset;
}

// This function predicts the output for a single input example using the provided weights and bias
// The activation is calculated as the weighted sum of the input features plus the bias term
// If the activation is greater than or equal to zero, the output is predicted as 1.0
// Otherwise, the output is predicted as 0.0

double predict_output(std::vector<std::string> row, std::vector<double> weights, double bias) {
    double activation = bias;
    for (std::size_t i = 0; i < row.size() - 1; i++) {
        // Calculate the weighted sum of the input features
        activation += std::stod(row[i]) * weights[i];
    }
    // If the activation is greater than or equal to zero, predict output as 1.0
    // Otherwise, predict output as 0.0
    return activation >= 0.0 ? 1.0 : 0.0;
}

// This function trains a perceptron model using the provided dataset
// The initial weights are set to zero
// The bias is a hyperparameter that controls the decision boundary
// The algorithm runs for a specified number of iterations
// For each iteration, the weights are updated for each training example
// The error between the predicted and actual output is calculated for each example
// The weights are updated using the error and the input features of the example
// The bias is also updated using the error and the bias value
// The final weights are returned as the output of the function

std::vector<double> train_perceptron(std::vector<std::vector<std::string>> data, double bias, int num_iter) {
    std::vector<double> weights(data[0].size() - 1, 0.0);
    for (int iter = 0; iter < num_iter; iter++) {
        for (std::size_t i = 1; i < data.size(); i++) {
            auto row = data[i];
            double predicted_output = predict_output(row, weights, bias);
            double error = std::stod(row.back()) - predicted_output;
            for (std::size_t j = 0; j < row.size() - 1; j++) {
                // Update the weight for each input feature based on the error and the feature value
                weights[j] += error * std::stod(row[j]);
            }
            // Update the bias using the error and the bias value
            weights.back() += error * bias;
        }
    }
    // Return the final weights
    return weights;
}


int main() {
    // Load dataset from CSV file and remove header row
    auto dataset = load_csv("moons.csv");
    dataset.erase(dataset.begin());

    // Shuffle dataset randomly
    std::random_shuffle(dataset.begin(), dataset.end());

    // Split dataset into training and testing data (80/20 split)
    auto train_data = std::vector<std::vector<std::string>>(dataset.begin(), dataset.begin() + dataset.size() * 0.8);
    auto test_data = std::vector<std::vector<std::string>>(dataset.begin() + dataset.size() * 0.8, dataset.end());

    // Train the perceptron on the training data using a learning rate of 0.1 and 100 epochs
    auto weights = train_perceptron(dataset, 0.1, 100);

    // Test the perceptron on the testing data and calculate accuracy
    double accuracy = 0.0;
    for (auto row : test_data) {
        // Predict output using the trained weights and a bias of 1.0
        double predicted_output = predict_output(row, weights, 1.0);

        // Check if predicted output matches actual output, and increment accuracy counter if it does
        if (predicted_output == std::stod(row.back())) {
            accuracy += 1.0;
        }
    }
    accuracy /= test_data.size();  // Calculate accuracy as a fraction of the total test data size

    // Print the accuracy on the testing data
    std::cout << "Accuracy on test data: " << accuracy << std::endl;

    return 0;
}