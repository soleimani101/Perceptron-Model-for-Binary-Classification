# Perceptron Model for Binary Classification

This is a C++ implementation of a Perceptron model for binary classification. The model takes input data in the form of a CSV file, where each row represents a training example, and the last column represents the class label (0 or 1). The implementation uses the sigmoid function to make predictions, and the weights are updated using the error between the predicted and actual output.
# Requirements

   - C++ compiler
   - iostream library
   - fstream library
   - vector library
   - algorithm library
   - cmath library

# Usage

Prepare a CSV file with training data, where each row represents a training example, and the last column represents the class label (0 or 1).
Change the file name in the load_csv() function to match the name of your CSV file.
Run the program to train the model and test it on the testing data.The accuracy of the model on the testing data will be printed on the console.

# Functions

# sigmoid()

This function calculates the sigmoid function for the given input.

```c++
double sigmoid(double z)
```

# load_csv()

This function loads a CSV file and returns its contents as a vector of vector of strings.

```c++
std::vector<std::vector<std::string>> load_csv(std::string filename)
```

# predict_output()

This function predicts the output for a single input example using the provided weights and bias.

```C++
double predict_output(std::vector<std::string> row, std::vector<double> weights, double bias)
```

# train_perceptron()

This function trains a perceptron model using the provided dataset.

```C++
std::vector<double> train_perceptron(std::vector<std::vector<std::string>> data, double bias, int num_iter)
```

