# Neural Network Without TensorFlow/Keras Using Only Math

This project implements a neural network from scratch, using only mathematical libraries in Python, without the use of TensorFlow, Keras, or any other high-level deep learning library. The neural network is trained and evaluated on the MNIST dataset.

## Description

The project includes:
- Reading and preprocessing MNIST data.
- Implementation of forward and backward propagation.
- Using ReLU and softmax activation functions.
- Training the neural network using gradient descent with momentum.
- Visualization of examples from the dataset and performance metrics during training.

## Requirements

To run this project, you need to have the following libraries installed:

- numpy>=1.0
- pandas>=1.0
- matplotlib>=3.0

You can install the dependencies using the `requirements.txt` file with the following command:

```bash
pip install -r requirements.txt
```

## Usage

1. Clone the repository:

```bash
git clone https://github.com/your-username/your-repository.git
cd your-repository
```

2. Make sure to install the dependencies:

```bash
pip install -r requirements.txt
```

3. Run the main script in your favorite development environment or in a Jupyter Notebook.

## Project Structure

- `mnist_train.csv`: Data file for the MNIST training set.
- `main.py` or `notebook.ipynb`: Main file containing the code to train and evaluate the neural network.
- `requirements.txt`: File listing the necessary dependencies for the project.

## Example Execution

The script trains a neural network on the MNIST dataset and shows some example predictions. It also plots the loss and accuracy during training.

### Visualization of Examples

```python
fig, axes = plt.subplots(2, 5, figsize=(10, 5))
fig.suptitle('Examples of digits in the MNIST dataset')

for i in range(10):
    ax = axes[i // 5, i % 5]
    ax.imshow(X_train[:, i].reshape(28, 28) * 255, cmap='gray')
    ax.set_title(f'Label: {Y_train[i]}')
    ax.axis('off')

plt.show()
```

### Training Results

```python
W1, b1, W2, b2, losses, accuracies = gradient_descent(X_train, Y_train, 0.1, 500, 0.01, 0.9)

fig, ax1 = plt.subplots()

ax2 = ax1.twinx()
ax1.plot(range(0, 500, 10), losses, 'g-')
ax2.plot(range(0, 500, 10), accuracies, 'b-')

ax1.set_xlabel('Iterations')
ax1.set_ylabel('Loss', color='g')
ax2.set_ylabel('Accuracy', color='b')

plt.title('Loss and Accuracy during Training')
plt.show()
```

### Prediction Tests

```python
test_prediction(0, W1, b1, W2, b2)
test_prediction(1, W1, b1, W2, b2)
test_prediction(2, W1, b1, W2, b2)
test_prediction(3, W1, b1, W2, b2)
```
