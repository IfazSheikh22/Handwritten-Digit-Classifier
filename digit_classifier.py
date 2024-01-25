# Import the necessary libraries
import tensorflow as tf  # TensorFlow is a free and open-source software library for machine learning and artificial intelligence

# Load the MNIST dataset from TensorFlow's dataset API
mnist = tf.keras.datasets.mnist  # MNIST dataset, which is a set of 70,000 small images of digits handwritten by high school students and employees of the US Census Bureau

# Split the MNIST dataset into train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()  # x_train and x_test are arrays of grayscale image data with shape (num_samples, 28, 28); y_train and y_test are arrays of digit labels (integers in range 0-9) with shape (num_samples,)

# Normalize the pixel values from [0, 255] to [0, 1] for better performance
x_train, x_test = x_train / 255.0, x_test / 255.0

# Define the model architecture
model = tf.keras.models.Sequential([  # Sequential model is appropriate for a plain stack of layers where each layer has exactly one input tensor and one output tensor
  tf.keras.layers.Flatten(input_shape=(28, 28)),  # Flatten layer transforms the format of the images from a two-dimensional array (of 28 by 28 pixels) to a one-dimensional array (of 28 * 28 = 784 pixels)
  tf.keras.layers.Dense(128, activation='relu'),  # Dense layer is just a regular densely-connected neural network layer with 128 nodes (or neurons) and 'relu' activation function
  tf.keras.layers.Dense(10)  # Dense layer with 10 nodes without an activation function
])

# Compile the model
model.compile(optimizer='adam',  # Adam optimization is a stochastic gradient descent method that is based on adaptive estimation of first-order and second-order moments
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),  # Loss function is used to measure how well the model did on training, and then tries to improve on it using the optimizer
              metrics=['accuracy'])  # Accuracy is the fraction of the images that are correctly classified

# Train the model
model.fit(x_train, y_train, epochs=5)  # Fit the model to the training data

# Evaluate the model on the test data
model.evaluate(x_test,  y_test, verbose=2)
