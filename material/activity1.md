# Activity 1:


> Start by uploading the [cereal](../datasets/cereal.csv) dataset  to your Colab environment. You can do this by clicking on the 'Files' tab on the left sidebar, then selecting 'Upload' and choosing the 'cereal.csv' file from your local machine."

## Part 1/3: Introduction to Keras

1. **Import Libraries**:
```python
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
import pandas as pd
```

We begin by importing the necessary libraries. NumPy is imported to handle numerical computations, Matplotlib for visualization, Pandas for data manipulation, Keras for building neural network models, and train_test_split from scikit-learn for splitting the data into training and testing sets.

2. **Load Datasets**:

```python
# Load datasets
cereal_data = pd.read_csv("cereal.csv")
```

We load the cereal and concrete datasets from CSV files. Ensure you have these files downloaded and placed in your working directory.

3. **Preprocess Datasets**:

```python
# Preprocess datasets
cereal_features = cereal_data[['calories', 'protein', 'fat', 'sodium', 'fiber', 'carbo', 'sugars', 'potass', 'vitamins']]
cereal_target = cereal_data['rating']

# Convert non-numeric values to NaN and then fill or drop them
cereal_features = cereal_features.apply(pd.to_numeric, errors='coerce')
cereal_features.dropna(inplace=True)
cereal_target = cereal_target[cereal_features.index]
```

For simplicity, we preprocess each dataset by selecting a subset of features and treating the task as a regression problem. We extract features and target variables from each dataset.

**Split datasets into training and testing sets**

```python
# Split datasets into training and testing sets
cereal_X_train, cereal_X_test, cereal_y_train, cereal_y_test = train_test_split(cereal_features, cereal_target, test_size=0.2, random_state=42)
```

4. **Define the Model**:

```python
# Define a Sequential model
model = Sequential()

# Add dense layers
model.add(Dense(64, activation='relu', input_shape=(cereal_X_train.shape[1],)))
model.add(Dense(32, activation='relu'))
model.add(Dense(1))
```

We define a Sequential model in Keras. A Sequential model allows you to create neural networks layer-by-layer. We add dense layers to the model, specifying the number of neurons and activation functions.

5. **Compile the Model**:

```python
# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])

# Print model summary
model.summary()
```

After defining the model architecture, we compile it. Compiling the model involves specifying the optimizer, loss function, and metrics to monitor during training.

6. **Train the Model**:

```python
# Train the model
history_cereal = model.fit(cereal_X_train, cereal_y_train, epochs=10, batch_size=64, validation_data=(cereal_X_test, cereal_y_test), verbose=1)
```

We train the model on the cereal dataset using the fit method. We specify the training data, validation data, number of epochs, batch size, and verbosity level.

7. **Visualize Training History**:
```python
# Plot training history
plt.plot(history_cereal.history['loss'], label='Cereal Train Loss')
plt.plot(history_cereal.history['val_loss'], label='Cereal Validation Loss')
plt.title('Training and Validation Loss for Cereal Dataset')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()
```

Finally, we plot the training and validation loss over epochs using Matplotlib. This allows us to visualize how the model's performance changes during training.

> Here's the [complete code](./lab1part1.py)

----
## Part 2/2: Early stopping

In Part 2, we'll modify the code to include early stopping. This modified prevents overfitting during model training. Early stopping allows the training process to stop if the validation loss does not improve for a certain number of epochs, thereby preventing overfitting.

**Part 2/3: Implementing Early Stopping**

1. **Import Libraries**:
```python
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
import pandas as pd
```

2. **Load and Preprocess Dataset**:
```python
# Load datasets
cereal_data = pd.read_csv("cereal.csv")
concrete_data = pd.read_csv("concrete.csv")

# Preprocess datasets
cereal_features = cereal_data[['calories', 'protein', 'fat', 'sodium', 'fiber', 'carbo', 'sugars', 'potass', 'vitamins']]
cereal_target = cereal_data['rating']

# Convert non-numeric values to NaN and then fill or drop them
cereal_features = cereal_features.apply(pd.to_numeric, errors='coerce')
cereal_features.dropna(inplace=True)
cereal_target = cereal_target[cereal_features.index]

# Split datasets into training and testing sets
cereal_X_train, cereal_X_test, cereal_y_train, cereal_y_test = train_test_split(cereal_features, cereal_target, test_size=0.2, random_state=42)
```

3. **Define the Model**:
```python
# Define a Sequential model
model = Sequential()

# Add dense layers
model.add(Dense(64, activation='relu', input_shape=(cereal_X_train.shape[1],)))
model.add(Dense(32, activation='relu'))
model.add(Dense(1))
```

4. **Compile the Model**:
```python
# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
```

5. **Define Early Stopping Callback**:
```python
# Define early stopping callback
early_stopping = EarlyStopping(monitor='val_loss', patience=3, verbose=1, restore_best_weights=True)

# Print model summary
model.summary()
```

In this step, we define an early stopping callback using Keras' `EarlyStopping` class. The purpose of this callback is to monitor the validation loss during training and stop the training process when the validation loss stops improving. 

- **monitor**: This parameter specifies the quantity to be monitored during training. In this case, we set it to `'val_loss'`, indicating that we want to monitor the validation loss.

- **patience**: Patience is the number of epochs with no improvement after which training will be stopped. In our code, we set it to `3`, meaning that training will stop if the validation loss does not decrease for 3 consecutive epochs.

- **verbose**: This parameter controls the verbosity mode. If set to `1`, it prints messages about the early stopping condition being triggered. If set to `0`, it operates silently. 

- **restore_best_weights**: When set to `True`, this parameter restores the model weights from the epoch with the lowest validation loss. This ensures that the model's performance is not adversely affected by training beyond the point of early stopping.

6. **Train the Model with Early Stopping**:
```python
# Train the model with early stopping
history_cereal = model.fit(cereal_X_train, cereal_y_train, epochs=100, batch_size=64,
                           validation_data=(cereal_X_test, cereal_y_test),
                           callbacks=[early_stopping], verbose=1)
```

7. **Visualize Training History**:
```python
# Plot training history
plt.plot(history_cereal.history['loss'], label='Cereal Train Loss')
plt.plot(history_cereal.history['val_loss'], label='Cereal Validation Loss')
plt.title('Training and Validation Loss for Cereal Dataset')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()
```

> Here's the [complete code](./lab1part2.py)

**Does the early stopping callback help against overfitting?**

The early stopping callback is an effective technique for combating overfitting. Here's how it helps:

1. **Prevents Overfitting**: Early stopping prevents overfitting by stopping the training process when the model's performance on the validation set starts to degrade. This ensures that the model does not continue to learn the idiosyncrasies of the training data at the expense of generalization to unseen data.

2. **Promotes Generalization**: By stopping training at an optimal point, early stopping promotes better generalization of the model to unseen data. It helps strike a balance between fitting the training data well and avoiding overfitting.

3. **Saves Computational Resources**: Early stopping saves computational resources by terminating training early when further iterations are unlikely to yield significant improvements. This is particularly useful when training deep neural networks that can be computationally intensive.

Overall, the early stopping callback is a valuable tool in the machine learning practitioner's arsenal for preventing overfitting and improving the generalization performance of neural network models.

### Part 3/3: discussions

- Refer to this code snippet:

```python
model = Sequential()
model.add(Dense(64, activation='relu', input_shape=(cereal_X_train.shape[1],)))
model.add(Dense(32, activation='relu'))
model.add(Dense(1))
```

1. What type of neural network architecture is being defined in the code snippet?
2. What does the `Sequential()` function do in this code?
3. What is the purpose of the `add()` method in this code?
4. How many layers does the neural network have in total?
5. What activation function is used for the first two layers, and why might this be a common choice?
6. What is the purpose of specifying `input_shape` in the first `Dense` layer?
7. How many neurons are there in each layer, and why might these specific numbers have been chosen?
8. What does the last `Dense` layer represent in the neural network?
9. What does the absence of an activation function in the last `Dense` layer indicate about the network's output?
10. How would you describe the overall architecture and flow of information in this neural network model?



<details>
<summary>Sample answers</summary>

1. The code snippet defines a feedforward neural network architecture.
2. The `Sequential()` function initializes a sequential model, which allows us to define the neural network as a sequence of layers.
3. The `add()` method adds layers to the neural network sequentially.
4. The neural network has three layers in total.
5. The ReLU (Rectified Linear Activation) function is used for the first two layers. ReLU is a common choice for hidden layers in neural networks because it introduces non-linearity and helps alleviate the vanishing gradient problem.
6. The `input_shape` parameter in the first `Dense` layer specifies the shape of the input data. In this case, it defines the number of features in the input data.
7. There are 64 neurons in the first `Dense` layer and 32 neurons in the second `Dense` layer. These specific numbers might have been chosen based on the complexity of the problem and experimentation.
8. The last `Dense` layer represents the output layer of the neural network.
9. The absence of an activation function in the last `Dense` layer indicates that it is a linear layer. This means that the output of the network will be a linear combination of the inputs, which is suitable for regression tasks.
10. The overall architecture consists of an input layer, followed by two hidden layers with ReLU activation functions, and an output layer. Information flows sequentially from the input layer through the hidden layers to the output layer, with each layer performing transformations on the input data.


</details>