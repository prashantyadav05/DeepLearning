# Deep Learning
## Multilayer Perceptrons
### Neurons
The building block for neural networks are artificial neurons. These are simple computational units that have weighted input signals and produce an output signal using an activation function.
#### Neuron Weights
You may be familiar with linear regression, in which case the weights on the inputs are very much like the coefficients used in a regression equation. Like linear regression, each neuron also has a bias which can be thought of as an input that always has the value 1.0 and it too must be weighted. For example, a neuron may have two inputs in which case it requires three weights. One for each input and one for the bias. Weights are often initialized to small random values, such as values in the range 0 to 0.3, although more complex initialization schemes can be used. Like linear regression, larger weights indicate increased complexity and fragility of the model. It is desirable to keep weights in the network small and regularization techniques can be used.
#### Activation
The weighted inputs are summed and passed through an activation function, sometimes called a transfer function. An activation function is a simple mapping of summed weighted input to the output of the neuron. It is called an activation function because it governs the threshold at which the neuron is activated and the strength of the output signal. Historically simple step activation functions were used where if the summed input was above a threshold, for example 0.5, then the neuron would output a value of 1.0, otherwise it would output a 0.0. Traditionally nonlinear activation functions are used. This allows the network to combine the inputs in more complex ways and in turn provide a richer capability in the functions they can model. Nonlinear functions like the logistic function also called the sigmoid function were used that output a value between 0 and 1 with an s-shaped distribution, and the hyperbolic tangent function also called Tanh that outputs the same distribution over the range -1 to +1. More recently the rectifier activation function has been shown to provide better results.
#### Define model
Models in Keras are defined as a sequence of layers. We create a Sequential model and add layers one at a time until we are happy with our network topology. The first thing to get right is to ensure the input layer has the right number of inputs. This can be specified when creating the first layer with the input dim argument and setting it to 8 for the 8 input variables. Fully connected layers are defined using the Dense class. We can specify the number of neurons in the layer as the first argument and specify the activation function using the activation argument.
```python
model = Sequential()
model.add(Dense(12, input_dim=8, kernel_initializer="glorot_uniform", bias_initializer="zeros", activation="relu"))
model.add(Dense(8, kernel_initializer="glorot_uniform", bias_initializer="zeros", activation="relu"))
model.add(Dense(1, kernel_initializer="glorot_uniform", bias_initializer="zeros", activation="sigmoid"))
```
#### Compile Model
Specifiying the loss function, optimizers & metrices
```python
model.compile(loss= binary_crossentropy , optimizer= adam , metrics=[ accuracy ])
```
#### Fit Model
#### Evaluate Model

### Evaluate The Performance of Deep Learning Models
#### Automatic Verification Dataset
```python
model.fit(X, Y, validation_split=0.33, nb_epoch=150, batch_size=10)
```
#### Manual Verification Dataset
In this example we use the handy train test split() function from the Python scikit-learn machine learning library to separate our data into a training and test dataset. We use 67% for training and the remaining 33% of the data for validation. The validation dataset can be specified to the fit() function in Keras by the validation data argument. It takes a tuple of the input and output datasets.
```python
model.fit(X_train, y_train, validation_data=(X_test, y_test), nb_epoch=150, batch_size=10)
```
#### Manual k-Fold Cross Validation
### Use Keras Models With Scikit-Learn For General Machine Learning
#### Evaluate Models with Cross Validation
#### Grid Search Deep Learning Model Parameters
### Project: Multiclass Classification Of Flower Species
### Project: Binary Classification Of Sonar Returns:
### Project: Regression Of Boston House Prices:
## Advanced Multilayer Perceptrons
## Convolutional Neural Networks
## Recurrent Neural Networks
