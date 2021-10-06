import numpy as np
import pandas as pd

class Perceptron:

  # Basic initialization function
  def __init__(self,eta,epochs): # eta = learning rate . This is basic initialization
    # weight initialization
    self.weights=np.random.randn(3)*1e-4 # Here weights are taken as 3 values since matrix required is 3x1 and these weights are starting from small values
    print(f"initial weights before training:{self.weights}")
    # Learning rate initialization
    self.eta=eta
    # Epochs initialization
    self.epochs=epochs
    
  # Activation function 
  def activationFunction(self, inputs, weights):
    z = np.dot(inputs, weights) # z=W*X
    return np.where(z>0, 1, 0)
    
  # Training model with X and y
  def fit(self, X, y):
    self.X=X
    self.y=y
    # X with bias is X matrix concatenated with -1 matrix
    X_with_bias=np.c_[self.X, -np.ones((len(self.X),1))] # here bias=-np.ones(len(self.X),1) is an array of -1 in the shape 4x1
    print(f"X with bias:\n{X_with_bias}")
    # Creating loops for each epoch
    for epoch in range(self.epochs):
      print(f"for epoch:\n{epoch}")
      print("--"*10)
      # Finding predicted output
      y_hat= self.activationFunction(X_with_bias, self.weights) # Forward propagation
      print(f"Predicted value after forward pass:\n{y_hat}")
      # Calculating error 
      self.error= self.y - y_hat
      print(f"error:\n{self.error}")
      #updating weights
      self.weights= self.weights + self.eta * np.dot(X_with_bias.T, self.error) # Backward Propagation
      print(f"Updated weights after {epoch}/{self.epochs}:\n {self.weights}")
      print("######"*10)
    
  # Predicting model using X
  def predict(self, X):
    X_with_bias=np.c_[X, -np.ones((len(X),1))]
    return self.activationFunction(X_with_bias, self.weights)

  # Function for calculating loss
  def total_loss(self):
    total_loss=np.sum(self.error)
    print(f"total loss:\n{total_loss}")
    return total_loss
  