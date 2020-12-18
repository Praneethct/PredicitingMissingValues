import numpy as np
import math
from statistics import mean

class NeuralNetwork():
    
    def __init__(self, X, y, layers = [5, 2], learning_rate = 0.01, epochs = 5, batch_size = 250):
        self.weights = None
        self.X = X
        self.y = y
        self.layers = layers
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        

    def weights_init(self):
        
        self.weights = []
        for i in range(len(self.layers)):
            temp = []
            for j in range(self.layers[i]):
                
                if i == 0:
                    temp.append(np.random.normal(0,0.4, size = 1 + self.X.shape[1]))
                
                else:
                    temp.append(np.random.normal(0,0.4, size = 1 + self.layers[i-1]))
            self.weights.append(temp)
        
        self.outputLayerWeights =  np.random.normal(0,0.4, size = 1 + self.layers[-1])
    
    def gradient_init(self):
        self.gradient = []
        for i in range(len(self.layers)):
            temp = []
            for j in range(self.layers[i]):
                
                if i == 0:
                    temp.append(np.zeros(1 + self.X.shape[1]))
                
                else:
                    temp.append(np.zeros(1 + self.layers[i-1]))
            self.gradient.append(temp)
        self.gradientOutputLayer = [0] * len(self.outputLayerWeights)
    
    def sigmoid(self,x):
        if x < 0:
            return 1 - 1 / (1 + math.exp(x))
        else:
            return 1 / (1 + math.exp(-x))
    
    def output_activation(self,x):
        return x
    
    def derivate(self,x):  
        return self.sigmoid(x) *(1 - self.sigmoid(x))

    def linear_derivative(self, x):
        return 1.0
    
    def loss(self, X, y):
        pred= []
        for i in X:
            pred.append(self.feed_forward_network(i))
        return mean([(a_i - b_i)**2 for a_i, b_i in zip(pred, y)])


    def predict(self,X):
        pred = []
        for i in X:
            pred.append(self.feed_forward_network(i))
        return pred

    def predict_row(self,X):
        out = self.feed_forward_network(X)
        return out
    
    def loss(self, pred, actual):
        return 2.0 * (pred- actual)

    def feed_forward_network(self, x):
        self.x = np.append(x, 1.0)
        self.out = []
        for i in range(len(self.layers) + 1):
            outputFromCurrLayer = []
            
            if i == 0:
                for j in range(self.layers[i]):
                    z = self.sigmoid(np.dot(self.weights[i][j],self.x))
                    outputFromCurrLayer.append(z)
                temp = outputFromCurrLayer.copy()
                self.out.append(temp)
                outputFromCurrLayer.append(1.0)
                outputFromPrevLayer = outputFromCurrLayer.copy()
            
            elif i == len(self.layers):
                return self.output_activation(np.dot(self.outputLayerWeights,outputFromPrevLayer))
            
            else:
                for j in range(self.layers[i]):
                    z = self.sigmoid(np.dot(self.weights[i][j],outputFromPrevLayer))
                    outputFromCurrLayer.append(z)
                temp = outputFromCurrLayer.copy()
                self.out.append(temp)
                outputFromCurrLayer.append(1.0)
                outputFromPrevLayer = outputFromCurrLayer.copy()


    def update_delta(self, delta):
        for l in reversed(range(len(self.layers))):
            delta_forward = delta.copy()
            delta = [0] * self.layers[l-1]
            
            if l == 0 :
                for j in range(self.layers[l]):
                    der_layer = self.derivate(np.dot(self.x , self.weights[l][j]))
                    for i in range(len(self.weights[l][j])):
                        if i == len(self.weights[l][j]) - 1:
                            self.gradient[l][j][i] = self.gradient[l][j][i] +  (delta_forward[j] * der_layer * 1.0)
                        else :
                            self.gradient[l][j][i] = self.gradient[l][j][i] +   (delta_forward[j] * der_layer * self.x[i])
            
            else :
                for j in range(self.layers[l]):
                    der_layer = self.derivate(np.dot(np.append(self.out[l - 1], 1.0) , self.weights[l][j]))
                    for i in range(len(self.weights[l][j])):
                        if i == len(self.weights[l][j]) - 1:
                            self.gradient[l][j][i] = self.gradient[l][j][i] +  (delta_forward[j] * der_layer * 1.0)
                        else :
                            d = delta_forward[j] * der_layer * self.weights[l][j][i]
                            delta[i] = delta[i] + d
                            self.gradient[l][j][i] = self.gradient[l][j][i] + (delta_forward[j] * der_layer * self.out[l - 1][i])
        return delta


    def back_propagation(self, pred, actual):
        
        delta = []
        der_outter_layer = self.linear_derivative(np.dot(np.append(self.out[len(self.layers) -1], 1.0) , self.outputLayerWeights))
        for i in range(len(self.outputLayerWeights)):
            if i == len(self.outputLayerWeights) - 1:
                self.gradientOutputLayer[i] = self.gradientOutputLayer[i] + (self.loss(pred, actual) * der_outter_layer * 1)
            else :
                d = self.loss(pred, actual) * der_outter_layer * self.outputLayerWeights[i]
                self.gradientOutputLayer[i] = self.gradientOutputLayer[i] + (self.loss(pred, actual) * der_outter_layer * self.out[len(self.layers) -1][i])
                delta.append(d) 
        delta = update_delta(delta)
        
        
    
    def update_weights(self, n):
        for i in range(len(self.outputLayerWeights)):
            self.outputLayerWeights[i] = self.outputLayerWeights[i] - (self.learning_rate *  self.gradientOutputLayer[i]/n)
        
        for l in reversed(range(len(self.layers))):
            for j in range(self.layers[l]):
                for i in range(len(self.weights[l][j])):
                    self.weights[l][j][i] = self.weights[l][j][i] - (self.learning_rate *  self.gradient[l][j][i] /n)

    
    def fit(self,X,y,X_val = None, Y_val = None):
        self.X = X
        self.y = y
        self.weights_init()
        self.gradient_init()
        i = 0
        loss_val_old = -1
        tol_count = 0
        while i < self.epochs:
            for j in range(len(X)):
                if j%self.batch_size ==0 and j != 0 or j == len(X) -1:
                    if j == len(X) -1:
                        self.update_weights(j%self.batch_size)
                    else:
                        self.update_weights(self.batch_size)
                    self.gradient_init()
                    p = self.feed_forward_network(X[j])
                    self.back_propagation(p,y[j])
                else:
                    p = self.feed_forward_network(X[j])
                    self.back_propagation(p,y[j])
            i = i+1