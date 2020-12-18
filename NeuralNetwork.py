import numpy as np
import math
from statistics import mean

class NeuralNetwork():
    
    def __init__(self, X = None , y = None, layers = [5, 2], learning_rate = 0.01, 
                epochs = 5, tol = 0.1, batch_size = 250):
        self.weights = None
        self.X = X
        self.y = y
        self.activationHidden = self.sigmoid
        self.method = method
        self.activationOut = self.linear
        self.derivate_out = self.linear_der
        self.out_class = 'Linear'
        self.layers = layers
        self.derivate_rest = self.sigmoid_der
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.tol = tol
        self.batch_size = batch_size
        

    def weightsInitialisation(self):
        #Initialising a numpy array of dim(hiddenlayers, neurons) to store weights
        self.weights = []
        for i in range(len(self.layers)):
            temp = []
            for j in range(self.layers[i]):
                #first hidden layer
                if i == 0:
                    temp.append(np.random.normal(0,0.4, size = 1 + self.X.shape[1]))
                #rest hidden layers
                else:
                    temp.append(np.random.normal(0,0.4, size = 1 + self.layers[i-1]))
            self.weights.append(temp)
        #Weights for the final output layer
        self.outputLayerWeights =  np.random.normal(0,0.4, size = 1 + self.layers[-1])
    
    def gradientInitialisation(self):
        self.gradient = []
        for i in range(len(self.layers)):
            temp = []
            for j in range(self.layers[i]):
                #first hidden layer
                if i == 0:
                    temp.append(np.zeros(1 + self.X.shape[1]))
                #rest hidden layers
                else:
                    temp.append(np.zeros(1 + self.layers[i-1]))
            self.gradient.append(temp)
        self.gradientOutputLayer = [0] * len(self.outputLayerWeights)
    
    def sigmoid(self,x):
        if x < 0:
            return 1 - 1 / (1 + math.exp(x))
        else:
            return 1 / (1 + math.exp(-x))
    
    def linear(self,x):
        return x
    
    def sigmoid_der(self,x):  
        return self.sigmoid(x) *(1 - self.sigmoid(x))

    def linear_der(self, x):
        return 1.0
    
    def squareErrorLoss(self,x,y):
        return (self.feedForward(x) - y)**2
    
    def error(self, X, y):
        pred= []
        for i in X:
            pred.append(self.feedForward(i))
        return mean([(a_i - b_i)**2 for a_i, b_i in zip(pred, y)])


    def predict(self,X):
        pred = []
        for i in X:
            pred.append(self.feedForward(i))
        return pred

    def predict_row(self,X):
        out = self.feedForward(X)
        return out
    
    def loss(self, pred, actual):
        return 2.0 * (pred- actual)

    def feedForward(self, x):
        self.x = np.append(x, 1.0)
        self.out = []
        for i in range(len(self.layers) + 1):
            outputFromCurrLayer = []
            #For first Layer
            if i == 0:
                for j in range(self.layers[i]):
                    z = self.activationHidden(np.dot(self.weights[i][j],self.x))
                    outputFromCurrLayer.append(z)
                temp = outputFromCurrLayer.copy()
                self.out.append(temp)
                outputFromCurrLayer.append(1.0)
                outputFromPrevLayer = outputFromCurrLayer.copy()
            #Output Layer
            elif i == len(self.layers):
                return self.activationOut(np.dot(self.outputLayerWeights,outputFromPrevLayer))
            #Rest all Layers
            else:
                for j in range(self.layers[i]):
                    z = self.activationHidden(np.dot(self.weights[i][j],outputFromPrevLayer))
                    outputFromCurrLayer.append(z)
                temp = outputFromCurrLayer.copy()
                self.out.append(temp)
                outputFromCurrLayer.append(1.0)
                outputFromPrevLayer = outputFromCurrLayer.copy()

    def backProp(self, pred, actual):
        #Weight updation for Output Layer
        delta = []
        der_outter_layer = self.derivate_out(np.dot(np.append(self.out[len(self.layers) -1], 1.0) , self.outputLayerWeights))
        for i in range(len(self.outputLayerWeights)):
            if i == len(self.outputLayerWeights) - 1:
                self.gradientOutputLayer[i] = self.gradientOutputLayer[i] + (self.loss(pred, actual) * der_outter_layer * 1)
            else :
                d = self.loss(pred, actual) * der_outter_layer * self.outputLayerWeights[i]
                self.gradientOutputLayer[i] = self.gradientOutputLayer[i] + (self.loss(pred, actual) * der_outter_layer * self.out[len(self.layers) -1][i])
                delta.append(d) 
        #For all other Layers
        for l in reversed(range(len(self.layers))):
            delta_forward = delta.copy()
            delta = [0] * self.layers[l-1]
            #For the first layer
            if l == 0 :
                for j in range(self.layers[l]):
                    der_layer = self.derivate_rest(np.dot(self.x , self.weights[l][j]))
                    for i in range(len(self.weights[l][j])):
                        if i == len(self.weights[l][j]) - 1:
                            self.gradient[l][j][i] = self.gradient[l][j][i] +  (delta_forward[j] * der_layer * 1.0)
                        else :
                            self.gradient[l][j][i] = self.gradient[l][j][i] +   (delta_forward[j] * der_layer * self.x[i])
            #Rest all the layers
            else :
                for j in range(self.layers[l]):
                    der_layer = self.derivate_rest(np.dot(np.append(self.out[l - 1], 1.0) , self.weights[l][j]))
                    for i in range(len(self.weights[l][j])):
                        if i == len(self.weights[l][j]) - 1:
                            self.gradient[l][j][i] = self.gradient[l][j][i] +  (delta_forward[j] * der_layer * 1.0)
                        else :
                            d = delta_forward[j] * der_layer * self.weights[l][j][i]
                            delta[i] = delta[i] + d
                            self.gradient[l][j][i] = self.gradient[l][j][i] + (delta_forward[j] * der_layer * self.out[l - 1][i])
    
    def updateWeights(self, n):
        for i in range(len(self.outputLayerWeights)):
            self.outputLayerWeights[i] = self.outputLayerWeights[i] - (self.learning_rate *  self.gradientOutputLayer[i]/n)
        #For all other Layers
        for l in reversed(range(len(self.layers))):
            for j in range(self.layers[l]):
                for i in range(len(self.weights[l][j])):
                    self.weights[l][j][i] = self.weights[l][j][i] - (self.learning_rate *  self.gradient[l][j][i] /n)

    
    def fit(self,X,y,X_val = None, Y_val = None):
        self.X = X
        self.y = y
        self.weightsInitialisation()
        self.gradientInitialisation()
        i = 0
        error_val_old = -1
        tol_count = 0
        while i < self.epochs:
            for j in range(len(X)):
                if j%self.batch_size ==0 and j != 0 or j == len(X) -1:
                    if j == len(X) -1:
                        self.updateWeights(j%self.batch_size)
                    else:
                        self.updateWeights(self.batch_size)
                    self.gradientInitialisation()
                    p = self.feedForward(X[j])
                    self.backProp(p,y[j])
                else:
                    p = self.feedForward(X[j])
                    self.backProp(p,y[j])
            i = i+1