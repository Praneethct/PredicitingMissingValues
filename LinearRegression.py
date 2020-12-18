import numpy as np

class LinearRegression():
    
    def __init__(self, method = None, _lambda = 0.1):
        self.method = method
        self._lambda = _lambda
        
    def generate_data(self, data, target):
        data['Bias'] = 1
        self.X = data.drop(target, axis = 1).values
        data.drop('Bias', axis = 1, inplace = True)
        self.Y = data[target].values

    def update_weights(self):
        for i in range(len(self.weights)):
            denominator = np.matmul(self.X[:,i].T, self.X[:,i])
            pred = (self.Y - np.matmul(self.X,self.weights))
            upper_bound = (np.matmul((-1 * self.X[:,i].T), pred) + (self._lambda/2))/ denominator
            lower_bound = (np.matmul((-1 * self.X[:,i].T), pred) - (self._lambda/2))/ denominator
            if  upper_bound < self.weights[i] :
                self.weights[i] = self.weights[i] + (np.matmul((self.X[:,i].T),pred) - (self._lambda/2))/ denominator
            elif lower_bound > self.weights[i] :
                self.weights[i] = self.weights[i] +(np.matmul((self.X[:,i].T),pred) + (self._lambda/2))/ denominator
            else:
                self.weights[i] = 0
        
    def fit(self, data, target):
        self.data = data
        self.target = target
        self.generate_data(self.data, self.target)
        
        if self.method == "Linear" or not self.method:
            self.weights = np.matmul(np.linalg.inv(np.matmul(self.X.T, self.X)), np.matmul(self.X.T, self.Y))
        elif self.method == "Ridge" :
            self.weights = np.matmul(np.linalg.inv(np.matmul(self.X.T, self.X) + (self._lambda * np.identity(self.X.shape[1]))), np.matmul(self.X.T, self.Y))
        elif self.method == "Lasso":
            self.weights = [0 for i in range(self.X.shape[1])]
            while True:
                old_weights = self.weights.copy()
                update_weights()
                updates = [k - l for k, l in zip(old_weights, self.weights)]
                if max(updates) < 1e-2 and abs(min(updates)) < 1e-2:
                    break


    def predict_row(self, row):
        y_pred = np.sum(np.multiply(self.weights, row))
        return y_pred
        
    def predict(self,test):
        test['bias'] = 1
        y_predicted = []
        for index,row in test.iterrows():
            y_predicted.append(self.predict_row(row))
        return y_predicted
    
    def training_error(self):
        predicted_y = self.predict(self.data.drop(self.target, axis = 1))
        mse = []
        for i in range(len(predicted_y)):
            err = ((predicted_y[i] - self.Y[i])**2)
            mse.append(err)
        return sum(mse)/len(mse)
    
    def error(self, test):
        test = test.reset_index(drop = True)
        predicted_y = self.predict(test.drop(self.target, axis = 1))
        mse = []
        for i in range(len(predicted_y)):
            err = ((predicted_y[i] - test[self.target][i])**2)
            mse.append(err)
        return sum(mse)/len(mse)


class min_max_scaler():
    def __init__ (self, data, y = None):
        self.data = data
        self.y = y
        self.min_params = {}
        self.max_params = {}
        self.single = {}
        for i in data.columns:
            if (i != self.y) :
                if (len(data[i].unique())>1):
                    self.min_params[i] = min(data[i])
                    self.max_params[i] = max(data[i])
                else:
                    self.single[i] = 0.001
    
    def transform(self, data):
        copy = data.copy()
        for i in data.columns:
            if ((i != self.y) and (i not in self.single.keys())):
                copy[i] = data[i].apply(lambda x : (x- self.min_params[i]) / (self.max_params[i] - self.min_params[i]))
            elif (i in self.single.keys()):
                copy[i] = self.single[i]
            else:
                copy[i] = data[i]
        return copy