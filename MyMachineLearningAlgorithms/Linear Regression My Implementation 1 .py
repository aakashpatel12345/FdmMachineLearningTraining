# -*- coding: utf-8 -*-
"""
Created on Fri Feb  7 10:07:51 2020

@author: aakash.patel
"""

import pandas as pd
import numpy as np

class AakashLinearRegression():
    
    
    def __init__(self):
        self.coefficients = np.array
        
    
    def gradient_descent(self,x,y,learning_rate,num_iterations):
        sample_size = len(x)
        constant_column = np.array(np.ones((sample_size,1)).astype(int))
        x = np.append(arr = constant_column, values = x ,axis = 1)
        num_of_coefs = x.shape[1] # includes y-intercept
        matrix_of_coef = np.array(np.zeros((num_of_coefs, 1)))

        for i in range(num_iterations):
            y_pred = np.dot(x,matrix_of_coef) #why doesn't return a scalar - check documentation
       
            cost = (1/(2*sample_size)) *  sum((y-y_pred)** 2)
       
            matrix_of_gradients_of_coefs = np.array(np.zeros((num_of_coefs, 1)))
            for j in range(num_of_coefs):
                difference = y-y_pred
                matrix_of_gradients_of_coefs[j,0] = -(1/sample_size) * sum(np.dot(x[:,j],difference))
                matrix_of_coef[j,0] = matrix_of_coef[j,0] - (learning_rate*matrix_of_gradients_of_coefs[j,0])
        self.coefficients = matrix_of_coef   
        print ("The gradient of term 1 in this iteration is {}".format(matrix_of_coef[1,0]))
        print ("The gradient of term 2 in this iteration is {}".format(matrix_of_coef[2,0]))
        print("The y_intercept in this iteration is {}".format(matrix_of_coef[0,0]))
        print("This is iteration number {} and it costs {}".format(i, cost))
        print('\n')
        return self.coefficients
    
    def fit(self, x_train, y_train, learning_rate=0.0000001, num_iterations = 10):
        self.x_train = x_train
        self.y_train = y_train
        self.coefficients = self.gradient_descent(x_train,y_train,learning_rate,num_iterations)
        
    
    def display_coefficients(self):
        print ("The y intercept is ",self.coefficients[0])
        for i in range (0,(len(self.coefficients)-1)):
            print ("Coefficient", str(i+1) ,"  is ",self.coefficients[i+1])    
    
    def predict(self, x_test):
        #do y = mx + c 
        #adds column of ones onto the intercept
        sample_size = len(x_test)
        constant_column = np.array(np.ones((sample_size,1)).astype(int))
        x_test = np.append(arr = constant_column, values = x_test ,axis = 1)
        
        y_predicted = np.zeros(x_test.shape[0])
        j=0
        for row in x_test: 
            #print (x_test[0][0])
            for i in range(0, x_test.shape[1]):
                y_predicted[j] +=  (self.coefficients[i]*row[i])
            
            j += 1
        
        print(y_predicted)
        return y_predicted
    
    
    


data = pd.read_csv("../Datasets/admissions.csv")

x=np.array(data[["TOEFL Score" , "GRE Score"]])
y=np.array(data[["CGPA"]])
       
ALR = AakashLinearRegression()

ALR.fit(x,y)
#ALR.display_coefficients()
y_pred = ALR.predict(x)      


'''
    def predict(self, x_test):
        #Needs to return a list of predictions
        
        predictions = []
        for row in x_test:
            label = self.closest(row)
            predictions.append(label)
        return predictions
    
    def closest(self, row):
        best_distance = euclideanDistance(row, self.x_train[0])
        best_index = 0
        for i in range(1, len(self.x_train)):
            distance = euclideanDistance(row, self.x_train[i])
            if distance < best_distance:
                best_index = i
                best_distance = distance
        return self.y_train[best_index]
        '''
