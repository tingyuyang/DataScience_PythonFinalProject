from __future__ import division
import csv
import math
import numpy as np
from matplotlib import pyplot as plt
import random
from collections import Counter
from linear_algebra import distance, vector_subtract, scalar_multiply
import math, random
#this is the code for conpute the regression model. However, the result is not satisfying as we expected. (the code mistakes appear on line 153)


#arranging data section
with open('day.csv', 'r') as f:
   reader = csv.reader(f, delimiter=',')
   rows = [r for r in reader]
   
date = [row[1] for row in rows]
season = [float(row[2])for row in rows]
year = [float(row[3])for row in rows]
weathersit = [float(row[8])for row in rows]
temp = [float(row[9])for row in rows]
atemp = [float(row[10])for row in rows]
hum = [float(row[11])for row in rows]
windspeed = [float(row[12])for row in rows]
casual = [float(row[13])for row in rows]
registered = [float(row[14])for row in rows]
cnt = [float(row[15])for row in rows]
cntlow=[] #when the temperature is below 10 celsius=cold
for i in range(0, 731):
    if temp[i] <=0.2438:
        cntlow.append(cnt[i])
cnthigh=[] # when the temp is above 20 celsius=hot
for i in range(0, 731):
    if temp[i] >=0.4878:
        cnthigh.append(cnt[i])
cntnice=[]#when the weather the pleasant
for i in range(0, 731):
    if temp[i] >0.2439 and temp[i] <0.4878 :
        cntnice.append(cnt[i])

onelist=[]
for i in range(0,731):
    onelist.append(1)
    
#function section
def dot(v,w):
    return sum(v_i * w_i for v_i,w_i in zip(v,w))

def predict(x_i,beta):
    return dot(x_i,beta)
    
def error(x_i,y_i,beta):
    return y_i - predict(x_i,beta)
    
def squared_error(x_i,y_i,beta):
    return error(x_i,y_i,beta)**2
    
def squared_error_gradient(x_i,y_i,beta):
    return[-2 * x_ij * error(x_i,y_i,beta) for x_ij in x_i]

def step(v, direction, step_size):
    """move step_size in the direction from v"""
    return [v_i + step_size * direction_i
            for v_i, direction_i in zip(v, direction)]
            
def safe(f):
    """define a new function that wraps f and return it"""
    def safe_f(*args, **kwargs):
        try:
            return f(*args, **kwargs)
        except:
            return float('inf')         # this means "infinity" in Python
    return safe_f

def in_random_order(data):
    """generator that returns the elements of data in random order"""
    indexes = [i for i, _ in enumerate(data)]  # create a list of indexes
    random.shuffle(indexes)                    # shuffle them
    for i in indexes:                          # return the data in that order
        yield data[i]


def minimize(target_fn, gradient_fn, x, y, theta_0, alpha_0=0.01):

    data = zip(x, y)
    theta = theta_0                             # initial guess
    alpha = alpha_0                             # initial step size
    min_theta, min_value = None, float("inf")   # the minimum so far
    iterations_with_no_improvement = 0
    
    # if we ever go 100 iterations with no improvement, stop
    while iterations_with_no_improvement < 100:
        value = sum( target_fn(x_i, y_i, theta) for x_i, y_i in data )

        if value < min_value:
            # if we've found a new minimum, remember it
            # and go back to the original step size
            min_theta, min_value = theta, value
            iterations_with_no_improvement = 0
            alpha = alpha_0
        else:
            # otherwise we're not improving, so try shrinking the step size
            iterations_with_no_improvement += 1
            alpha *= 0.9

        # and take a gradient step for each of the data points        
        for x_i, y_i in in_random_order(data):
            gradient_i = gradient_fn(x_i, y_i, theta)
            theta = vector_subtract(theta, scalar_multiply(alpha, gradient_i))
            
    return min_theta


def estimate_beta(x, y):
    beta_initial = [random.random() for x_i in x[0]]
    return minimize(squared_error, 
                    squared_error_gradient, 
                    x, y, 
                    beta_initial, 
                    0.001)

#in order to compute the data for the first 600 data points.-list1                   
cnt1=cnt[0:600]
onelist1=onelist[0:600]
weathersit1=weathersit[0:600]
atemp1 = atemp[0:600]
hum1 = hum[0:600]
windspeed1 = windspeed[0:600]

#the last 131 data points-list2
cnt2=cnt[601:731]
onelist2=onelist[601:731]
weathersit2=weathersit[601:731]
atemp2 = atemp[601:731]
hum2 = hum[601:731]
windspeed2 = windspeed[601:731]

#beta compute                    
wholedata=list(zip(cnt1,onelist1,weathersit1,atemp1,hum1,windspeed1))# wil be the xi here
random.shuffle(wholedata) # to random the data set to make it more accurate
y = [row[0] for row in wholedata] #y will be the cnt here
x = [row[1:] for row in wholedata] # x will be 1, weathersit,atemp,hum....
random.seed(0)
beta = estimate_beta(x,y)

#the multiple regression model
print ("Total renters = {}+ Weather Situation*{} + Feeling temperature *{} + Humidity* {} + Wind Speed *{} ".format(beta[0],beta[1],beta[2],beta[3],beta[4]))

#compare list2 for the function
#5 data sets we choosed randomly, we use random function. so they give us 4 random figures between 0 to 130: 58,29,97,124,3
y1=beta[0]+ weathersit2[58]*beta[1] + atemp2[58]*beta[2]+ hum2[58]*beta[3] + windspeed2[58]*beta[4] #58 should be 58+601=659days
####something went wrong, when I try to print the result, the prediction is totally off
print("Our prediction of total bike renters for Day 659 days is {}, and the actual bike renters of that day is {}".format(y1,cnt2[58]))

y2=beta[0]+ weathersit[29]*beta[1] + atemp[29]*beta[2]+ hum[29]*beta[3] + windspeed[29]*beta[4] #630days
#print("Our prediction of total bike renters for Day 630 days is {}, and the actual bike renters of that day is {}".format(y2,cnt2[29]))




