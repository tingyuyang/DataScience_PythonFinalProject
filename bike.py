# -*- coding: utf-8 -*-
"""
Created on Wed Apr 20 12:24:50 2016

@author: tingyuyang
"""
#10/41 = 0.2439
#20/41 = 0.4878
import csv
import math
import numpy as np
from matplotlib import pyplot as plt

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


#function section 
def mean(x):
    return sum(x) / len(x)

def median(v):
    """finds the 'middle-most' value of v"""
    n = len(v)
    sorted_v = sorted(v)
    midpoint = n // 2

    if n % 2 == 1:
        # if odd, return the middle value
        return sorted_v[midpoint]
    else:
        # if even, return the average of the middle values
        lo = midpoint - 1
        hi = midpoint
        return (sorted_v[lo] + sorted_v[hi]) / 2
        
def de_mean(x):
    """translate x by subtracting its mean (so the result has mean 0)"""
    x_bar = mean(x)
    return [x_i - x_bar for x_i in x]
    
def sum_of_squares(v):
    """v_1 * v_1 + ... + v_n * v_n"""
    return dot(v, v)

def variance(x):
    """assumes x has at least two elements"""
    n = len(x)
    deviations = de_mean(x)
    return sum_of_squares(deviations) / (n - 1)

def standard_deviation(x):
    return math.sqrt(variance(x))

def dot(v, w):
    """v_1 * w_1 + ... + v_n * w_n"""
    return sum(v_i * w_i for v_i, w_i in zip(v, w))

def covariance(x, y):
    n = len(x)
    return dot(de_mean(x), de_mean(y)) / (n - 1)

def correlation(x, y):
    stdev_x = standard_deviation(x)
    stdev_y = standard_deviation(y)
    if stdev_x > 0 and stdev_y > 0:
        return covariance(x, y) / stdev_x / stdev_y
    else:
        return 0 # if no variation, correlation is zero
 
def prob(x,y):
    return len(x) / len(y)


#print statistic figures
print ("Mean of total renters is {}".format(mean(cnt)))
print ("Mean of total renters when the day is cold is{}".format(mean(cntlow)))
print ("Mean of total renters when the day is hot is {}".format(mean(cnthigh)))
print ("Mean of total renters when the day is pleasant is {}".format(mean(cntnice)))
print(" ")
print ("Median of all the renters is {}".format(median(cnt)))
print ("Median of all the renters when the day is cold is {}".format(median(cntlow)))
print ("Median of all the renters when the day is hot is {}".format(median(cnthigh)))
print ("Median of all the renters when the day is pleasant is {}".format(median(cntnice)))
print(" ")
print ("Standard Deviation of all the renters is {}".format(standard_deviation(cnt)))
print ("Standard Deviation of all the renters when the day is cold is {}".format(standard_deviation(cntlow)))
print ("Standard Deviation of all the renters when the day is hot is {}".format(standard_deviation(cnthigh)))
print ("Standard Deviation of all the renters when the day is pleasant is {}".format(standard_deviation(cntnice)))
print(" ")


#plot the figures
plt.scatter (atemp,cnt)
plt.xlabel('Feeling Temperature('
       + u'\N{DEGREE SIGN}' + 'C)')
plt.ylabel('Total Renters')
fit = np.polyfit(atemp,cnt,1)
fit_fn = np.poly1d(fit) 
plt.plot(atemp,cnt, 'yo', atemp, fit_fn(atemp), '--k')
plt.show()
print ("Correlation of feeling temperature and total bike renters is {}".format(correlation(atemp,cnt)))

plt.scatter (temp,cnt)
plt.xlabel('Temperature('
       + u'\N{DEGREE SIGN}' + 'C)')
plt.ylabel('Total Renters')
fit = np.polyfit(temp,cnt,1)
fit_fn = np.poly1d(fit) 
plt.plot(temp,cnt, 'yo', temp, fit_fn(temp), '--k')
plt.show()
print ("Correlation of temperature and total bike renters is {}".format(correlation(temp,cnt)))

plt.scatter (atemp,cnt)
plt.scatter (temp,cnt)
plt.xlabel('Feeling Temperature('
       + u'\N{DEGREE SIGN}' + 'C)')
plt.ylabel('Total Renters')
fit = np.polyfit(atemp,cnt,1)
fit_fn = np.poly1d(fit) 
plt.plot(atemp,cnt, 'yo', atemp, fit_fn(atemp), '--k')
plt.show()
print ("This is the graph to compare between the feeling temperature and temperature")

plt.scatter (hum,cnt)
plt.xlabel('Humidity')
plt.ylabel('Total Renters')
fit = np.polyfit(hum,cnt,1)
fit_fn = np.poly1d(fit) 
plt.plot(hum,cnt, 'yo', hum, fit_fn(hum), '--k')
plt.show()
print ("Correlation of humidity and total bike renters is {}".format(correlation(hum,cnt)))

plt.scatter (windspeed,cnt)
plt.xlabel('Wind speed')
plt.ylabel('Total Renters')
fit = np.polyfit(windspeed,cnt,1)
fit_fn = np.poly1d(fit) 
plt.plot(windspeed,cnt, 'yo', windspeed, fit_fn(windspeed), '--k')
plt.show()
print ("Correlation of wind speed and total bike renters is {}".format(correlation(windspeed,cnt)))

plt.scatter (weathersit,cnt)
plt.xlabel('Weather Conditions')
plt.ylabel('Total Renters')
plt.show()
print ("Correlation of Weather condition and total bike renters is {}".format(correlation(weathersit,cnt)))








