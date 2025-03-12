import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


import torch
import torch.nn as nn
from tqdm import tqdm # progress bar

torch.manual_seed(42); # set seed for reproducibility of results


dataset = pd.read_csv("student_performance.csv")
df = dataset[['hours_studied', 'prev_score', 'extra_activ', 'sleep_h', 'practiced_exams', 'performance_index']]

print('Read dataset completed successfully.')
print('Total number of rows: {0}\n\n'.format(len(df.index)))
df.head(200)

#generate a correlation matrix

test = df.hours_studied
print(df)
print(df.corr(numeric_only = True))



''' features '''
hours_studied = torch.tensor(df.hours_studied.values, dtype=torch.float32)
prev_score = torch.tensor(df.prev_score.values, dtype=torch.float32)
#extra_activ = torch.tensor(df.extra_activ.values, dtype=torch.float32)
sleep_h = torch.tensor(df.sleep_h.values, dtype=torch.float32)
practiced_exams = torch.tensor(df.practiced_exams.values, dtype=torch.float32)

''' target '''

performance_index = torch.tensor(df.performance_index.values, dtype=torch.float32)

''' start with random values for the coefficients '''

a = torch.randn(1, requires_grad = True) 
b = torch.randn(1, requires_grad = True)
c = torch.randn(1, requires_grad = True)
d = torch.randn(1, requires_grad = True)
e = torch.randn(1, requires_grad = True)

'''define the linear model as a function of the independent variables and the coefficients by a simple multiplication '''

def Model(hours_studied:torch.tensor, prev_score:torch.tensor, sleep_h:torch.tensor, practiced_exams:torch.tensor):
    '''
        computes f(x; a, b, c, d) = e + ax_1 + bx_2 + cx_3 + dx_4
        for independent variables x_1, x_2, x_3, x_4

        Arguments:
            hours_studied (tensor) with the values of tv investment (x_1)
            prev_score (tensor) with the values of radio investment (x_2)
            sleep_h (tensor) with the newspaper investment (x_3).
            practiced_exaks (tensor) with the newspaper investment (x_4).
    
    Note: coefficients a, b, c, d and e must be previoulsy 
    defined as tensors with requires_grad = True
    
    Returns a tensor with the backward() method
    '''
    return a * hours_studied + b * prev_score + c * sleep_h + d * practiced_exams + e

'''
    generate first prediction
'''

predicted = Model(hours_studied, prev_score, sleep_h, practiced_exams)

'''
    compare the predicted values with the actual values
'''

print(predicted.shape)
print(performance_index.shape)

'''
    Before performing the minimization, the model predicts badly sales because it only starts at random values of the TV, Radio and newspapers coefficients.
    The red line is the ideal prediction.
'''

plt.figure(figsize=(5,5))
plt.scatter(performance_index, predicted.detach(), c='k', s=4)
plt.xlabel('performance index'), plt.ylabel('predicted');
x = y = range(300)
plt.plot(x,y, c='brown')
plt.xlim(0,120), plt.ylim(0,120);
plt.text(60,60, f'e     = {e.item():2.4f}', fontsize=10);
plt.text(60,50, f'hours studied     = {a.item():2.4f}', fontsize=10);
plt.text(60,40, f'previous exam scores   = {b.item():2.4f}', fontsize=10);
plt.text(60,30, f'hours slept = {c.item():2.4f}', fontsize=10);
plt.text(60,20, f'practiced exams  = {d.item():2.4f}', fontsize=10);

plt.show()

'''
    minimization function
'''

def MSE(y_predicted:torch.tensor, y_target:torch.tensor):
    '''
        computes the mean squared error between the predicted and the actual values
    '''
    return torch.mean((y_predicted - y_target) ** 2)

predicted = Model(hours_studied, prev_score, sleep_h, practiced_exams)
loss = MSE(y_predicted = predicted, y_target=performance_index)
print(loss) # 2143.1174

# initial values for the coefficients is random, gradients are not calculated
print(f'a = {float(a.item()):+2.4f}, df(a)/da = {a.grad}') 
print(f'b = {float(b.item()):+2.4f}, df(b)/da = {b.grad}') 
print(f'c = {float(c.item()):+2.4f}, df(c)/dc = {c.grad}') 
print(f'd = {float(d.item()):+2.4f}, df(d)/dd = {d.grad}') 
print(f'e = {float(e.item()):+2.4f}, df(e)/de = {e.grad}') 

'''
    compute the gradients
'''

loss.backward()

myMSE = list()
for i in tqdm(range(5_000)):
    a.grad.zero_()
    b.grad.zero_()
    c.grad.zero_()
    d.grad.zero_()
    e.grad.zero_()

    predicted = Model(hours_studied=hours_studied, prev_score=prev_score, sleep_h=sleep_h, practiced_exams=practiced_exams) # forward pass (compute results)
    loss = MSE(y_predicted = predicted, y_target = performance_index) # calculate MSE
    
    loss.backward() # compute gradients
    myMSE.append(loss.item()) # append loss
    with torch.no_grad():
        a -= a.grad * 1e-6
        b -= b.grad * 1e-6
        c -= c.grad * 1e-6
        d -= d.grad * 1e-6
        e -= e.grad * 1e-6
        
plt.plot(myMSE);
plt.xlabel('Epoch (#)'), plt.ylabel('Mean squared Errors')
plt.show()

plt.figure(figsize=(5,5))
plt.scatter(performance_index, predicted.detach(), c='k', s=4)
plt.xlabel('performance index'), plt.ylabel('predicted');
x = y = range(300)
plt.plot(x,y, c='brown')
plt.xlim(0,120), plt.ylim(0,120);
plt.text(60,60, f'e     = {e.item():2.4f}', fontsize=10);
plt.text(60,50, f'hours studied     = {a.item():2.4f}', fontsize=10);
plt.text(60,40, f'previous exam scores   = {b.item():2.4f}', fontsize=10);
plt.text(60,30, f'hours slept = {c.item():2.4f}', fontsize=10);
plt.text(60,20, f'practiced exams  = {d.item():2.4f}', fontsize=10);
plt.show()