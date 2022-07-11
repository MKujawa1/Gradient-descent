import numpy as np 
import matplotlib.pyplot as plt 

x = np.arange(100)

m = 1
b = 1

y_data = m*x+b+np.random.randn(len(x))*5

m = 0
b = 0
n = len(x)
lr = 0.00001 ### Learning rate

mse = []
for i in range(80):

    y = m*x+b 
    deriv_m = (-2/n*sum(x*(y_data-y)))
    deriv_b = (-2/n*sum((y_data-y)))
    
    mse.append(np.mean(y_data-y)**2)

    m = m - lr*deriv_b
    b = b - lr*deriv_b




plt.figure()
plt.subplot(1,2,1)
plt.scatter(x,y_data)
plt.plot(x,y,'r')
plt.subplot(1,2,2)
plt.plot(mse)
plt.show()


x_len = 100

x2 = np.linspace(-x_len//2,x_len//2,x_len)
x = np.linspace(-x_len//2,x_len//2,x_len)
a = 0.2
b = 0.2
c = 0.1
y_data = a*x2**2+b*x2+c+np.random.randn(x_len)*20

a = 0
b = 0
c = 0
n = x_len
lr = 0.00000005
mse = []
for i in range(80):

    y = a*x**2+b*x+c

    deriv_a = (-2/n)*sum(x**2*(y_data-y))
    deriv_b = (-2/n)*sum(x*(y_data-y))
    deriv_c = (-2/n)*sum(y_data-y)
    
    mse.append(np.mean(y_data-y)**2)
    
    a = a - lr*deriv_a
    b = b - lr*deriv_b
    c = c - lr*deriv_c

plt.figure()
plt.subplot(1,2,1)
plt.scatter(x,y_data)
plt.plot(x,y,'r')
plt.subplot(1,2,2)
plt.plot(mse)
plt.show()