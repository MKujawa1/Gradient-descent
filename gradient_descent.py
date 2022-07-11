import numpy as np 
import matplotlib.pyplot as plt 

def generate_linear_data(x_len =100,m =1, b = 1,noi = 5):
    '''
    Generate linear data with noise.

    Parameters
    ----------
    x_len : int, optional
        x range size. The default is 100.
    m : float, optional
        m parameter. The default is 1.
    b : float, optional
        b parameter. The default is 1.
    noi : float, optional
        Amplitude of noise. The default is 5.

    Returns
    -------
    array
        Linear function with noise.

    '''
    x = np.arange(x_len)
    return m*x+b+np.random.randn(len(x))*noi

def generate_quadratic_data(x_len = 100,a = 0.2 ,b = 0.2, c = 0.1,noi = 20):
    '''
    Generate quadratic data with noise

    Parameters
    ----------
    x_len : int, optional
        x range size. The default is 100.
    a : float, optional
        a parameter. The default is 0.2.
    b : float, optional
        b parameter. The default is 0.2.
    c : float, optional
        c parameter. The default is 0.1.
    noi : float, optional
        Amplitude of noise. The default is 20.

    Returns
    -------
    array
        Quadratic function with noise.

    '''
    x = np.linspace(-x_len//2,x_len//2,x_len)
    return a*x**2+b*x+c+np.random.randn(x_len)*noi

def linear_function(in_data,epochs = 80,lr = 0.00001):
    '''
    Gradient descent for linear function.

    Parameters
    ----------
    in_data : array
        Linear data with noise.
    epochs : int, optional
        Loop increment . The default is 80.
    lr : float, optional
        Learning rate. The default is 0.00001.

    Returns
    -------
    y : array
        Fit data.
    mse : list 
        List of Mean Squared Errors.

    '''
    m = 0 ### init parameter
    b = 0 ### init parameter
    n = len(in_data) ### Number of samples
    x = np.arange(n)
    ### Create empty list of MSE
    mse = [] 
    ### Start calculations 
    for _ in range(epochs):
        ### Get linear function with parameters
        y = m*x+b 
        ### Get values of partial derivatives
        deriv_m = (-2/n*sum(x*(in_data-y)))
        deriv_b = (-2/n*sum((in_data-y)))
        ### Get MSE
        mse.append(np.mean(in_data-y)**2)
        ### Update functions parameters
        m = m - lr*deriv_m
        b = b - lr*deriv_b
    
    return y,mse

def quadratic_function(in_data,epochs = 800, lr = 0.00000005):
    '''
    Gradient descent for quadratic function.

    Parameters
    ----------
    in_data : array
        Quadratic data with noise.
    epochs : int, optional
        Loop increment . The default is 80.
    lr : float, optional
        Learning rate. The default is 0.00001.

    Returns
    -------
    y : array
        Fit data.
    mse : list 
        List of Mean Squared Errors.

    '''
    a = 0 ### init parameter
    b = 0 ### init parameter
    c = 0 ### init parameter
    n = len(in_data) ### number of samples
    x = np.linspace(-n//2,n//2,n) ### x range
    ### Create empty list of MSE
    mse = []
    ### Start calculations
    for _ in range(epochs):
        ### Get quadratic function with parameters
        y = a*x**2+b*x+c
        ### Get values of partial derivatives
        deriv_a = (-2/n)*sum(x**2*(in_data-y))
        deriv_b = (-2/n)*sum(x*(in_data-y))
        deriv_c = (-2/n)*sum(in_data-y)
        ### Get MSE
        mse.append(np.mean(in_data-y)**2)
        ### Update function parameters
        a = a - lr*deriv_a
        b = b - lr*deriv_b
        c = c - lr*deriv_c
        
    return y,mse

if __name__ == '__main__':
    ### Get linear data, fit and error
    y_linear = generate_linear_data()
    y_fit_linear, mse_linear = linear_function(y_linear,epochs = 80,lr = 0.00001)
    ### Get quadratic data, fit and error
    y_quadratic = generate_quadratic_data()
    y_fit_quadratic, mse_quadratic = quadratic_function(y_quadratic, epochs = 50, lr = 0.00000005)
    ### Display results
    plt.figure(figsize = (8,6))
    plt.suptitle('Gradient descent')
    plt.subplot(2,2,1)
    plt.ylabel('Amp')
    plt.xlabel('X')
    plt.scatter(np.arange(len(y_linear)),y_linear,label = 'raw data')
    plt.plot(np.arange(len(y_linear)),y_fit_linear,'r',label = 'fit')
    plt.legend()
    plt.subplot(2,2,2)
    plt.ylabel('MSE')
    plt.xlabel("Epochs")
    plt.plot(np.arange(len(mse_linear)),mse_linear)
    plt.subplot(2,2,3)
    plt.ylabel('Amp')
    plt.xlabel('X')
    x = np.linspace(-len(y_quadratic)//2,len(y_quadratic)//2,len(y_quadratic))
    plt.scatter(x,y_quadratic,label = 'raw data')
    plt.plot(x,y_fit_quadratic,'r',label = 'fit')
    plt.legend()
    plt.subplot(2,2,4)
    plt.ylabel('MSE')
    plt.xlabel("Epochs")
    plt.plot(np.arange(len(mse_quadratic)),mse_quadratic)
    plt.tight_layout()
    plt.show()