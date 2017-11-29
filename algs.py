import numpy as np

def pla_alg(x,y_true,max_iter=1000,w_est=None):
    '''
    perceptron learning algorithm
    input:
    x = NxNs matrix, data variable
    y_true = Ns matrix, ground truth labels 
    max_iter = maximum number of iteration
    
    Ns = number of samples
    N = dimension of features
    
    output:
    w_est = 1xN matrix, estimated weight
    n_miscls = misclassified examples in each iteration
    i = the number of iterations taken to converge,
        i==max_iter indicates possible non-convergence
    '''
    if w_est is None:
        w_est = np.random.rand(1,len(x)) #1x3

    n_miscls = []

    for i in range(max_iter):
        y_est = (w_est.dot(x)>0)*1.0+(w_est.dot(x)<=0)*-1.0 #1x100
        y_est = y_est[0] #100 length array from 1x100 matrix

        mis_cls = (y_est*y_true)<0


        y_sub = y_true[mis_cls]
        x_sub = x[:,mis_cls]
        n_sub = len(y_sub)
        n_miscls.append(n_sub)
        if n_sub>0:
            ch = np.random.choice(n_sub,size=1)

            w_est = w_est + y_sub[ch]*x_sub[:,ch].T

        else:
            break
    
    return w_est,n_miscls,i+1

def logreg_mle(x,y_true,learning_rate,max_iter=1000,w_est=None):
    '''
    Logistic Regression with MLE and GD algorithm
    input:
    x = NxNs matrix, data variable
    y_true = Ns matrix, ground truth labels
    learning_rate = gradient descent hyperparameter
    max_iter = maximum number of iteration
    
    Ns = number of samples
    N = dimension of features
    
    output:
    w_est = 1xN matrix, estimated weight
    n_miscls = misclassified examples in each iteration
    i = the number of iterations taken to converge,
        i==max_iter indicates possible non-convergence
    '''
    w_N = len(x)
    
    if w_est is None:
        w_est = np.random.rand(1,w_N)
    
    n_miscls = []
    
    for i in range(max_iter):
        grad_E = - np.mean(y_true*x / (1 + np.exp(y_true*w_est.dot(x))),axis=1)
        
        y_est = (w_est.dot(x)>0)*1.0+(w_est.dot(x)<=0)*-1.0 #1x100
        y_est = y_est[0] #100 length array from 1x100 matrix

        w_est = w_est - learning_rate * grad_E / np.sqrt(np.sum(grad_E*grad_E))
        if (i+1)%100 == 0 :
            learning_rate /= 2
            
        mis_cls = np.sum((y_est*y_true)<0)
        n_miscls.append(mis_cls)
        
        if mis_cls == 0:
            break
    
    return w_est,n_miscls,i
    
