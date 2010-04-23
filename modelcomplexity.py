import numpy as np
import scipy.optimize as op

# DUMMY FUNCTION; should interpolate
def grid_model(x,xgrid,ygrid):
    return 0.0 * x

def ln_likelihood_grid(xdata,ydata,yinvar,params,model_info):
    (foo, xgrid, epsilon) = model_info
    ymodel = grid_model(xdata,xgrid,params)
    chi2 = np.sum(yinvar * (ymodel - ydata)**2)
    penalty = np.sum(epsilon * (params[:-1]-params[1:])**2)
    return (-0.5) * (chi2 + penalty)

def polynomial_model(x,coeffs):
    fn = 1.0
    y = 0.0 * x
    for coeff in coeffs:
        y += coeff * fn
        fn *= x
    return y

def ln_likelihood_polynomial(xdata,ydata,yinvar,params,model_info):
    ymodel = polynomial_model(xdata,params)
    chi2 = np.sum(yinvar * (ymodel - ydata)**2)
    return (-0.5) * chi2

# DUMMY FUNCTION
def optimize(params):
    return params

def maximum_likelihood(xdata,ydata,yinvar,ln_likelihood,model_info):
    params = model_info[0] # first guess
    params = optimize(params)
    return params

# would be better to pack (xgrid,epsilon) as model_info and pass in just the
# likelihood function itself
def crossval_ln_likelihood(xdata,ydata,yinvar,ln_likelihood,model_info):
    numerator = 0.0
    denominator = 0.0
    for j,ydatum in enumerate(ydata):
        xdataj = np.append(xdata[:j],xdata[j+1:])
        ydataj = np.append(ydata[:j],ydata[j+1:])
        yinvarj = np.append(yinvar[:j],yinvar[j+1:])
        params = maximum_likelihood(xdataj,ydataj,yinvarj,ln_likelihood,model_info)
        numerator += 0.5 * yinvar[j] * (ydatum - grid_model(xdata[j],xgrid,ygrid))**2
        denominator += 1.0
    return (numerator / denominator)

def generate_data():
    N = 20
    xdata = np.random.uniform(size=(N,))
    ydata = 1.0 + xdata - xdata*xdata + xdata*xdata*xdata
    yinvar = np.ones(N)
    ydata += np.random.normal(0.0,1.0,size=ydata.shape) / np.sqrt(yinvar)
    return (xdata,ydata,yinvar)

def generate_grid():
    K = 50
    return np.arange(0,1,1.0/(K-1))

if __name__ == "__main__":
    print "Hello world."
    (xdata,ydata,yinvar) = generate_data()
    xgrid = generate_grid()
    ygrid = 0.0 * xgrid # first guess
    for epsilon in [0.1,1.0,10.0]:
        model_info = (ygrid, xgrid, epsilon)
        ln_likelihood = ln_likelihood_grid
        cll = crossval_ln_likelihood(xdata,ydata,yinvar,
                                     ln_likelihood,model_info)
        print epsilon, cll
    for K in [1,3,5]:
        coeffs = np.zeros(K+1) # first guess
        cll = crossval_ln_likelihood(xdata,ydata,yinvar,
                                     ln_likelihood,model_info)
        print K, cll
    print "Goodbye world."
