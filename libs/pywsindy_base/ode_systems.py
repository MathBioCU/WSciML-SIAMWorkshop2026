import numpy as np
import time
from WENDy import *
from scipy.integrate import solve_ivp

def HindmarshRose():
    a, b, c, d, r, s, xR, I, ts = 1.0, 3.0, 1.0, 5.0, 1e-3, 4.0, -3.19/4.0, 0.0, 10.0
    features = [
        [lambda x, y, z: y, lambda x, y, z: x**3, lambda x, y, z: x**2, lambda x, y, z: z, lambda x, y, z: x*0+1.0],
        [lambda x, y, z: x*0+1.0, lambda x, y, z: x**2, lambda x, y, z: y],
        [lambda x, y, z: x, lambda x, y, z: x*0+1.0, lambda x, y, z: z]
    ]
    params = [np.array([ts, -a*ts, b*ts, -1*ts, I*ts]), np.array([c*ts, -d*ts, -1*ts]), np.array([r*s*ts, -r*s*xR*ts, -r*ts])]

    x0 = np.array([-1.31,-7.6,-0.2])
    t =  np.linspace(0, 10, 1024)
    tspan = (t[0], t[-1])
    tol_ode = 1e-15
    rhs_p = lambda t, x: rhs_fun(features, params, x)
    true_vec = np.concatenate(params).reshape(-1, 1)
    options_ode_sim = {"rtol": tol_ode, "atol": tol_ode*np.ones(len(x0))}

    t0 = time.time()
    sol = solve_ivp(rhs_p, t_span = tspan, y0=x0, t_eval=t, rtol=tol_ode, atol=tol_ode)
    #print("sim time =", time.time() - t0)
    x = sol.y.T
    t = sol.t
    #plt.plot(t, x)
    return x, t, params, x0, true_vec, features, rhs_p

def lorenz():
    features = [
        [lambda x, y, z: y, lambda x, y, z: x],
        [lambda x, y, z: x, lambda x, y, z: x*z, lambda x, y, z: y],
        [lambda x, y, z: x*y, lambda x, y, z: z]
    ]
    params = [np.array([10, -10]), np.array([28, -1, -1]), np.array([1, -8/3])]

    x0 = np.array([-8, 10, 27])
    t =  np.linspace(0, 10, 501)
    tspan = (t[0], t[-1])
    tol_ode = 1e-15
    rhs_p = lambda t, x: rhs_fun(features, params, x)
    true_vec = np.concatenate(params).reshape(-1, 1)
    options_ode_sim = {"rtol": tol_ode, "atol": tol_ode*np.ones(len(x0))}

    t0 = time.time()
    sol = solve_ivp(rhs_p, t_span = tspan, y0=x0, t_eval=t, rtol=tol_ode, atol=tol_ode)
    #print("sim time =", time.time() - t0)
    x = sol.y.T
    t = sol.t
    #plt.plot(t, x)
    return x, t, params, x0, true_vec, features, rhs_p

def logistic_growth():
    features = [
        [lambda x: x, lambda x: x**2]
    ]
    params = [np.array([ 1, -1])]
    x0 = np.array([ 0.01 ])
    t =  np.linspace(0, 10, 501)
    tspan = (t[0], t[-1])
    tol_ode = 1e-15

    rhs_p = lambda t, x: rhs_fun(features, params, x)
    true_vec = np.concatenate(params).reshape(-1, 1)
    options_ode_sim = {"rtol": tol_ode, "atol": tol_ode*np.ones(len(x0))}

    t0 = time.time()
    sol = solve_ivp(rhs_p, t_span = tspan, y0=x0, t_eval=t, rtol=tol_ode, atol=tol_ode)
    #print("sim time =", time.time() - t0)
    x = sol.y.T
    t = sol.t
    #plt.plot(t, x)
    return x, t, params, x0, true_vec, features, rhs_p

def wsindy_ode_defaults(ode_name):
    if ode_name == 'Linear':
        ode_params = np.array([[[-0.1, 2], [-2, -0.1]]]) 
        x0 = np.array([3,0]).T
        t_span = np.array([0, 15])
        t_eval = np.linspace(0, 15, 1501)
    elif ode_name == 'Logistic_Growth':
        ode_params = np.array([2])
        x0 = np.array([0.01]).T
        t_span = np.array([0, 10])
        t_eval = np.arange(0, 10, 0.005)
    elif ode_name == 'Van_der_Pol':
        dt = 0.01
        ode_params = np.array([4])
        x0 = np.array([0,1]).T
        t_span = np.array([0, 30])
        t_eval = np.arange(0, 30, dt)
    elif ode_name == 'Duffing':
        mu = 0.2
        ode_params =  np.array([mu, mu**2/4*5,1])
        x0 = np.array([0,2]).T
        t_eval = np.arange(0, 30, 0.01)
        t_span = np.array([0,30])
    elif ode_name == 'Lotka_Volterra':
        alpha= 2/3
        beta = 4/3
        ode_params = np.array([alpha, beta,1, 1])
        x0 = np.array([10,10]).T
        t_span = np.array([0, 200])
        t_eval = np.arange(0, 200, 0.02)
    elif ode_name == 'Lorenz':
        ode_params = np.array([10, 8/3,27])
        t_span = np.array([0.001, 10])
        t_eval = np.linspace(0.001, 10, 5000)
        x0 = np.array([-8 ,10 ,27]).T
        #x0 = [rand(2,1)*30-15;rand*30+10]    
    return ode_params, t_span, t_eval, x0

def simODE(ode_name, noise_ratio, x0=None, t_span=None, t_eval=None, tol_ode = 1e-15, params=None):

    params_d, t_span_d, t_eval_d, x0_d =  wsindy_ode_defaults(ode_name)
    if params is None:
        params=params_d
    if t_span is None:
        t_span=t_span_d
    if t_eval is None:
        t_eval=t_eval_d
    if x0 is None:
        x0=x0_d
        
    if ode_name == 'Linear':
        A = params[0]
        def rhs(t, x): return A.dot(x)
        weights = []
        for i in range(len(A[0])):
            weights.append(np.insert(np.identity(
                len(A[0])), 2, np.array((A[i, :])), axis=1))
    elif ode_name == 'Logistic_Growth':
        pow = 2  # params[0]
        def rhs(t, x): return x - x**pow
        weights = [np.array([[1, 1],    [pow, -1]])]
    elif ode_name == 'Duffing':
        mu = params[0]
        alpha = params[1]
        beta = params[2]
        def rhs(t, x): return np.array([x[1], -mu*x[1] - alpha*x[0] - beta*x[0]**3])
        weights = [np.reshape(np.array([0, 1, 1]), (1, 3)), np.array(
            [[1, 0, -alpha], [0, 1, -mu], [3, 0, -beta]])]
    elif ode_name == 'Lotka_Volterra':
        alpha = params[0]
        beta = params[1]
        delta = params[2]
        gamma = params[3]
        def rhs(t, x): return np.array([alpha*x[0] - beta*x[0]*x[1], delta*x[0]*x[1] - gamma*x[1]])
        weights = [np.array([[1, 0, alpha], [1, 1, -beta]]),
                   np.array([[0, 1, -gamma], [1, 1, delta]])]
    elif ode_name == 'Van_der_Pol':
        mu = params[0]
        def rhs(t, x): return np.array([x[1], mu*x[1] - mu*x[0]**2*x[1] - x[0]])
        weights = [np.reshape(np.array([0, 1, 1]), (1, 3)), np.array(
            [[1, 0, -1], [0, 1, mu], [2, 1, -mu]])]
    elif ode_name == 'Lorenz':
        def lorenz_loc(x, sigma, beta, rho):
            a = sigma*(x[1] - x[0])
            b = x[0]*(rho - x[2]) - x[1]
            c = x[0]*x[1] - beta*x[2]
            return np.array([a, b, c])

        sigma = params[0]
        beta = params[1]
        rho = params[2]
        def rhs(t, x): return lorenz_loc(x, sigma, beta, rho)
        weights = [np.array([[0, 1, 0, sigma], [1, 0, 0, -sigma]]), np.array([[1, 0, 0, rho], [1, 0, 1, -1], [0, 1, 0, -1]]), np.array([[1, 1, 0, 1], [0, 0, 1, -beta]])]

    sol = solve_ivp(fun=rhs, t_eval=t_eval, t_span=t_span, y0=x0, rtol=tol_ode)
    

    x = sol.y.T
    xobs = addNoise(x, noise_ratio)
    return weights, sol.t, xobs, rhs, x0

def addNoise(x, noise_ratio):
    signal_power = np.sqrt(np.mean(x**2))
    sigma = noise_ratio*signal_power
    noise = np.random.normal(0, sigma, x.shape)
    xobs = x + noise
    return xobs
