import numpy as np
import scipy
import math as m

class SpikeResponseModel():
    ''' SpikeResponseModel

    Parameters (referring to Gerstner textbook):

    n_inputs: int
        Number of presynaptic inputs
    threshold: float
        Membrane potential around which spiking occurs. Referred to as 'nu' in the textbook
    dt: float
        Timescale of discretization
    beta: float
        Sets how steep the logistic function yielding spike probabilities is
    tau_m: float
        Membrane time constant employed in eqs. 4.35-4.37
    tau_s: float
        Synaptic time constant employed in eqs. 4.38
    u_r: float
        Magnitude of refractory function (eq. 4.35)
    C: float
        Membrane conductance (eq. 4.36)
    '''
    def __init__(self, n_inputs, threshold, dt, beta, tau_m, tau_s, u_r, C):

        self.n_inputs = n_inputs
        self.threshold = threshold
        self.dt = dt
        self.beta = beta
        self.tau_m = tau_m
        self.tau_s = tau_s
        self.u_r = u_r
        self.C = C

    '''
    function simulate

    Parameters:

    T: float
        Total time window over which to simulate neuron output

    input: list of lists. 
        Outer length should match n_inputs. Inner lists are comprised of a list of spike times
        for each presynaptic input

    returns list of output spike times
    '''
    def simulate(self, T, input):

        # High level strategy: Given the inputs, time evolve eq. 4.34 over timescale dt. 
        # During each time bin, use the current value of the membrane voltage to probabilistically 
        # generate a spike. Recall that given u_t, the spike generation mechanism looks like:

        # f(u) = np.exp(self.beta * (u_t - self.threshold)) * self.dt
        # p(spike) in time bin [t, t + self.dt] = f(u)/(1 + f(u))

        # Return the list of output spikes at the end
        
        def Hsf(s):
            # Heaviside step function
            if (s>0):
                return 1
            else:
                return 0
            
        def alpha(s):
            # input current pulse
            return (1/self.tau_s)*m.exp(-1*s/self.tau_s)*Hsf(s)
        
        def fai(s):
            # eq. 4.35
            return (self.u_r)*(m.exp(-1*s/self.tau_m))
        
        def sigma(s, t):
            # eq. 4.36
            def subfunction(x):
                return (m.exp(-1*x/self.tau_m))*alpha(t-x)
            value, error = scipy.integrate.quad(subfunction, 0, s)
            return (1/self.C)*value
            
        def Kai(s, t):
            # eq. 4.37
            return (1/self.C)*(m.exp(-1*t/self.tau_m))*Hsf(s-t)*Hsf(t)
        
        def u_i(t, ti):
            # eq. 4.34
            def subfunction2(s):
                return Kai(t-ti, s)*alpha(t-s)
            
            n = 0
            for i in range(self.n_inputs):
                for j in range(len(input[i])):
                    n = n+sigma(t-ti, t-input[i][j])
            
            return fai(t-ti) + n + scipy.integrate.quad(subfunction2, 0, np.inf)
        
        def pspike(u):
            # probability of generating a spike in the time bin [t, t+dt]
            def f(u):
                return np.exp(self.beta*(u - self.threshold))*self.dt
            return f(u)/(1 + f(u))
        
        # let we suppose that the first spike happens at t=0
        ti = 0
        timelist = []
        timelist.append(ti)
        
        for i in range(T/self.dt):
            t = self.dt*(i+1)
            u = u_i(t, ti)
            probability = pspike(u)
            binary = np.random.binomial(n=1, p=probability)
            if (binary==1):
                ti = t
                timelist.append(t)
        
        print(timelist)