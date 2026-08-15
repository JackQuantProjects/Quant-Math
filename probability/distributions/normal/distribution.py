import math
import numpy as np


class Normal():

    def __init__(self, mu, sigma):
        '''
        mu is the mean of the distribution
        sigma is the standard deviation

        variance is the variance of the distribution
        '''

        # catch edge cases
        if sigma <= 0:
            raise ValueError("sigma cannot be negative or zero")

        self.mu = mu
        self.sigma = sigma

        self.variance = self.sigma ** 2

    def pdf(self, x):
        '''
        returns the probability density at x
        '''
        coeffetient  = 1/(self.sigma * math.sqrt(2 * math.pi))
        ePart = math.e ** (-0.5 * ((x-self.mu)/(self.sigma))**2)

        return coeffetient * ePart   
        

    def cdf(self, x):
        '''
        returns the probability of all values below and at x
        '''
        z = (x - self.mu) / (self.sigma * math.sqrt(2))

        return 0.5 * (1 + math.erf(z))  

    def sample(self, n):
        '''
        returns a numpy array of n values generated
        using the distribution
        '''
        return np.random.normal(self.mu, self.sigma, n)
