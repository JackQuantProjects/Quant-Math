import math
import numpy as np

class Uniform():

    def __init__(self, a, b):
        '''
        a is the lower bound of the distribution
        b is the upper bound of the distribution

        p is the probability of each event

        mean is the expected value

        std is the standard deviation
        '''
    
        #catch edge cases

        if (a>=b):
            raise ValueError('cannot generate if the lower bound is bigger than the upper bound')

        self.a = a
        self.b = b

        self.p = 1/(b-a)

        self.mean = (a+b)/2

        self.std = (b-a)/math.sqrt(12)

    def pdf(self, x):
        '''
        returns The probability at the value x
        '''
        #catch edge cases
        if (self.a > x) or (self.b < x):
            return 0

        return self.p

    def cdf(self, x):
        '''
        returns the probability of all the values below and at x
        '''
        #catch edge cases
        if (self.a > x):
            return 0
        elif (self.b < x):
            return 1
        

        return ((x-self.a)*self.p)

    def sample(self, n):
        '''
        returns a numpy array of n values generated using the distribution
        '''

        if not isinstance(n, int):
            raise TypeError('sample size must be an integer')

        if n < 0:
            raise ValueError('sample size cannot be negative')
        
        array = np.random.uniform(0, 1, n) 
        
        array = array*(self.b-self.a) + self.a

        return array


        

