import math
import random

class Exponential:
    def __init__(self, lam):
        '''
        creates an exponential distribution with rate parameter lambda

        Parameters:
        lam (float): rate parameter, must be greater than 0

        Raises:
        ValueError: if lam is less than or equal to 0
        '''
        # edge case
        if lam <= 0:
            raise ValueError("lambda must be positive")

        self.lam = lam
        self.mu = 1/lam
        self.variance = 1 / lam**2

    def pdf(self, x):
        '''
        returns the probability density at x

        f(x) = lambda * exp(-lambda * x), for x >= 0
        f(x) = 0, for x < 0

        Parameters:
        x (float): point at which to evaluate the density

        Returns:
        float: probability density at x
        '''
        # edge case
        if x < 0:
            return 0

        return (self.lam * math.exp(-self.lam * x))

    def cdf(self, x):
        '''
        returns the cumulative probability up to x

        F(x) = 1 - exp(-lambda * x), for x >= 0
        F(x) = 0, for x < 0

        Parameters:
        x (float): point at which to evaluate the cumulative probability

        Returns:
        float: probability that X <= x
        '''
        # edge case
        if x < 0:
            return 0

        return (1 - math.exp(-self.lam * x))

    def sample(self):
        '''
        generates a random sample from the exponential distribution

        Uses inverse transform sampling:

        X = -ln(1 - U) / lambda

        where U is uniformly distributed on (0, 1).

        Returns:
        float: random sample from the distribution
        '''
        u = random.random()
        return (-math.log(1 - u) / self.lam)
