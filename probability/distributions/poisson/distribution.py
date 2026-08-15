import math
import random


class Poisson:
    def __init__(self, lam):
        '''
        creates a Poisson distribution with rate parameter lambda

        Parameters:
        lam (float): expected number of events, must be greater than 0

        Raises:
        ValueError: if lam is less than or equal to 0
        '''
        # edge case
        if lam <= 0:
            raise ValueError("lambda must be positive")

        self.lam = lam
        self.mu = lam
        self.variance = lam

    def pmf(self, k):
        '''
        returns the probability that the distribution takes the value k

        P(X = k) = exp(-lambda) * lambda^k / k!

        Parameters:
        k (int): non-negative integer value

        Returns:
        float: probability that X equals k

        Raises:
        TypeError: if k is not an integer
        '''
        # edge cases
        if k < 0:
            return 0
        if not isinstance(k, int):
            raise TypeError("k must be an integer")

        return (math.exp(-self.lam) * (self.lam ** k)) / (math.factorial(k))

    def cdf(self, k):
        '''
        returns the cumulative probability up to and including k

        P(X <= k) = sum(P(X = x)) for x = 0, ..., k

        Parameters:
        k (int): non-negative integer value

        Returns:
        float: probability that X is less than or equal to k

        Raises:
        TypeError: if k is not an integer
        '''
        # edge cases
        if k < 0:
            return 0
        if not isinstance(k, int):
            raise TypeError("k must be an integer")

        p = 0

        for x in range(k+1):
            p += self.pmf(x)

        return p

    def sample(self):
        '''
        generates a random sample from the Poisson distribution

        Uses Knuth's algorithm to generate a Poisson-distributed
        random integer.

        Returns:
        int: random sample from the distribution
        '''
        L = math.exp(-self.lam)
        k = 0
        p = 1

        while p > L:
            k += 1
            p *= random.random()

        return k - 1
