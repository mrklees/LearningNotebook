import numpy as np
from multiprocessing import Pool
import matplotlib.pyplot as plt
from bokeh.plotting import show, figure


class Population(object):
    """Class for generating a population and sampling from it"""
    
    def __init__(self, pop_mean=0, pop_scale=1, pop_size=10000):
        self.pop_mean = pop_mean
        self.pop_scale = pop_scale
        self.pop_size = pop_size
        self.population = np.random.normal(pop_mean, pop_scale, pop_size)
        self.true_median = np.median(self.population)
        
    def sample(self, size):
        return np.random.choice(self.population, size=size)
    
    @staticmethod
    def min_max(sample, ix):
        sample.sort()
        mn, mx = sample[ix-1], sample[-(ix)]
        return mn, mx
    
    def check_rule(self, size, ix):
        smpl = self.sample(size)
        mn, mx = self.min_max(smpl, ix)
        return int(mn < self.true_median < mx)
    
    def monte(self, sample_size, ix, n_samples=1000):
        return sum([self.check_rule(sample_size, ix) for i in range(n_samples)])/n_samples
        
    def get_prob_for_ixs(self, sample_size, ix_high, ix_low=1, n_samples=10000):
        return [self.monte(sample_size, ix, n_samples=n_samples) for ix in range(ix_low, ix_high)]
    
    @staticmethod
    def find_nearest(array, value=0.90):
        """We typically would like to know which index is closest to ~93.5% 
        
        """
        idx = (np.abs(array-value)).argmin()
        return idx, array[idx]
    
    def plot_ixs(self, sample_size, ix_high, ix_low=1, n_samples=10000):
        """Plots the probability of the median between the specified indices
        
        If we already have a sample, then we can visualize the desired probability
        for various indexes and determine visually what the best index value
        might be.
        """
        probs = self.get_prob_for_ixs(sample_size, ix_high, ix_low, n_samples)
        plt.plot(range(ix_low, ix_high), probs)
        best_ix, best = self.find_nearest(np.array(probs))
        plt.vlines(best_ix+1, 0, 1)
        plt.show()
        print(f"For a sample size of {sample_size}, you can use the {best_ix+1} values\n" +
              f"from the min/max and still have {best*100:.3f}% confidence\n" +
              f"that the median is between those values.")