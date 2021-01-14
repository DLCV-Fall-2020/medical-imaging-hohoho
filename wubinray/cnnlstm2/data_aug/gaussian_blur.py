import cv2 
import numpy as np

np.random.seed(87)

class GaussianBlur(object):
    
    # Implements Gaussian blur as described in the SimCLR paper
    def __init__(self, kernel_size, min=0.1, max=2.0):
        self.min = min
        self.max = max
        self.kernel_size = kernel_size #10% of img_size
        self.kernel_size += 1 if kernel_size%2==0 else 0

    def __call__(self, sample):
        sample = np.array(sample)
        
        p = np.random.random_sample() #p=0.5 to blur

        if p < 0.5:
            sigma = (self.max - self.min) * np.random.random_sample() + self.min
            sample = cv2.GaussianBlur(sample, 
                    (self.kernel_size,self.kernel_size), sigma)

        return sample

