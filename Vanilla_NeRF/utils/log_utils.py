import wandb
import numpy as np
import cv2
import os
import torch

class MeanTracker(object):
    def __init__(self):
        self.reset()

    def add(self, input, weight=1.):
        for key, l in input.items():
            if not key in self.mean_dict:
                self.mean_dict[key] = 0
            self.mean_dict[key] = (self.mean_dict[key] * self.total_weight + l) / (self.total_weight + weight)
        self.total_weight += weight

    def has(self, key):
        return (key in self.mean_dict)

    def get(self, key):
        return self.mean_dict[key]

    def as_dict(self):
        return self.mean_dict

    def reset(self):
        self.mean_dict = dict()
        self.total_weight = 0

    def print(self, f=None):
        for key, l in self.mean_dict.items():
            if f is not None:
                print("{}: {}".format(key, l), file=f)
            else:
                print("{}: {}".format(key, l))

def compute_rmse(prediction, target):
    return torch.sqrt((prediction - target).pow(2).mean())
