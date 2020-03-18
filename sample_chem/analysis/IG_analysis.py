import os
import sys
import joblib
import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

import pandas as pd
import codecs

import scipy.stats

file_list = sorted(os.listdir())

for file in file_list:
    base, ext = os.path.splitext(file)
    if ext == '.jbl':
        print(file)
        with open(file, mode="rb") as f:
            data = joblib.load(f)
            
        igs = np.sum(data["embedded_layer_IG"][0], axis=1)
        igs_st = scipy.stats.zscore(igs)
        for index, item in enumerate(data["amino_acid_seq"]):
            print(data["amino_acid_seq"][index] + "\t", end="")
            print(igs_st[index])
