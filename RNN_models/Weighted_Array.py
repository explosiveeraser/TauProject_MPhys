import numpy as np
from tqdm import tqdm, trange

def weight_array(ar, weights):

    zipped = zip(ar, weights)

    weighted = []

    for i in tqdm(zipped, desc="weighting"):
        for j in range(int(i[1])):

            weighted.append(i[0])

    return weighted

