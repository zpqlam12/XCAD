import numpy as np

weights = np.load('vgg16_weights.npz')
keys = sorted(weights.keys())
for i, k in enumerate(keys):
    print i, k, np.shape(weights[k])

for i in range(len(keys)-6):
    if i == 0:
        dic = {keys[0]: weights[keys[0]]}
    else:
        dic[keys[i]] = weights[keys[i]]


np.savez('vgg16_weights_nonfc.npz', **dic)

