import pickle
import os

import matplotlib.cm as cm
import matplotlib.pyplot as plt

with open(os.path.join('data', 'raw', 'train.pkl'), 'rb') as f:
    (train_x, train_y) = pickle.load(f)
print(type(train_x))

# display some digits
plt.imshow(train_x[0].reshape((56, 56)), cmap=cm.Greys_r)
plt.show()