import matplotlib.pyplot as plt
import nnfs 
from nnfs.datasets import vertical_data, spiral_data
#This script visualized the data imported from nnfs datasets so that I am able to understand it, as well as
#assist me with creating a basic network
nnfs.init()

X, y = vertical_data(samples=100, classes=3)

plt.scatter(X[:,0], X[:,1], c=y, s=40, cmap="brg")
plt.show()
