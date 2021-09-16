import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sklearn

x = np.linspace(-np.pi,np.pi,100)
y = np.sin(x)

plt.plot(x,y)
plt.show()