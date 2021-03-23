import numpy as np
import matplotlib.pylab as plt
from matplotlib import colors

x = np.arange(10)
y = np.arange(10)

alphas = np.linspace(0.1, 1, 10)
rgba_colors = np.zeros((10,4))
# for red the first column needs to be one
rgba_colors[:, 0:3] = colors.to_rgb('C0')
# the fourth column needs to be your alphas
rgba_colors[:, 3] = alphas

plt.scatter(x, y, color=rgba_colors)
plt.show()