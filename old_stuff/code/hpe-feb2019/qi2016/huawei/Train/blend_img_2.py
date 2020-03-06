# sphinx_gallery_thumbnail_number = 3
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize


def normal_pdf(x, mean, var):
    return np.exp(-(x - mean)**2 / (2*var))


# Generate the space in which the blobs will live
xmin, xmax, ymin, ymax = (0, 100, 0, 100)
n_bins = 100
xx = np.linspace(xmin, xmax, n_bins)
yy = np.linspace(ymin, ymax, n_bins)

# Generate the blobs. The range of the values is roughly -.0002 to .0002
means_high = [20, 50]
means_low = [50, 60]
var = [150, 200]

gauss_x_high = normal_pdf(xx, means_high[0], var[0])
gauss_y_high = normal_pdf(yy, means_high[1], var[0])

gauss_x_low = normal_pdf(xx, means_low[0], var[1])
gauss_y_low = normal_pdf(yy, means_low[1], var[1])

weights_high = np.array(np.meshgrid(gauss_x_high, gauss_y_high)).prod(0)
weights_low = -1 * np.array(np.meshgrid(gauss_x_low, gauss_y_low)).prod(0)
weights = weights_high + weights_low

# We'll also create a grey background into which the pixels will fade
greys = np.ones(weights.shape + (3,)) * 70

# First we'll plot these blobs using only ``imshow``.
vmax = np.abs(weights).max()
vmin = 0
cmap = plt.cm.jet

# Create an alpha channel of linearly increasing values moving to the right.
alphas = np.ones(weights.shape)*0.5
# alphas[:, 30:] = np.linspace(1, 0, 70)

# Normalize the colors b/w 0 and 1, we'll then pass an MxNx4 array to imshow
colors = Normalize(vmin, vmax, clip=True)(weights)
colors = cmap(colors)

# Now set the alpha channel to the one we created above
colors[..., -1] = alphas

# Create the figure and image
# Note that the absolute values may be slightly different
fig, ax = plt.subplots()

ax.imshow(greys)
ax.imshow(colors, extent=(xmin, xmax, ymin, ymax))
ax.set_axis_off()


plt.show()