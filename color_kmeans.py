# USAGE.jpg
# python color_kmeans.py --image images/jp.png --clusters 3

# import the necessary packages
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import utils

def coulor_Kmeans(xmin,ymin,xmax,ymax,image,clusters):
    image_new=image[xmin:xmax,ymin:ymax,:]
    image_new = image_new.reshape((image_new.shape[0] * image_new.shape[1], 3))

# cluster the pixel intensities
    clt = KMeans(n_clusters = clusters)
    clt.fit(image_new)

# build a histogram of clusters and then create a figure
# representing the number of pixels labeled to each color
    hist = utils.centroid_histogram(clt)
    bar = utils.plot_colors(hist, clt.cluster_centers_)

# show our color bart
    plt.figure()
    plt.axis("off")
    plt.imshow(bar)
    plt.show()
    return clt.cluster_centers_