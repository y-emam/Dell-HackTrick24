from matplotlib.pyplot import imread
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import numpy as np
import cv2


def segment_image():
    image = imread("ireland.jpg")

    print(image.shape)
    # Reshaping the image into a 2D array of pixels and 3 color values (RGB)
    pixel_vals = image.reshape((-1, 3))

    # Convert to float type
    pixel_vals = np.float32(pixel_vals)

    print(pixel_vals.shape)

    kmeans = KMeans(n_clusters=2)

    kmeans.fit(pixel_vals)

    segmented_image = kmeans.cluster_centers_[kmeans.labels_]
    segmented_image = segmented_image.reshape(image.shape)

    plt.imshow(segmented_image / 255)
    plt.show()

    return {"segmented_image": segment_image.astype(int), "clusterer": kmeans}


segment_image()
