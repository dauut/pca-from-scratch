import numpy as np
import matplotlib.pyplot as plt
import os
import pca


def compress_images(DATA, k):

    # if it is not exist create output directory
    if not os.path.exists("Output"):
        os.makedirs("Output")

    faces = []
    filenames = []
    for face in DATA:
        filenames.append(face[1])
        faces.append(face[2])

    # we convert each image to a feature (column)
    faces = np.asarray(faces)
    faces = np.transpose(faces)

    # pca processes
    Z = pca.compute_Z(faces)
    COV = pca.compute_covariance_matrix(Z)
    L, PCS = pca.find_pcs(COV)
    Z_star = pca.project_data(Z, PCS, L, k, 0)

    # we use components for compressing files, we need to reduce component count
    component_matrix = np.delete(PCS, range(k, PCS.shape[1]), axis=1)
    Ut = component_matrix.T
    X_compressed = np.dot(Z_star, Ut)


    # write all images
    for ftr_ind in range(faces.shape[1]):
        # we need to reshape the feature as image.
        img = np.reshape(X_compressed[:,ftr_ind], (60, 48))
        plt.imsave("Output" + "/" + filenames[ftr_ind] + "_img.jpg", img, cmap="gray")

    print()


def load_data(input_dir):

    # collect all data file names
    file_name_list = []
    for root, dirs, files in os.walk(input_dir):
        for filename in files:
            file_name_list.append(filename)

    # read each image and properties
    image_list = []
    for i in range(len(file_name_list)):
        file_path = input_dir
        file_path += "/"
        filename = file_name_list[i]
        file_path += filename
        image = plt.imread(file_path)
        float_image = np.asarray(image.flatten(), dtype=float)
        image_list.append((image, filename, float_image))

    return image_list

