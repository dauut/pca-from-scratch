import numpy as np
import pca as p
import compress as c

TRAINING_DATA = "Data/Train/"
TEST_DATA = "Data/Test/"

X = c.load_data(TRAINING_DATA)

# c.compress_images(X, 10)
# c.compress_images(X, 100)
# c.compress_images(X, 500)
# c.compress_images(X, 1000)
c.compress_images(X, 2000)

# X = c.load_data(TEST_DATA)
# c.compress_images(X, 2000)

# X = np.array([[-1, -1], [-1, 1], [1, -1], [1, 1]])
# X = np.array([[1, 1], [2, 7], [3, 3], [4, 4], [5, 5]])
# Z = p.compute_Z(X)
# COV = p.compute_covariance_matrix(Z)
# L, PCS = p.find_pcs(COV)
# Z_star = p.project_data(Z, PCS, L, 2, 0)
# print(Z_star)


exit()