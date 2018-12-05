# PCA Application


## PCA
Principal Component Analysis is a one of the best way to reduce feature dimensionality. In this project, I developed PCA
 and use in an example application. In `pca.py` we take input data matrix as `numpy array` like:
<br>
`X = np.array([[-1, -1], [-1, 1], [1, -1], [1, 1]])` and apply PCA algorithm steps: 

- Standardized Z matrix (`compute_Z(X)`)
- We calculate covariance matrix, it is basically dot product of Z *transpose* and Z itself (`compute_covariance_matrix(Z)`)
- Components, eigen values and eigen vectors (`find_pcs(COV)`)
- Lastly, we project data (`project_data(Z, PCS, L, k, var)`)
 
## Application


*This project developed for the Machine Learning Course (CS 691) at UNR*

