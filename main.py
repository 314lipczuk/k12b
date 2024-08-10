"""
Principal Component Analysis - a sketch

Jak to ogarnąć?
1. Co to jest?
	Technika statystyczna.
	Robi dimentionality reduction, izoluje Principal component.
	To znaczy?
2. Co to robi?
3. Z czego to się składa?
	Standarize data
	Compute covariance Mx
	Calculate eigen(values|vectors)
	Sort them
	Select principal components from then
	Transform/return the data
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import numpy.random as rnd

def read_data(filename):
	data = pd.read_csv(filename)
	return data

def main():
	# Read data
	#data = read_data("./data/Global25_PCA_modern_scaled.txt")

	print(data)
	# It is the scaled dataset, i presume there is no further need to clean it.
    	# Perhaps.
	# Check that to make sure.

#main()




def test():
	mu = np.array([10,13])
	sigma = np.array([[3.5, -1.8], [-1.8,3.5]])

	print("Mu ", mu.shape)
	print("Sigma ", sigma.shape)

	# Create 1000 samples using mean and sigma
	org_data = rnd.multivariate_normal(mu, sigma, size=(1000))
	print("Data shape ", org_data.shape)

	mean = np.mean(org_data, axis= 0)
	print("Mean ", mean.shape)
	mean_data = org_data - mean
	print("Data after subtracting mean ", org_data.shape, "\n")

	cov = np.cov(mean_data.T)
	cov = np.round(cov, 2)
	print("Covariance matrix ", cov.shape, "\n")

	# Perform eigen decomposition of covariance matrix
	eig_val, eig_vec = np.linalg.eig(cov)
	print("Eigen vectors ", eig_vec)
	print("Eigen values ", eig_val, "\n")

	# Sort eigen values and corresponding eigen vectors in descending order
	indices = np.arange(0,len(eig_val), 1)
	indices = ([x for _,x in sorted(zip(eig_val, indices))])[::-1]
	eig_val = eig_val[indices]
	eig_vec = eig_vec[:,indices]
	print("Sorted Eigen vectors ", eig_vec)
	print("Sorted Eigen values ", eig_val, "\n")

	# Get explained variance
	sum_eig_val = np.sum(eig_val)
	explained_variance = eig_val/ sum_eig_val
	print(explained_variance)
	cumulative_variance = np.cumsum(explained_variance)
	print(cumulative_variance)

	# Take transpose of eigen vectors with data
	pca_data = np.dot(mean_data, eig_vec)
	print("Transformed data ", pca_data.shape)

	## Compute reconstruction loss
	#loss = np.mean(np.square(recon_data - org_data))
	# print("Reconstruction loss ", loss)

test()