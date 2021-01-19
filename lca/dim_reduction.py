import matplotlib.pyplot as plt
import numpy as np

def PCA_var_explained_plots(adata):
	n_rows = 1
	n_cols = 2
	fig = plt.figure(figsize=(n_cols*4.5, n_rows*3))
	# variance explained
	ax1 = fig.add_subplot(n_rows, n_cols, 1)
	x1 = range(len(adata.uns['pca']['variance_ratio']))
	y1 = adata.uns['pca']['variance_ratio']
	ax1.scatter(x1, y1, s=3)
	ax1.set_xlabel('PC'); ax1.set_ylabel('Fraction of variance explained')
	ax1.set_title('Fraction of variance explained per PC')
	# cum variance explainend
	ax2 = fig.add_subplot(n_rows, n_cols, 2)
	cml_var_explained = np.cumsum(adata.uns['pca']['variance_ratio'])
	x2 = range(len(adata.uns['pca']['variance_ratio']))
	y2 = cml_var_explained
	ax2.scatter(x2, y2, s=4)
	ax2.set_xlabel('PC')
	ax2.set_ylabel('Cumulative fraction of variance explained')
	ax2.set_title('Cumulative fraction of variance explained by PCs')
	plt.tight_layout()
	plt.show()