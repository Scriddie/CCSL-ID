import scanpy as sc
import anndata as ann
import numpy as np
import scipy as sp
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib import colors
import os 
import scrublet as scr
import seaborn as sb
import rpy2
import rpy2.robjects as ro
from rpy2.robjects.packages import importr
# ro.r('library()')


plt.rcParams['figure.figsize']=(8,8) #rescale figures
sc.settings.verbosity = 3
sc.set_figure_params(dpi=200, dpi_save=300)
sb.set_context(context='poster')

file_path_raw = 'scanpy_course/3k_PBMC/'

# The data directory contains all processed data and `anndata` files.
data_dir = 'PBMC_Colabs/data/'

"""The tables directory contains all tabular data output, e.g. in `.csv` or `.xls` file format. That applies to differential expression test results or overview tables such as the number of cells per cell type."""

table_dir = 'PBMC_Colabs/tables/'

"""The default figure path is a POSIX path calles 'figures'. If you don't change the default figure directory, scanpy creates a subdirectory where this notebook is located.  """

sc.settings.figdir = 'PBMC_Colabs/figures/'

import datetime
today = datetime.date.today().strftime('%y%m%d')

"""
The dataset consists of 4k PBMCs (Human) provided by 10X Genomics. The data is an mtx directory with an `mtx` file (*i.e.* count matrix), two `tsv` files with barcodes (*i.e.* cell indices) and features (*i.e.* gene symbols). `Scanpy` unpacks the files (if the files are in `gz` archive format) and creates an `anndata` object with the `read_10x_mtx` function. The dataset is not filtered.
"""

file_path_raw = file_path_raw + 'raw_gene_bc_matrices/'

adata_raw = sc.read_10x_mtx(path=file_path_raw, cache=True)

"""Let us check the dataset size. """

adata_raw.shape

print('Total number of observations: {:d}'.format(adata_raw.n_obs))

"""# Quality control

## Remove empty droplets

The dataset contains an excessive amount of "cells", which are in fact empty droplets. Let us remove these barcodes prior to further quality control. We use emptyDrops to compute if a cell is a cell or an empty droplet.

It must be noted that CellRanger 3.0 has incorporated the EmptyDrops algorithm to distinguish cells from empty droplets.

Prepare input for EmptyDrops.
"""

sparse_mat = adata_raw.X.T
genes = adata_raw.var_names
barcodes = adata_raw.obs_names


# TODO run this R stuff
"""Run EmptyDrops."""
nr, nc = sparse_mat.shape
ro.r("library(Matrix)")
ro.r.sparseMatrix(
    i=ro.IntVector(nr + 1),
    j=ro.IntVector(nc + 1),
    x=ro.FloatVector(sparse_mat.data),
    dims=ro.IntVector(sparse_mats.hape))
# ro.r.matrix(sparse_mat, nrow=nr, ncol=nc)

# Commented out IPython magic to ensure Python compatibility.
# %%R -i sparse_mat -i genes -i barcodes -o barcodes_filtered -o ambient_genes
# 
# sce <- SingleCellExperiment(assays = list(counts = sparse_mat), colData=barcodes)
# rownames(sce) <- genes 
# ambient <- emptyDrops(counts(sce))
# is_cell <- ambient$FDR <= 0.05 #False discovery rate for cells
# threshold_ambient <- 0.005 #threshold level of ambient RNA
# ambient_genes <- names(ambient@metadata$ambient[ambient@metadata$ambient> threshold_ambient,])
# barcodes_filtered <- barcodes[which(is_cell)]

# """Empty drops returns a list of potentially ambient genes and the barcodes, which belong to actual cells."""

# print(ambient_genes)

# print(barcodes_filtered)

# """Let us create a filtered data matrix using the filtered barcodes."""

# adata = adata_raw[np.in1d(adata_raw.obs_names, barcodes_filtered)].copy()

# adata

# """**BONUS**: Examine the level of background gene expression.

# Save the filtered data set to file.
# """

# adata.write(data_dir + 'data_filtered.h5ad')

# """**COMMENT:** End of first session.

# ## Compute quality control metrics

# Read data from file to begin with the quality control.
# """

# adata = sc.read(data_dir + 'data_filtered.h5ad')

# """Data quality control can be split into cell QC and gene QC. Typical quality measures for assessing the quality of a cell include the number of molecule counts (UMIs), the number of expressed genes, and the fraction of counts that are mitochondrial. A high fraction of mitochondrial reads being picked up can indicate cell stress, as there is a low proportion of nuclear mRNA in the cell. It should be noted that high mitochondrial RNA fractions can also be biological signals indicating elevated respiration.

# `Scanpy` provides the `calculate_qc_metrics` function, which computes the following QC metrics:
# On the cell level (`.obs` level):
# * `n_genes_by_counts`: Number of genes with positive counts in a cell
# * `log1p_n_genes_by_counts`: Log(n+1) transformed number of genes with positive counts in a cell
# * `total_counts`: Total number of counts for a cell
# * `log1p_total_counts`: Log(n+1) transformed total number of counts for a cell
# * `pct_counts_in_top_50_genes`: Cumulative percentage of counts for 50 most expressed genes in a cell
# * `pct_counts_in_top_100_genes`: Cumulative percentage of counts for 100 most expressed genes in a cell
# * `pct_counts_in_top_200_genes`: Cumulative percentage of counts for 200 most expressed genes in a cell
# * `pct_counts_in_top_500_genes`: Cumulative percentage of counts for 500 most expressed genes in a cell

# On the gene level (`.var` level):
# * `n_cells_by_counts`: Number of cells this expression is measured in
# * `mean_counts`: Mean expression over all cells
# * `log1p_mean_counts`: Log(n+1) transformed mean expression over all cells
# * `pct_dropout_by_counts`: Percentage of cells this feature does not appear in
# * `total_counts`: Sum of counts for a gene
# * `log1p_total_counts`: Log(n+1) transformed sum of counts for a gene
# """

# sc.pp.calculate_qc_metrics(adata, inplace=True)

# """We further aim to determine the fraction of mitochondrial counts per cell.
# Please note that mitochondrial genes in human start with 'MT-'
# """

# mt_gene_mask = np.flatnonzero([gene.startswith('MT-') for gene in adata.var_names])
# # the `.A1` is only necessary as X is sparse (to transform to a dense array after summing)
# adata.obs['mt_frac'] = np.sum(adata[:, mt_gene_mask].X, axis=1).A1/adata.obs['n_counts']

# """Let us visualize the number of expressed genes and the number of counts as a scatter plot.

# **Task:** Create a scatter plot with the library size against the number of genes. Create a second plot, where you only show cells with a library size of less than 10,000 counts. Color by the fraction of mitochondrial reads. 

# **Questions:** How can we describe the relation of library size vs number of expressed genes vs mitochondrial reads?
# """

# #Data quality summary plots
# p1 = sc.pl.scatter(adata = , x='' , y='', color='', size=40)
# #hint: temporary subsetting of the anndata object by .obs works like adata[adata.obs['key']<value]
# p2 = sc.pl.scatter(adata = , x='', y='', 
#                    color='', size=40)

# """**Task:** Below, you find a the code to create a violin plot of the library size. Create another two violin plots displaying the number of genes and the fraction of mitochondrial reads.

# **Questions:** How do the count data distribute within the sample? 

# """

# #Sample quality plots
# rcParams['figure.figsize']=(7,7)
# t1 = sc.pl.violin(adata, 'n_counts',
#                   #groupby='sample',
#                   size=2, log=True, cut=0)
# t1 = sc.pl.violin() #display number of genes
# t2 = sc.pl.violin() #display the fraction of mitochondrial reads

# """Examine the overall library complexity. 

# **Task:** Plot the top 20 highest expressed genes. Use the function `sc.pl.highest_expr_genes`. 

# **Questions:** Which genes do you find? Are they specific for a cell type?
# """

# ?sc.pl.highest_expr_genes #? before a function opens the documentation of this function



# """**Task:** How many counts come from the top 50/100/200/500 highest expressed genes? Visualize the fraction as violin plot. (Hint: Already computed with the `calculate_qc_metrics` function.)"""

# #Sample quality plots
# rcParams['figure.figsize']=(7,7) #set figure size
# t3 = sc.pl.violin(adata, keys = ['', '', '', ''], rotation = 90)

# """**BONUS:** Visualize the log-transformed total counts vs the log-transformed number of expressed genes with distribution plots on the side of each axis. (Hint: Use the `jointplot` from the `seaborn` package).

# **Conclusions:** By looking at plots of the number of genes versus the number of counts with MT fraction information, we can assess whether there are cells with unexpected summary statistics. It is important here to look at these statistics jointly.  We should probably still filter out some cells with very few genes as these may be difficult to annotate later. This will be true for the initial cellular density between 1000-4000 counts and < ~500 genes.

# Furthermore it can be seen in the main cloud of data points, that cells with lower counts and genes tend to have a higher fraction of mitochondrial counts. These cells are likely under stress or are dying. When apoptotic cells are sequenced, there is less mRNA to be captured in the nucleus, and therefore fewer counts overall, and thus a higher fraction of counts fall upon mitochondrial RNA. If cells with high mitochondrial activity were found at higher counts/genes per cell, this would indicate biologically relevant mitochondrial activity.

# **Task:** Create a histogram for the total number of counts. Further, create a histogram for the low count and high count regime, each. Here, you have to decide on a reasonable threshold.

# Note: `pandas` does some histogram plotting with `adata.obs['n_counts'].hist()`, howecer, you will obtain prettier plots with `distplot` from `seaborn`.
# """

# #Thresholding decision: counts
# rcParams['figure.figsize']=(20,5)
# fig_ind=np.arange(131, 134)
# fig = plt.figure()
# fig.subplots_adjust(hspace=0.4, wspace=0.6)

# p3 = sb.distplot(adata.obs['total_counts'], 
#                  kde=False, 
#                  ax=fig.add_subplot(fig_ind[0]))
# p4 = sb.distplot( #histogram for low count regime
#                  ax=fig.add_subplot(fig_ind[1]))
# p5 = sb.distplot( #histogram for high count regime
#                  ax=fig.add_subplot(fig_ind[2]))
# plt.show()

# """**Conclusions:**
# Histograms of the number of counts per cell show a small peak of groups of cells with fewer than **XX** counts, which are likely uninformative given the overall distribution of counts. This may be cellular debris found in droplets.

# On the upper end of counts, we see a population of cells with high counts with decaying slope at **XX** counts. We estimate this population to range until **XX** counts. This estimation is performed by visually tracing a Gaussian around the population.

# **Task:** Create a histogram for the total number of genes. Further, create a histogram for the low gene count regime.
# """

# #Thresholding decision: genes

# rcParams['figure.figsize']=(20,5)
# fig_ind=np.arange(131, 133)
# fig = plt.figure()
# fig.subplots_adjust(hspace=0.4, wspace=0.6) #create a grid for subplots

# p6 = sb.distplot(adata.obs['n_genes_by_counts'], 
#                  kde=False, bins=60, ax=fig.add_subplot(fig_ind[0]))
# #low number of genes regime
# p7 = sb.distplot( ax=fig.add_subplot(fig_ind[1]))
# plt.show()

# """**Conclusions:**
# Two populations of cells with low gene counts can be seen in the above plots. Given these plots, and the plot of genes vs counts above, we decide to filter out cells with fewer than **XX** genes expressed. Below this we are likely to find dying cells or empty droplets with ambient RNA. Looking above at the joint plots, we see that we filter out the main density of low gene cells with this threshold.

# In general it is a good idea to be permissive in the early filtering steps, and then come back to filter out more stringently when a clear picture is available of what would be filtered out. This is available after visualization/clustering. For demonstration purposes we stick to a simple (and slightly more stringent) filtering here.

# **Task:** Create a histogram for the fraction of mitochondrial genes. Further, create a histogram for the high fraction regime.
# """

# #Thresholding decision: mitochondrial reads

# rcParams['figure.figsize']=(20,5)
# fig_ind=np.arange(131, 133)
# fig = plt.figure()
# fig.subplots_adjust(hspace=0.4, wspace=0.6)

# p8 = sb.distplot(  #display the fraction of mitochondrial reads
#                  ax=fig.add_subplot(fig_ind[0]))

# p9 = sb.distplot(  #display the fraction of mitochondrial reads for the high fraction (in this case a threshold of 0.2 as high)
#                  ax=fig.add_subplot(fig_ind[1]))
# plt.show()

# """**Task:** Filter your cells according for the total number of counts, number of expressed genes and fraction of mitochondrial reads. Check the number of remaining cells after each filtering step."""

# # Filter cells according to identified QC thresholds:
# print('Total number of cells: {:d}'.format(adata.n_obs))

# sc.pp.filter_cells(adata, min_counts = )
# print('Number of cells after min count filter: {:d}'.format(adata.n_obs))

# sc.pp.filter_cells(adata, max_counts = )
# print('Number of cells after max count filter: {:d}'.format(adata.n_obs))

# adata = adata[adata.obs['mt_frac'] < ]
# print('Number of cells after MT filter: {:d}'.format(adata.n_obs))

# sc.pp.filter_cells(adata, min_genes = )
# print('Number of cells after gene filter: {:d}'.format(adata.n_obs))

# """**Task:** Next, filter out non-expressed genes. Check the number of remaining genes after filtering."""

# #Filter genes:
# print('Total number of genes: {:d}'.format(adata.n_vars))

# # Min 20 cells - filters out 0 count genes
# sc.pp.filter_genes(adata, )

# print('Number of genes after cell filter: {:d}'.format(adata.n_vars))

# """The filtering is performed based on the thresholds we identified from the QC plots. Genes are also filtered if they are not detected in at least **XX** cells. This reduces the dimensions of the matrix by removing 0 count genes and genes which are not sufficiently informative of the dataset.

# ### Doublet score

# Let us estimate the amount of doublets in the dataset. Here, we use the tool `scrublet` that simulates doublet gene expression profiles based on the data. We apply it for each sample separately.
# """

# adata.obs['doublet_score']= np.zeros(adata.shape[0])
# adata.obs['doublet'] = np.zeros(adata.shape[0])

# # filtering/preprocessing parameters:
# min_counts = 2
# min_cells = 3
# vscore_percentile = 85
# n_pc = 50

# # doublet detector parameters:
# expected_doublet_rate = 0.02 
# sim_doublet_ratio = 3
# n_neighbors = 15



# scrub = scr.Scrublet(counts_matrix = adata.X,  
#                      n_neighbors = n_neighbors,
#                      sim_doublet_ratio = sim_doublet_ratio,
#                      expected_doublet_rate = expected_doublet_rate)

# doublet_scores, predicted_doublets = scrub.scrub_doublets( 
#                     min_counts = min_counts, 
#                     min_cells = min_cells, 
#                     n_prin_comps = n_pc,
#                     use_approx_neighbors = True, 
#                     get_doublet_neighbor_parents = False)

# adata.obs['doublet_score'] = doublet_scores
# adata.obs['doublet'] = predicted_doublets

# """**Tasks:** Plot the doublet score as a histogram and as violin plot. """

# rcParams['figure.figsize']=(6,6)
# sb.distplot() #histogram of the doublet score
# plt.show()

# rcParams['figure.figsize']=(15,7)
# sc.pl.violin() #violin plot of the doublet score

# """### filtering doublets

# Scrublet proposed a different threshold than we would choose based upon the histogram plot of the doublet scores.

# **Tasks:** Decide on a threshold to filter doublets.
# """

# thr = #add threshold
# ix_filt = adata.obs['doublet_score']<=thr

# adata = adata[ix_filt].copy()
# print('Number of cells after doublet filter: {:d}'.format(adata.n_obs))

# """### Summarize sample information

# In order to group by `batch` (for future purposes, because we presently deal with one sample), let us add a `batch` covariate to the `adata` object.
# """

# adata.obs['batch'] = '1'

# df = adata.obs[['n_genes_by_counts','total_counts', 'batch']]
# df_all = pd.DataFrame(df.groupby(by='batch')['n_genes_by_counts'].apply(np.mean).values,
#                       index=df.groupby(by='batch')['n_genes_by_counts'].apply(np.mean).index,
#                       columns=['mean_genes'])

# df_all['median_genes']=df.groupby(by='batch')['n_genes_by_counts'].apply(np.median).values
# df_all['mean_counts']=df.groupby(by='batch')['total_counts'].apply(np.mean).values
# df_all['median_counts']=df.groupby(by='batch')['total_counts'].apply(np.median).values

# """Display table."""

# df_all

# """**Task:** Save the summary table to file (`csv` or `xlsx` format) to the `tables` subdirectory. """



# """Save post-QC data to file."""

# adata.write(data_dir + 'data_postQC.h5ad')

# """**COMMENT:** End of second session and day 1. """