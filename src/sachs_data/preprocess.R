###
# Read and prepare sachs data for experiment
###

library(bnlearn)
library(qgraph)
library(readxl)
library(dplyr)
library(R.utils)
rm(list=ls())

# download from primary source
download.file("https://science.sciencemag.org/highwire/filestream/586570/field_highwire_adjunct_files/1/Sachs.SOM.Datasets.zip", destfile="sachs.zip")
unzip("sachs.zip")

# get filenames
parent_dir = "Data Files"
files = list.files(parent_dir)

### NoTears uses all main results - that doesn't make any sense since they are interventional!!!
# # only use main results (just like notears; see .doc for more info)
# files = files[endsWith(files, ".xls") & sapply(files, function(x){substr(as.character(x), 2, 2) == "."})]

files = c("1. cd3cd28.xls")
files = sapply(files, function(x) {file.path(parent_dir, x)})

load_pp = function(fname){
  df = read_xls(fname)
  if ("pip2" %in% colnames(df)) df$PIP2 = df$pip2; df$pip2 = NULL
  if ("pip3" %in% colnames(df)) df$PIP3 = df$pip3; df$pip3 = NULL
  return(df)
}

dfs = lapply(files, load_pp)
sachs_raw = bind_rows(dfs)

# align naming convention
sachs = sachs_raw %>% 
  rename(
    Mek = pmek,
    Raf = praf,
    Plcg = plcg,
    Jnk = pjnk,
    Erk = !!"p44/42",
    Akt = pakts473
  )

# order columns alphabetically
sachs = sachs[,order(colnames(sachs))]
print(sachs)

# comparison with secondary source for data integrity
download.file("www.bnlearn.com/book-crc/code/sachs.data.txt.gz", destfile="sachs_bnlearn.txt.gz")
gunzip("sachs_bnlearn.txt.gz")
sachs_secondary = read.table("sachs_bnlearn.txt", header=TRUE)
sachs_secondary = sachs_secondary[, order(colnames(sachs_secondary))]
stopifnot(sum(sachs == sachs_secondary) == prod(dim(sachs)))
stopifnot(sum(colnames(sachs) == colnames(sachs_secondary)) == dim(sachs)[2])

# save ordered data
write.csv(sachs, "sachs.csv", row.names=FALSE)

# sachs consensus network
sachs.modelstring <-
  paste("[PKC][PKA|PKC][Raf|PKC:PKA][Mek|PKC:PKA:Raf]",
        "[Erk|Mek:PKA][Akt|Erk:PKA][P38|PKC:PKA]",
        "[Jnk|PKC:PKA][Plcg][PIP3|Plcg][PIP2|Plcg:PIP3]",sep="")
dag.sachs <- model2network(sachs.modelstring)
qgraph(dag.sachs)

# consensus adjacency matrix (is correct format np.triu)
adj_mat = amat(dag.sachs)
print(adj_mat)

# column name check for data integrity
stopifnot(sum(colnames(sachs) == colnames(adj_mat)) == dim(adj_mat)[1])

# save consensus adjacency matrix
write.csv(adj_mat, "consensus_adj_mat.csv", row.names=TRUE)

