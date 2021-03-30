library(ggplot2)
library(gridExtra)
library(readxl)
library(mclust)


setwd('~/OneDrive/Studies/MSc_Thesis/CCSL-ID/')
basepath = './data/sachs/Data Files/'

obs = read_xls(paste0(basepath, '1. cd3cd28.xls'))
names(obs) = sapply(names(obs), toupper)
obs$type = 'obs'
print(paste('obs', dim(obs)))
int = read_excel(paste0(basepath, '8. pma.xls'))
names(int) = sapply(names(int), toupper)
int$type = 'int'
print(paste('int', dim(int)))

df = rbind(obs, int)

ggplot(df) +
  geom_point(aes(x=PKC, y=PRAF, color=type), alpha=0.3) 

ggplot(df) +
  geom_density(aes(x=PKC)) +
  geom_density(aes(x=PRAF))

pkc.model = Mclust(df$PKC, 3)
plot(pkc.model, what='density', main='Mclust Classification')
pkc.model$loglik

praf.model = Mclust(df$PRAF, 3)
plot(praf.model, what='density')
praf.model$loglik
