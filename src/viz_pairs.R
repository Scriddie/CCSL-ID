library(ggplot2)
library(gridExtra)
setwd('~/OneDrive/Studies/MSc_Thesis/CCSL-ID/')

show = function(i){
  filler = paste(rep('0', 4 - nchar(as.character(i))), collapse='')
  print(paste0(filler, i))
  path = paste0('./data/pairs/pair', filler, i, '.txt')
  t = read.table(path)
  # par(mfrow=c(1,3)) 
  # d1 = ggplot(t) + geom_density(aes(V1))
  # d2 = ggplot(t) + geom_density(aes(V2))
  # grid.arrange(d1, d2, ncol=2)
  # plot(t$V1, t$V2)
  plt = ggplot(t) +
    geom_point(aes(x=V1, y=V2))
  print(plt)
}

show(2)

# for (i in 1:9){
  # show(i)
  # Sys.sleep(5)
# }

# show(8)
