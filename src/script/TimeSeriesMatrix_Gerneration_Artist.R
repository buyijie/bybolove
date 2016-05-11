# generate time series matrix, each column for an artist
library(data.table)
library(forecast)
library(TSA)
library(vars)
ts_all_artist<-fread("ts_artist_play.csv",header=T)
ts_all_artist[is.na(ts_all_artist$op_num)]$op_num<-0
artist_list<-fread("artist_list.csv",header=T)

ts_matrix<-ts_all_artist[artid==artist_list[1]]$op_num
for (i in 2:50){
  ts_matrix<-cbind(ts_matrix, ts_all_artist[artid==artist_list[i]]$op_num)
}
colnames(ts_matrix)<-artist_list$artid
write.csv(ts_matrix, 'ts_artist_play_matrix.csv', row.names=FALSE, quote=FALSE)