## plot time series for op_num of every artist in 6 months given.
## we can try to find some patterns by observing time series plots.

if(substr(getwd(),nchar(getwd())-3,nchar(getwd()))!="data"){
  setwd("../../data/")
}

require(data.table)

dt<-fread("ts_artist_play.csv",header=T)  ## plot other files, ts_artist.csv, ts_artist_download...
artist_list<-fread("artist_list.csv",header=T)
dt[is.na(dt$op_num)]$op_num<-0

plot_ts<-function(art){
  ts<-ts(dt[artid==artist_list[art]]$op_num)
  plot(ts,type="b",main=art)
}

dt_song_info<-fread("mars_tianchi_songs.csv",header=F)
colnames(dt_song_info)<-c("sid","artid","pub_time","ini_play","lang","gender")
setkey(dt_song_info,artid,pub_time)

date_list<-data.table(Ds=unique(dt$Ds),rank=c(1:183))
setkey(date_list,Ds)

dt_pub_in_range<-dt_song_info[dt_song_info$pub_time %in% date_list$Ds]
setkey(dt_pub_in_range,artid,pub_time)

dt_pub_in_range<-merge(dt_pub_in_range,date_list,by.x="pub_time",by.y="Ds")
setkey(dt_pub_in_range,artid,rank)

for (i in 1:50){
  jpeg(paste("ts_play_",i,".jpg",sep=""))   ## plot other files, ts_artist.csv, ts_artist_download...
  plot_ts(i)
  abline(v=dt_pub_in_range[artid==artist_list[i]]$rank,lty="dotted")
  dev.off()
}

