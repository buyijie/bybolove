## This script generate time series for artist-all_action, artist-play, artist-download, artist-favorite.
## data.table:
##   ts_artist(art_id,Ds,op_num)
##   ts_artist_play(art_id,Ds,op_num)
##   ts_artist_download(art_id,Ds,op_num)
##   ts_artist_favorite(art_id,Ds,op_num)
##   artist_list(art_id)
## csv file: 
##   ts_artist.csv
##   ts_artist_play.csv
##   ts_artist_download.csv
##   ts_artist_favorite.csv
##   artist_list.csv
##
## !!! In some date, some artist don't have record, op_num of this row is set as NA !!!!!!

if(substr(getwd(),nchar(getwd())-3,nchar(getwd()))!="data"){
  setwd("../../data/")
}

require(data.table)

complete_date<-function(ts,artist_list,date_list){
  for (artid_tmp in artist_list){
    date_lack<-date_list[! date_list %in% ts[artid==artid_tmp]$Ds]
    row_tmp<-length(date_lack)
    dt_new<-data.table(rep(artid_tmp,row_tmp),date_lack,rep(NA,row_tmp))
    colnames(dt_new)<-colnames(ts)
    ts<-rbind(ts,dt_new)
  }
  return(ts)
}

dt_usr_info<-fread("mars_tianchi_user_actions.csv",header=F)
colnames(dt_usr_info)<-c("uid","sid","gmt","action","Ds")
setkey(dt_usr_info,sid)

dt_song_info<-fread("mars_tianchi_songs.csv",header=F)
colnames(dt_song_info)<-c("sid","artid","pub_time","ini_play","lang","gender")
setkey(dt_song_info,sid)

dt_usr_song_info<-merge(dt_usr_info,dt_song_info,by="sid")
dt_usr_song_info[,c("uid","sid","gmt","pub_time","ini_play","lang","gender"):=NULL]
setkey(dt_usr_song_info,action,artid,Ds)

artist_list<-data.table(artid=unique(dt_usr_song_info$artid))
setkey(artist_list,artid)
write.table(artist_list,"artist_list.csv",sep=",",row.names=FALSE,quote=FALSE)

date_list<-data.table(Ds=unique(dt_usr_song_info$Ds))
setkey(date_list,Ds)

ts_artist<-dt_usr_song_info[,list(op_num=.N),by="artid,Ds"]
setkey(ts_artist,artid,Ds)
ts_artist<-complete_date(ts_artist,artist_list$artid,date_list$Ds)
setkey(ts_artist,artid,Ds)
write.table(ts_artist,"ts_artist.csv",sep=",",row.names=FALSE,quote=FALSE)

ts_artist_play<-dt_usr_song_info[action==1][,list(op_num=.N),by="artid,Ds"]
setkey(ts_artist_play,artid,Ds)
ts_artist_play<-complete_date(ts_artist_play,artist_list$artid,date_list$Ds)
setkey(ts_artist_play,artid,Ds)
write.table(ts_artist_play,"ts_artist_play.csv",sep=",",row.names=FALSE,quote=FALSE)

ts_artist_download<-dt_usr_song_info[action==2][,list(op_num=.N),by="artid,Ds"]
setkey(ts_artist_download,artid,Ds)
ts_artist_download<-complete_date(ts_artist_download,artist_list$artid,date_list$Ds)
setkey(ts_artist_download,artid,Ds)
write.table(ts_artist_download,"ts_artist_download.csv",sep=",",row.names=FALSE,quote=FALSE)

ts_artist_favorite<-dt_usr_song_info[action==3][,list(op_num=.N),by="artid,Ds"]
setkey(ts_artist_favorite,artid,Ds)
ts_artist_favorite<-complete_date(ts_artist_favorite,artist_list$artid,date_list$Ds)
setkey(ts_artist_favorite,artid,Ds)
write.table(ts_artist_favorite,"ts_artist_favorite.csv",sep=",",row.names=FALSE,quote=FALSE)






