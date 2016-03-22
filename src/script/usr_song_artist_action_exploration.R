## see recorded days inlude: usr-song-play, usr-song-download, usr-song-favoriate,
##                           all_usr-song-play, all_usr-song-download, all_usr-song-favorite,
##                           usr-artist-play, usr-artist-download, usr-artist-favorite, 
##                           all_usr-artist-play, all_usr-artist-download, all_usr-artist-favorite.
## data.table:
##   dt_usr_info(uid,sid,gmt,action,Ds)
##   dt_song_info(sid,artid,pub_time,ini_play,lang,gender)
##   dt_usr_song_info(uid,sid,gmt,action,Ds,artid,pub_time,ini_play,lang,gender)
##
##   dt_action_usr_song(uid,sid,days)       dt_action_usr_song_sorted(uid,sid,days)
##   dt_action_song(sid,days)               dt_action_song_sorted(sid,days)
##   dt_action_usr_artist(uid,artid,days)   dt_action_usr_artist_sorted(uid,artid,days)
##   dt_action_artist(artid,days)           dt_action_artist_sorted(artid,days)
##
## csv file output (under data/ folder, no output default, if need ouptut uncomment realted code):
##   action_usr_song.csv        action_usr_song_sorted.csv
##   action_song.csv            action_song_sorted.csv
##   action_usr_artist.csv      action_usr_artist_sorted.csv
##   action_artist.csv          action_artist_sorted.csv
##
## jpeg file output (under data/ folder):
##   usr_song_play.jpg    usr_song_download.jpg    usr_song_favorite.jpg
##   song_play.jpg        usr_song_download.jpg    usr_song_favorite.jpg
##   usr_artist_play.jpg  usr_artist_download.jpg  usr_artist_favorite.jpg
##   artist_play.jpg      artist_download.jpg      artist_favorite.jpg

if(substr(getwd(),nchar(getwd())-3,nchar(getwd()))!="data"){
  setwd("../../data/")
}

require(data.table)

dt_usr_info<-fread("mars_tianchi_user_actions.csv",header=F)
colnames(dt_usr_info)<-c("uid","sid","gmt","action","Ds")
setkey(dt_usr_info,action,uid,sid)

## action-usr-song
dt_action_usr_song<-dt_usr_info[,list(days=length(unique(Ds))),by="action,uid,sid"]
#write.table(dt_action_usr_song,"action_usr_song.csv",sep=",",row.names=FALSE,quote=FALSE)
dt_action_usr_song_sorted<-dt_action_usr_song[order(action,-days)]
setkey(dt_action_usr_song_sorted,action)
#write.table(dt_action_usr_song_sorted,"action_usr_song_sorted.csv",sep=",",row.names=FALSE,quote=FALSE)
jpeg('usr_song_play.jpg')
plot(1:nrow(dt_action_usr_song_sorted[action==1]),dt_action_usr_song_sorted[action==1]$days,xlab="(uid,sid)_pairs",ylab="recorded_days_play")
dev.off()
jpeg('usr_song_download.jpg')
plot(1:nrow(dt_action_usr_song_sorted[action==2]),dt_action_usr_song_sorted[action==2]$days,xlab="(uid,sid)_pairs",ylab="recorded_days_download")
dev.off()
jpeg('usr_song_favorite.jpg')
plot(1:nrow(dt_action_usr_song_sorted[action==3]),dt_action_usr_song_sorted[action==3]$days,xlab="(uid,sid)_pairs",ylab="recorded_days_favorite")
dev.off()

## action-all_usr-song
dt_action_song<-dt_usr_info[,list(days=length(unique(Ds))),by="action,sid"]
#write.table(dt_action_song,"action_song.csv",sep=",",row.names=FALSE,quote=FALSE)
dt_action_song_sorted<-dt_action_song[order(action,-days)]
setkey(dt_action_song_sorted,action)
#write.table(dt_action_song_sorted,"action_song_sorted.csv",sep=",",row.names=FALSE,quote=FALSE)
jpeg('song_play.jpg')
plot(1:nrow(dt_action_song_sorted[action==1]),dt_action_song_sorted[action==1]$days,xlab="sid",ylab="recorded_days_play")
dev.off()
jpeg('song_download.jpg')
plot(1:nrow(dt_action_song_sorted[action==2]),dt_action_song_sorted[action==2]$days,xlab="sid",ylab="recorded_days_download")
dev.off()
jpeg('song_favorite.jpg')
plot(1:nrow(dt_action_song_sorted[action==3]),dt_action_song_sorted[action==3]$days,xlab="sid",ylab="recorded_days_favorite")
dev.off()

dt_song_info<-fread("mars_tianchi_songs.csv",header=F)
colnames(dt_song_info)<-c("sid","artid","pub_time","ini_play","lang","gender")
setkey(dt_song_info,sid)
dt_usr_song_info<-merge(dt_usr_info,dt_song_info,by="sid")
setkey(dt_usr_song_info,action,uid,artid)

## action-usr-artist
dt_action_usr_artist<-dt_usr_song_info[,list(days=length(unique(Ds))),by="action,uid,artid"]
#write.table(dt_action_usr_artist,"action_usr_artist.csv",sep=",",row.names=FALSE,quote=FALSE)
dt_action_usr_artist_sorted<-dt_action_usr_artist[order(action,-days)]
setkey(dt_action_usr_artist_sorted,action)
#write.table(dt_action_usr_artist_sorted,"action_usr_artist_sorted.csv",sep=",",row.names=FALSE,quote=FALSE)
jpeg('usr_artist_play.jpg')
plot(1:nrow(dt_action_usr_artist_sorted[action==1]),dt_action_usr_artist_sorted[action==1]$days,xlab="(uid,artistid)_pair",ylab="recorded_days_play")
dev.off()
jpeg('usr_artist_download.jpg')
plot(1:nrow(dt_action_usr_artist_sorted[action==2]),dt_action_usr_artist_sorted[action==2]$days,xlab="(uid,artistid)_pair",ylab="recorded_days_download")
dev.off()
jpeg('usr_artist_favorite.jpg')
plot(1:nrow(dt_action_usr_artist_sorted[action==3]),dt_action_usr_artist_sorted[action==3]$days,xlab="(uid,artistid)_pair",ylab="recorded_days_favorite")
dev.off()

## action-all_user-artist
dt_action_artist<-dt_usr_song_info[,list(days=length(unique(Ds))),by="action,artid"]
#write.table(dt_action_artist,"action_artist.csv",sep=",",row.names=FALSE,quote=FALSE)
dt_action_artist_sorted<-dt_action_artist[order(action,-days)]
setkey(dt_action_artist_sorted,action)
#write.table(dt_action_artist_sorted,"action_artist_sorted.csv",sep=",",row.names=FALSE,quote=FALSE)
jpeg('artist_play.jpg')
plot(1:nrow(dt_action_artist_sorted[action==1]),dt_action_artist_sorted[action==1]$days,xlab="artistid",ylab="recorded_days_play")
dev.off()
jpeg('artist_download.jpg')
plot(1:nrow(dt_action_artist_sorted[action==2]),dt_action_artist_sorted[action==2]$days,xlab="artistid",ylab="recorded_days_download")
dev.off()
jpeg('artist_favorite.jpg')
plot(1:nrow(dt_action_artist_sorted[action==3]),dt_action_artist_sorted[action==3]$days,xlab="artistid",ylab="recorded_days_favorite")
dev.off()