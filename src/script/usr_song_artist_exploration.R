## see recorded days inlude: usr-song, all_usr-song, usr-artist, all_usr-artist
## data.table:
##   dt_usr_info(uid,sid,gmt,action,Ds)
##   dt_song_info(sid,artid,pub_time,ini_play,lang,gender)
##   dt_usr_song_info(uid,sid,gmt,action,Ds,artid,pub_time,ini_play,lang,gender)
##
##   dt_usr_song(uid,sid,days)       dt_usr_song_sorted(uid,sid,days)
##   dt_song(sid,days)               dt_song_sorted(sid,days)
##   dt_usr_artist(uid,artid,days)   dt_usr_artist_sorted(uid,artid,days)
##   dt_artist(artid,days)           dt_artist_sorted(artid,days)
##
## csv file output (under data/ folder, no output default, if need ouptut uncomment realted code):
##   usr_song.csv        usr_song_sorted.csv
##   song.csv            song_sorted.csv
##   usr_artist.csv      usr_artist_sorted.csv
##   artist.csv          artist_sorted.csv
##
## jpeg file output (under data/ folder):
##   usr_song.jpg
##   song.jpg
##   usr_artist.jpg
##   artist.jpg


if(substr(getwd(),nchar(getwd())-3,nchar(getwd()))!="data"){
  setwd("../../data/")
}

require(data.table)

dt_usr_info<-fread("mars_tianchi_user_actions.csv",header=F)
colnames(dt_usr_info)<-c("uid","sid","gmt","action","Ds")
setkey(dt_usr_info,uid,sid)

## usr-song
dt_usr_song<-dt_usr_info[,list(days=length(unique(Ds))),by="uid,sid"]
#write.table(dt_usr_song,"usr_song.csv",sep=",",row.names=FALSE,quote=FALSE)
dt_usr_song_sorted<-dt_usr_song[order(days,decreasing=T)]
#write.table(dt_usr_song_sorted,"usr_song_sorted.csv",sep=",",row.names=FALSE,quote=FALSE)
jpeg('usr_song.jpg')
plot(1:nrow(dt_usr_song_sorted),dt_usr_song_sorted$days,xlab="(uid,sid)_pairs",ylab="recorded_days")
dev.off()

## all_usr-song
dt_song<-dt_usr_info[,list(days=length(unique(Ds))),by="sid"]
#write.table(dt_song,"song.csv",sep=",",row.names=FALSE,quote=FALSE)
dt_song_sorted<-dt_song[order(days,decreasing=T)]
#write.table(dt_song_sorted,"song_sorted.csv",sep=",",row.names=FALSE,quote=FALSE)
jpeg('song.jpg')
plot(1:nrow(dt_song_sorted),dt_song_sorted$days,xlab="sid",ylab="recorded_days")
dev.off()

dt_song_info<-fread("mars_tianchi_songs.csv",header=F)
colnames(dt_song_info)<-c("sid","artid","pub_time","ini_play","lang","gender")
setkey(dt_song_info,sid)
dt_usr_song_info<-merge(dt_usr_info,dt_song_info,by="sid")
setkey(dt_usr_song_info,uid,artid)

## usr-artist
dt_usr_artist<-dt_usr_song_info[,list(days=length(unique(Ds))),by="uid,artid"]
#write.table(dt_usr_artist,"usr_artist.csv",sep=",",row.names=FALSE,quote=FALSE)
dt_usr_artist_sorted<-dt_usr_artist[order(days,decreasing=T)]
#write.table(dt_usr_artist_sorted,"usr_artist_sorted.csv",sep=",",row.names=FALSE,quote=FALSE)
jpeg('usr_artist.jpg')
plot(1:nrow(dt_usr_artist_sorted),dt_usr_artist_sorted$days,xlab="(uid,artistid)_pair",ylab="recorded_days")
dev.off()

## all_usr-artist
dt_artist<-dt_usr_song_info[,list(days=length(unique(Ds))),by="artid"]
#write.table(dt_artist,"artist.csv",sep=",",row.names=FALSE,quote=FALSE)
dt_artist_sorted<-dt_artist[order(days,decreasing=T)]
#write.table(dt_artist_sorted,"artist_sorted.csv",sep=",",row.names=FALSE,quote=FALSE)
jpeg('artist.jpg')
plot(1:nrow(dt_artist_sorted),dt_artist_sorted$days,xlab="artistid",ylab="recorded_days")
dev.off()