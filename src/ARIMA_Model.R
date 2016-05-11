#------------------------------score 5687.28931 log       1 stationary=T floor-----------------
#------------------------------score 5682.39148 log       1 stationary=T-----------------------
#------------------------------score 5673.88095 frequency 1 stationary=T-----------------------
#------------------------------score 5182.02237 frequency 1------------------------------------
#------------------------------score 4971.27051 frequency 7------------------------------------
#------------------------------score 5098.58481 frequency 15-----------------------------------
library(data.table)
library(forecast)
library(TSA)
library(vars)
ts_all_artist<-fread("ts_artist_play.csv",header=T)
ts_all_artist[is.na(ts_all_artist$op_num)]$op_num<-0
artist_list<-fread("artist_list.csv",header=T)

ts_list<-list()
for (i in 1:50){
  ts_list[[i]]<-ts(ts_all_artist[artid==artist_list[i]]$op_num,frequency=1)
}

ts_matrix<-ts_list[[1]]
for (i in 2:50){
  ts_matrix<-cbind(ts_matrix,ts_list[[i]])
}
ts_matrix<-ts_matrix+1
colnames(ts_matrix)<-c(1:50)

#plot_fit_result<-function(idx){
#  plot(ts_matrix[,idx],xlim=c(1,40))
#  lines(fore$mean[[idx]],col="red")
#  lines(fore$fitted[,idx],col="green")
#}

evaluate<-function(fore,i){
  ## return a F_score
  ## large F_score is good
  test_set<-ts_matrix[123:183,i]
  sigma<-0
  phi<-0
  F_score<-0
  sigma<-sqrt(sum(((test_set-c(floor(fore)))/test_set)^2)/61)
  phi<-sqrt(sum(test_set))
  F_score<-(1-sigma)*phi
  return (F_score)
}

#------------------ARIMA Model use 3-6 month predict 7-8 month-------------
result<-rep(0,51)
for (i in 1:50){
  fit<-auto.arima(log(window(ts_matrix[,i],start=c(1,1),end=c(122,1))),stationary=T)
  fore<-forecast(fit,h=61)
  result[i]<-evaluate(exp(fore$mean),i)
  result[51]<-result[51]+result[i]
}

