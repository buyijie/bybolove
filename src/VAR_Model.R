#--------------------------------score 5775.22677 type="const" p=1 log floor---------------
#--------------------------------score 5765.79952 type="const" p=1 log---------------
#--------------------------------score 5387.430296 type="const"----------------------
#--------------------------------score 4179.377815-----------------------------------
library(data.table)
library(forecast)
library(TSA)
library(vars)
ts_all_artist<-fread("ts_artist_play.csv",header=T)
ts_all_artist[is.na(ts_all_artist$op_num)]$op_num<-0
artist_list<-fread("artist_list.csv",header=T)

ts_list<-list()
for (i in 1:50){
  ts_list[[i]]<-ts(ts_all_artist[artid==artist_list[i]]$op_num,frequency=7)
}

ts_matrix<-ts_list[[1]]
for (i in 2:50){
  ts_matrix<-cbind(ts_matrix,ts_list[[i]])
}
ts_matrix<-ts_matrix+1
colnames(ts_matrix)<-c(1:50)

plot_fit_result<-function(idx){
  plot(ts_matrix[,idx],xlim=c(1,40))
  lines(fore$mean[[idx]],col="red")
  lines(fore$fitted[,idx],col="green")
}

evaluate<-function(fore){
  ## return a vector F_score(total,f1,...,f50)
  ## large F_score is good
  test_set<-ts_matrix[123:183,]
  sigma<-rep(0,50)
  phi<-rep(0,50)
  F_score<-rep(0,51)
  for (i in 1:50){
    sigma[i]<-sqrt(sum(((test_set[,i]-c(floor(exp(fore[[i]]))))/test_set[,i])^2)/61)
    phi[i]<-sqrt(sum(test_set[,i]))
    F_score[i]<-(1-sigma[i])*phi[i]
    F_score[51]<-F_score[51]+(1-sigma[i])*phi[i]
  }
  print (sum(phi))
  return (F_score)
}


#------------------VAR(p) Model use 3-6 month predict 7-8 month-------------
fit<-VAR(log(window(ts_matrix,start=c(1,1),end=c(18,3))),p=1,type="const")
fore<-forecast(fit,h=61)
evaluate(fore$mean)
