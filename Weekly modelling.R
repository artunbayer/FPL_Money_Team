##
rm(list=ls())

library(rSTAR)
library(randomForest)
library(dummies)
require(dplyr) 
require(ggplot2)
library(reshape2)
library(tidyverse)
library(refund)
library(synergyfinder)
library(glmnet)
library(coda)

d1 <- read.csv("1final.csv", row.names=1)
d2 <- read.csv("2final.csv", row.names=1)
d3 <- read.csv("3final.csv", row.names=1)
d4 <- read.csv("4final.csv", row.names=1)
d5 <- read.csv("5final.csv", row.names=1)


stacked_total <- read.csv("stacked_total.csv", row.names=1, sep=";")

stacked_total$name = paste(stacked_total$first_name," ",stacked_total$second_name)


D = full_join(d1,d2)
D = full_join(D,d3)
D = full_join(D,d4)
D = full_join(D,d5)

#Position df
df_position = D %>% select(c(first_name,second_name,position)) %>% distinct() %>%
  mutate(position = as.factor(position))  

stacked_total = inner_join(stacked_total,df_position%>%
                             dummy.data.frame(names = "position"))#change variable position into 4 dummy variables

#Helper function
AE = function(y,y_hat){
  return(abs(y_hat - y))
}
Team_Prediction <- function(y = 4, p = 1/2){
  {
  #Players that have stayed from y-2 to y-1 season
  who.stayed_last_year = inner_join(stacked_total %>% filter(year == y-1)  %>% 
                                      filter(week == round(p*max(week)))%>% select(name),
                                    stacked_total %>% filter(year == y - 2) %>% 
                                      filter(week == round(p*max(week))) %>% select(name)) %>% unique()
  who.stayed_last_year = who.stayed_last_year[!pmax(duplicated(who.stayed_last_year),duplicated(who.stayed_last_year,fromLast = TRUE)),]
  
  #Players that have stayed from y-1 to y season
  who.stayed_this_year = inner_join(stacked_total %>% filter(year == y) %>% 
                                      filter(week == round(p*max(week))) %>% select(name),
                          stacked_total %>% filter(year == y - 1) %>% 
                            filter(week == round(p*max(week)))%>%  select(name)) %>% unique()
  who.stayed_this_year = who.stayed_this_year[!pmax(duplicated(who.stayed_this_year),duplicated(who.stayed_this_year,fromLast = TRUE)),]
  
  
  #total points of  players in season y -1 (that stayed until season y-1)
    df_2 = stacked_total %>% filter(year == y - 2) %>%
    filter(name %in% who.stayed_last_year) %>% group_by(name) %>%
    summarize(max_pt = max(total_points))
  
  tp_2 = df_2$max_pt
  
  #total points of season y -1 (for players stayed from season y-2)
  df_1.2 = stacked_total %>% filter(year == y - 1) %>%
    filter(name %in% who.stayed_last_year) %>% group_by(name) %>%
    summarize(max_pt = max(total_points))
  
  #total points last year (for players stayed until season y-1)
  tp_1.2 = df_1.2$max_pt
  df_1.0 = stacked_total %>% filter(year == y - 1) %>%
    filter(name %in% who.stayed_this_year) %>% group_by(name) %>%
    summarize(max_pt = max(total_points))    
  
  #total points for players this year (that stayed from season y-1)
  df_0 = stacked_total %>% filter(year == y ) %>%
    filter(name %in% who.stayed_this_year) %>% group_by(name) %>%
    summarize(max_pt0 = max(total_points))%>%
    inner_join(df_1.0)
  
  tp_0 = df_0$max_pt0
  tp_1.0 = df_0$max_pt
  

  y.train1 = tp_1.2 #Season y-1 top scores for players from season y-2
  y.test1 = tp_0 #Season y top scores for players from season y-1
  
  points_previous_years = dcast(stacked_total %>% 
                                   filter(year < y) %>% 
                                   select(name, week,total_points)  %>% 
                                   select(c(name, week,total_points)), name ~week,mean) %>% 
    column_to_rownames(var = "name") %>% as.matrix() %>%
    ImputeNA()
  
  this_years = dcast(stacked_total %>% filter(year == y,name %in% who.stayed_this_year) %>% 
                       select(name, week,total_points)  %>% 
                       select(c(name, week,total_points)), name ~week,mean) %>% 
    column_to_rownames(var = "name") %>% as.matrix()  %>% ImputeNA()
  
  this_years.partial = this_years[,seq(p*ncol(this_years))]
  }
   fit.fpca = fpca.sc(points_previous_years)

     preds <-matrix(ncol = ncol(this_years), nrow = nrow(this_years))
   rownames(preds) <- rownames(this_years)
   for(i in seq(nrow(this_years.partial))){
     ef = fit.fpca$efunctions[seq(p*ncol(this_years)),]
     y_i = this_years.partial[i,] + rnorm(ncol(this_years.partial),0,0.00001)
     
     p.fac = 1/fit.fpca$evalues
     fit.test = cv.glmnet(x =ef, y = y_i %>% t(),alpha = 0,penalty.factor =  p.fac)
     preds[i,] = predict(fit.test, newx = fit.fpca$efunctions)[seq(ncol(this_years))]
   }
    score_hat = round(preds[, ncol(this_years)] )
  
   res = cbind(rownames(preds),score_hat) %>% 
     as.data.frame(); names(res) =  c("name","score_hat")

   
    res = res %>% mutate(score_hat = as.numeric(score_hat)) %>%
      left_join(stacked_total %>% 
                   filter(year == y) %>%
                   select(name,first_name,second_name) %>%
                   inner_join(df_position)%>% distinct(name, .keep_all = TRUE) ) %>%
      select(name,score_hat, position)
    plot( y.test1,res$score_hat,ylab = "score hat", xlab = "score", main = paste("prediction using Regularized FPCA ","for year ",y))
    lines(y.test1,y.test1)
    
    print("MAE STAT (for nonzero values):")
    print(summary(AE( y.test1,res$score_hat)))

       return(res)
   
   
}
#test case 1: year = 4, p = 1/2
res1 = Team_Prediction()

#test case 2:year = 3, p = 1/2
res2 = Team_Prediction(y = 3, p = 1/2)

#test case 3:year = 2, p = 1/2
res3 = Team_Prediction(y = 2, p = 1/2)

#test case 4:year = 3, p = 1/3
res4 = Team_Prediction(y = 3, p = 1/3)
