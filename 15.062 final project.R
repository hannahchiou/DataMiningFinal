# 15.062 Final Project

##### data cleaning, merging #####
stocks <- read.csv("Desktop/Stocks.csv")
head(stocks)
dim(stocks)
# trim to 2022-08-11 ✅
which(stocks$Date=="2022-08-11")
stocks.2 <- stocks[1:10744,]
# split up based on diff companies

factors <- read.csv("Desktop/Factors.csv")
head(factors)
dim(factors)
which(factors$Date=="2022-08-11")
factors.2 <- factors[5:10748,]

final <- cbind(stocks.2,factors.2)
head(final)

# Boeing (BA)

small.final <- final[,c("Date","BA","BAC","CVS","F","KO","Mkt.RF","SMB",
                        "HML","RMW","CMA","RF","MOM")]

##### exploratory #####
install.packages("ggplot2")
library("ggplot2")
install.packages("ggcorrplot")
library("ggcorrplot")

BA.cor <- cor(y=small.final$BA,x=small.final[,c("Mkt.RF","SMB",
                                      "HML","RMW","CMA","MOM")])
BA.cor
round(BA.cor,5)[1:6]

ggcorrplot(BA.cor)
par(mfrow=c(2,3))
plot(x=small.final$Mkt.RF,y=small.final$BA,xlab="Equity risk premium",
     col = rgb(red = 1, green = 0, blue = 0, alpha = 0.2),
     pch = 16)
plot(x=small.final$SMB,y=small.final$BA,xlab="SMB (size risk)",
     col = rgb(red = 1, green = 0, blue = 0, alpha = 0.2),
     pch = 16)
plot(x=small.final$HML,y=small.final$BA,xlab="HML (value risk)",
     col = rgb(red = 1, green = 0, blue = 0, alpha = 0.2),
     pch = 16)
plot(x=small.final$RMW,y=small.final$BA,xlab="RMW (profitability)",
     col = rgb(red = 1, green = 0, blue = 0, alpha = 0.2),
     pch = 16)
plot(x=small.final$CMA,y=small.final$BA,xlab="CMA (investment)",
     col = rgb(red = 1, green = 0, blue = 0, alpha = 0.2),
     pch = 16)
plot(x=small.final$MOM,y=small.final$BA,xlab="MOM (momentum)",
     col = rgb(red = 1, green = 0, blue = 0, alpha = 0.2),
     pch = 16)


##### creating linear regression models #####
# maybe just find summaries for training datasets

CAPM.BA <- lm(BA~(Mkt.RF),data=final)
summary(CAPM.BA)

## 
ff_three_factor.BA <- lm(BA~(Mkt.RF)+SMB+HML,data=final)
summary(ff_three_factor.BA )
# intercept is excess return outside of three factors
# negative intercept means generates loss (underperforms market) BUT not significant
# intercept very small bc daily return (also note that trading week = 5 days)
# (250*-4.598e-05 ) - underperforms market by 1.15% 
# about 255 days per year -> stock trading week is only 5 days

## 
ff_five_factor.BA <- lm(BA~Mkt.RF+SMB+HML+RMW+CMA,data=final)
summary(ff_five_factor.BA)
# underperforms but not statistically significant

## 
carhart.BA <- lm(BA~(Mkt.RF)+SMB+HML+MOM,data=final)
summary(carhart.BA)


#### train & test ####
BA <- small.final[,c("BA","Mkt.RF","SMB",
                     "HML","RMW","CMA","RF","MOM")]

set.seed(1)
n <- nrow(small.final)
n.train <- round(n*0.6) # Partition data for train and test/validation
id.train <- sample(n, n.train) # Create a random index
# Training and validation sets
train.BA <- BA[id.train,] 
valid.BA <- BA[-id.train,]

CAPM.BA <- lm(BA~(Mkt.RF),data=train.BA)
summary(CAPM.BA)
p.BAA <- predict(CAPM.BA,newdata=train.BA)
sqrt(mean((p.BAA - train.BA$BA)^2))
p.BA <- predict(CAPM.BA,newdata=valid.BA)
sqrt(mean((p.BA - valid.BA$BA)^2))
#r<-resid(CAPM.BA)
#plot(fitted(CAPM.BA),r)
AIC(CAPM.BA)
BIC(CAPM.BA)

ff_three_factor.BA <- lm(BA~(Mkt.RF)+SMB+HML,data=train.BA)
summary(ff_three_factor.BA)
p.BA <- predict(ff_three_factor.BA,newdata=train.BA)
sqrt(mean((p.BA - train.BA$BA)^2))
AIC(ff_three_factor.BA)
BIC(ff_three_factor.BA )

ff_five_factor.BA <- lm(BA~Mkt.RF+SMB+HML+RMW+CMA,data=train.BA)
p.BA <- predict(ff_five_factor.BA,newdata=train.BA)
summary(ff_five_factor.BA)
sqrt(mean((p.BA - train.BA$BA)^2))
AIC(ff_five_factor.BA)
BIC(ff_five_factor.BA )

carhart.BA <- lm(BA~(Mkt.RF)+SMB+HML+MOM,data=train.BA)
p.BA <- predict(carhart.BA,newdata=train.BA)
summary(carhart.BA)
sqrt(mean((p.BA - train.BA$BA)^2))
AIC(carhart.BA)
BIC(carhart.BA )

#### feature selection ####

###### lasso regression ######
library(glmnet)
library(usdm)

head(BA)
y <- BA$BA
x <- BA[,c(2:6,8)]

vifstep(x,th=5)
# no collinearity problems

cv_model <- cv.glmnet(as.matrix(x), y, family="gaussian", alpha = 1)
best_lambda <- cv_model$lambda.min
best_lambda
par(mfrow=c(1,1))
plot(cv_model)
best_model <- glmnet(x, y, alpha = 1, lambda = best_lambda)
coef(best_model,s=cv_model$lambda.min)
# best model includes all predictors

###### random forest feature importance (bagging) ######
library(randomForest)
install.packages("varImp")
library(varImp)
set.seed(1)

rf_model <- randomForest(BA~Mkt.RF+SMB+HML+RMW+CMA+MOM,data=BA,importance=TRUE)
rf_model
# only 32% of variance explained
imp_scores <- importance(rf_model,importance=TRUE)
imp_scores
imp_scores[,2]

#### model averaging ####
###### model stacking ######
library("caret")
stacked <- stack(list(CAPM.BA,ff_three_factor.BA,ff_five_factor.BA,carhart.BA))

#### neural networks ####
library("neuralnet")
set.seed(1)
nn <- neuralnet(BA~Mkt.RF+SMB+HML+RMW+CMA+MOM,data=train.BA,linear.output=F,hidden=1)
# justification for choosing 1 layer: less complex data, not as many layers

#### k fold cross validation (with k=10) ####
# creating folds
n <- nrow(BA)
K <- 10 # since we are performing 10-fold CV
n.fold <- round(n/K) # size of each fold = 1074

set.seed(1)
shuffle <- sample(1:n, n, replace=FALSE)
index.fold <- list() # create list of random subsets
for (i in 1:K) {
  if (i < K) {
    index.fold[[i]] <- shuffle[((i-1) * n.fold + 1) : (i * n.fold)]
  } else {
    index.fold[[i]] <- shuffle[((K-1) * n.fold + 1) : n]
  }
}

# calculating CV scores
CV.score.CAPM <- 0
for(i in 1:K) {
  # fit on data except ith fold
  CAPM.BA <- lm(BA~(Mkt.RF),data=BA[-index.fold[[i]],])
  # predict for ith fold
  Yhat.CAPM <- predict(CAPM.BA,newdata=BA[index.fold[[i]],])
  # error result
  CV.score.CAPM <- CV.score.CAPM + (1/n) * sum((BA$BA[index.fold[[i]]] - Yhat.CAPM)^2)
}
round(CV.score.CAPM,6)

# Fama French 3 factor
CV.score.ff3 <- 0
for(i in 1:K) {
  # fit on data except ith fold
  ff_three_factor.BA <- lm(BA~(Mkt.RF)+SMB+HML,data=BA[-index.fold[[i]],])
  # predict for ith fold
  Yhat.ff3 <- predict(ff_three_factor.BA,newdata=BA[index.fold[[i]],])
  # error result
  CV.score.ff3 <- CV.score.ff3 + (1/n) * sum((BA$BA[index.fold[[i]]] - Yhat.ff3)^2)
}
round(CV.score.ff3,6)

# Fama French 5 factor
CV.score.ff5 <- 0
for(i in 1:K) {
  # fit on data except ith fold
  ff_five_factor.BA <- lm(BA~Mkt.RF+SMB+HML+RMW+CMA,data=BA[-index.fold[[i]],])
  # predict for ith fold
  Yhat.ff5 <- predict(ff_five_factor.BA,newdata=BA[index.fold[[i]],])
  # error result
  CV.score.ff5 <- CV.score.ff5 + (1/n) * sum((BA$BA[index.fold[[i]]] - Yhat.ff5)^2)
}
round(CV.score.ff5,6)
# 0.000284

# Carhart 4 factor
CV.score.carhart <- 0 
for(i in 1:K) {
  # fit on data except ith fold
  carhart.BA <- lm(BA~(Mkt.RF)+SMB+HML+MOM,data=BA[-index.fold[[i]],])
  # predict for ith fold
  Yhat.carhart <- predict(carhart.BA,newdata=BA[index.fold[[i]],])
  # error result
  CV.score.carhart <- CV.score.carhart + (1/n) * sum((BA$BA[index.fold[[i]]] - Yhat.carhart)^2)
}
round(CV.score.carhart,6)
# 0.000287

# Lasso regression (all predictors)
CV.score.lasso <- 0 
for(i in 1:K) {
  # fit on data except ith fold
  lasso.BA <- lm(BA~(Mkt.RF)+SMB+HML+MOM+CMA+RMW,data=BA[-index.fold[[i]],])
  # predict for ith fold
  Yhat.lasso <- predict(lasso.BA,newdata=BA[index.fold[[i]],])
  # error result
  CV.score.lasso <- CV.score.carhart + (1/n) * sum((BA$BA[index.fold[[i]]] - Yhat.lasso)^2)
}
round(CV.score.lasso,6)
# 0.000318

# Random forest chosen features
CV.score.rf <- 0 
for(i in 1:K) {
  # fit on data except ith fold
  rf.BA <- lm(BA~(Mkt.RF)+HML+MOM,data=BA[-index.fold[[i]],])
  # predict for ith fold
  Yhat.rf<- predict(rf.BA,newdata=BA[index.fold[[i]],])
  # error result
  CV.score.rf <- CV.score.rf + (1/n) * sum((BA$BA[index.fold[[i]]] - Yhat.rf)^2)
}
round(CV.score.rf,6)
# 0.000287

# Neural network 
CV.score.nn <- 0 
for(i in 1:K) {
  # fit on data except ith fold
  nn.BA <- neuralnet(BA~Mkt.RF+SMB+HML+RMW+CMA+MOM,
                     data=BA[-index.fold[[i]],],linear.output=F,hidden=1)
  # predict for ith fold
  Yhat.nn<- predict(nn.BA,newdata=BA[index.fold[[i]],])
  # error result
  CV.score.nn <- CV.score.nn + (1/n) * sum((BA$BA[index.fold[[i]]] - Yhat.nn)^2)
}
round(CV.score.nn,6)
# 0.000433