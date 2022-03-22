library(gains)
library(dplyr)
library(ggplot2)
library(rpart)
library(rpart.plot)
library(forecast)
library(caret)
library(e1071)
library(neuralnet)
library(nnet)
library(arules)
library(arulesViz)


str(ls(pat="^V"))
project <- read.csv("predictive_maintenance.csv")
View(project)

dim(project)
names(project)

str(project)

project$Target <- as.factor(project$Target)
levels(project$Target)
failure_type <- as.factor(project$Failure.Type)
levels(failure_type)

#We can understand that there are 6 major types of failures in the sample data
#Heat Dissipation failure
#overstrain failure
#power failure
#Random Failure
#Tool Wear Failure
#No failure

#921

category_type <- as.factor(project$Type)
levels(category_type)

#There are 3 main types of products in the sample data, H, L and M.

summary(project)


pm_pt.df <- project[,-c(1,2,10)]
View(pm_pt.df)


library(fastDummies)
pm_pt.df <- dummy_cols(pm_pt.df, select_columns = 'Type',
                       remove_selected_columns = TRUE)

View(pm_pt.df)

set.seed(3)
train.index <- sample(c(1:dim(pm_pt.df)[1]), dim(pm_pt.df)[1]*0.6)  
train.df <- pm_pt.df[train.index, ]
valid.df <- pm_pt.df[-train.index, ]

#Running a logistic regression

log_pred <- glm(Target ~ ., data = pm_pt.df, family = "binomial")
options(scipen = 999)
summary(log_pred)
View(valid.df)
logit.reg.pred <- predict(log_pred, valid.df[, -6], type = "response")

# first 5 actual and predicted records
data.frame(actual = pm_pt.df$Target[1:5], predicted = logit.reg.pred[1:5])

#code for analyzing the performance of the logistic regression*****************

library(gains)

length(logit.reg.pred)
gain <- gains(valid.df$Target, logit.reg.pred, groups=10)

# plot lift chart
plot(c(0,gain$cume.pct.of.total*sum(valid.df$Target))~c(0,gain$cume.obs), 
     xlab="# cases", ylab="Cumulative", main="", type="l")
lines(c(0,sum(valid.df$Target))~c(0, dim(valid.df)[1]), lty=2)

# compute deciles and plot decile-wise chart
heights <- gain$mean.resp/mean(valid.df$Target)
midpoints <- barplot(heights, names.arg = gain$depth, ylim = c(0,9), 
                     xlab = "Percentile", ylab = "Mean Response", main = "Decile-wise lift chart")

# add labels to columns
text(midpoints, heights+0.5, labels=round(heights, 1), cex = 0.8)

#remove M
#number of 0s and 1s in the target column both for training and validation, oversampling

#Classification Tree

class.tree <- rpart(Target ~ ., data = train.df, 
                    control = rpart.control(minsplit = 1), method = "class")


prp(class.tree, type = 1, extra = 1, split.font = 1, varlen = -10)


deeper.ct <- rpart(Target ~ ., data = train.df, method = "class", cp = 0, minsplit = 1)

prp(deeper.ct, type = 1, extra = 1, split.font = 1, varlen = -10)


cv.ct <- rpart(Target ~ ., data = train.df, method = "class", 
               cp = 0, minsplit = 1)
length(cv.ct)
# use printcp() to print the table. 
printcp(cv.ct)
rpart.plot(cv.ct)

#  Table 9.4 completed -----------------------

#
#  -------------  Figure 9.12: Code for pruning the tree

# prune by lower cp

pruned.ct <- prune(cv.ct, 
                   cp = cv.ct$cptable[which.min(cv.ct$cptable[,"xerror"]),"CP"])

length(pruned.ct$frame$var[pruned.ct$frame$var == "<leaf>"])

rpart.plot(pruned.ct)
prp(pruned.ct, type = 1, extra = 1, split.font = 1, varlen = -10)  

pruned.ct.pred.valid <- predict(pruned.ct,valid.df,type = "class")

confusionMatrix(pruned.ct.pred.valid, as.factor(valid.df$Target), positive = "1")

train.df$Air.temperature..K.
# Neural Network

nn_all <- neuralnet(Target ~ ., data = train.df, hidden = 1, stepmax = 1e9, rep = 2) 



plot(nn_all, rep = "best")

prediction(nn_all)

predict_failure <- compute(nn_all, valid.df)

predict_failure$net.result


#confusion matrix.
#do same thing in python
