#### SATYANDER TRANSACTION PREDICTION ###

# Removing Variable
rm(list=ls())

# Setting working directory
setwd("D:/Eduvisor/Project/Project 2/R Code")
getwd()

# Loading Libraries

x = c("ggplot2","corrgram","DMwR","usdm","caret","randomForest","e1071",
      "DataCombine","doSNOW","inTrees","rpart.plot","rpart",'MASS','stats','utils','glmnet','pROC')

# Load Packages
lapply(x, require , character.only = TRUE )
rm(x)

# You are provided with an anonymized dataset containing numeric feature variables, the
# binary target column, and a string ID_code column. The task is to predict the value
# of target column in the test set.

# Loading the dataset
test_data = read.csv("test.csv", header = T, na.strings = c(" ", "", "NA"))
train_data = read.csv("train.csv", header = T, na.strings = c(" ", "", "NA"))

# Structure of data
str(train_data) 
str(test_data)  

# Summary of the data
summary(train_data)
summary(test_data)

# Fetching Records
head(train_data,10)
head(test_data,10)
##################################         Missing Values Analysis of Train Data      ###############################################

missing_val = data.frame(apply(train_data,2,function(x){sum(is.na(x))}))
missing_val$Columns = row.names(missing_val)
names(missing_val)[1] =  "Missing_percentage"
missing_val$Missing_percentage = (missing_val$Missing_percentage/nrow(train_data)) * 100
missing_val = missing_val[order(-missing_val$Missing_percentage),]
row.names(missing_val) = NULL
missing_val = missing_val[,c(2,1)]
write.csv(missing_val, "Miising_perc_train.csv", row.names = F)

##################################         Missing Values Analysis of Test Data      ###############################################

missing_val_test = data.frame(apply(test_data,2,function(x){sum(is.na(x))}))
missing_val_test$Columns = row.names(missing_val_test)
names(missing_val_test)[1] =  "Missing_percentage"
missing_val_test$Missing_percentage = (missing_val_test$Missing_percentage/nrow(test_data)) * 100
missing_val_test = missing_val_test[order(-missing_val_test$Missing_percentage),]
row.names(missing_val_test) = NULL
missing_val_test = missing_val_test[,c(2,1)]
write.csv(missing_val_test, "Miising_perc_test.csv", row.names = F)


# There are no missing values in both train and test data set . So that's a great thing.

# Correlations in train data
train_data$target<-as.numeric(train_data$target)
train_correlations<-cor(train_data[,c(2:202)])
train_correlations

# We can observed that the correlation between the train attributes is very small.

#Correlations in test data
test_correlations<-cor(test_data[,c(2:201)])
test_correlations

# We can observed that the correlation between the test attributes is very small.

# Split the training data using simple random sampling
train_index<-sample(1:nrow(train_data),0.75*nrow(train_data))
# Train data
training_data<-train_data[train_index,]
# Validation data
validation_data<-train_data[-train_index,]

# Dimension of train and validation data
dim(training_data)
dim(validation_data)

############################################# Logistic Regression model ###############################################

#Training dataset
X_train<-as.matrix(training_data[,-c(1,2)])
y_train<-as.matrix(training_data$target)
#validation dataset
X_validation<-as.matrix(validation_data[,-c(1,2)])
y_validation<-as.matrix(validation_data$target)
#test dataset
test<-as.matrix(test_data[,-c(1)])

#Logistic regression model
set.seed(667) # to reproduce results
lr_model <-glmnet(X_train,y_train, family = "binomial")
summary(lr_model)

#Cross validation prediction
set.seed(8909)
cross_validation_logistic_regression_model <- cv.glmnet(X_train,y_train,family = "binomial", type.measure = "class")
cross_validation_logistic_regression_model

#Plotting the missclassification error vs log(lambda) where lambda is regularization parameter

#Minimum lambda
cross_validation_logistic_regression_model$lambda.min

#plot the auc score vs log(lambda)
plot(cross_validation_logistic_regression_model)

#We can observed that miss classification error increases as increasing the log(Lambda).

###########################   Model performance on validation dataset   ############################################
set.seed(5365)
cross_validation_predict.logistic_regression<-predict(cross_validation_logistic_regression_model,X_validation,s = "lambda.min", type = "class")
cross_validation_predict.logistic_regression

#Accuracy of the model is not the best metric to use when evaluating the imbalanced datasets as it may be misleading.
#So, we are going to change the performance metric.

########################################  Confusion Matrix  ################################################

set.seed(695)
#actual target variable
target<-validation_data$target
#convert to factor
target<-as.factor(target)
#predicted target variable
#convert to factor
cross_validation_predict.logistic_regression<-as.factor(cross_validation_predict.logistic_regression)
confusionMatrix(data=cross_validation_predict.logistic_regression,reference=target)

# False Negative Rate
# FNR = False Negative / True Positive + False Negative
614 / (614 + 1287) * 100
# So our False Negative Rate is 32.29%

# Recall : The proportion of actually positive test cases that are correctly identified is 92.24%
44367/(44367 + 3732) *100

# Specificity :  The proportion of actually negative test cases that are correctly identified is 67.38%
1281 / (1287+614) * 100

#Accuracy

#############################    Random Forest      ###########################################################
set.seed(2732) # to reproduce 'results'
#convert to int to factor
train_data$target<-as.factor(train_data$target)
#fitting the random forest
Random_Forest<-randomForest(target~.,train_data[,-c(1)],ntree=50,importance=TRUE)
summary(Random_Forest)

# Feature importance by random forest
Variable_Importance<-varImp(Random_Forest,type=2, scale = FASLE)
Variable_Importance

# We can observed that the top important features are var_12, var_22, var_81,v var_109, var_139 and so on based on Mean decrease gini.
cnames = colnames(Variable_Importance)
cnames

# Plotting the graph
Variable_Importance_Plot = varImpPlot(Random_Forest,sort = TRUE , n.var = 20 , scale = TRUE)

#Predict train data using random forest model
Random_Forest_Predictions = predict(Random_Forest, train_data[,-c(1)])

########################    Confusion Matrix  For Random Forest  #################################################
ConfusionMatrix_Random_Forest = table(train_data$target , Random_Forest_Predictions)
confusionMatrix(ConfusionMatrix_Random_Forest)

# False Negative Rate
# FNR = False Negative / True Positive + False Negative
73 / (73+20025) * 100
# So our False Negative Rate is 36.32%

# Recall : The proportion of actually positive test cases that are correctly identified is 99.63%
20025/(20025 + 73) *100

# Specificity :  The proportion of actually negative test cases that are correctly identified is 100%
179902/ (179902+0) * 100


#Use Naive Bayes
#Develop model
NB_model = naiveBayes(target~., data = train_data)

#predict on test cases #raw
NB_Predictions = predict(NB_model, test_data[,2:201], type = 'class')

########################    Confusion Matrix  For Naive Bayes  #################################################
ConfusionMatrix_naiveBayes = table(train_data$target , NB_Predictions)
confusionMatrix(ConfusionMatrix_naiveBayes)

# Accuracy : 86.86%