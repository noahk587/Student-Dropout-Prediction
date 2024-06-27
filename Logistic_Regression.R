#' ---
#'  title:   "Student Dropout Prediction: Logistic Regression"
#'  author:  "Noah Khan"
#'  date:    "June 5, 2024"
#'  output:
#'    github_document:
#'      html_preview: true
#'      toc: yes
#' ---


#' ## Load Libraries, Functions, and Dataset
library(caTools)
library(fastDummies)
library(glmnet)
library(ggplot2)
library(plotROC)

# a seed will be used to reproduce all results
set.seed(12345)

# load the altered dataset
studentsData <- read.csv("data_altered.csv")

# load function to compute metric
source("testingMetrics.R")

# load function to plot roc curve
source("ROCplot.R")

#' ## Creating Indicator Variables
#'  Some of the predictors do not consist of binary values. Therefore, indicator
#'  variables are needed before fitting a logistic regression model.
#'  



# create the indicator variable for predictors that are not binary
studentsData <- dummy_cols(studentsData, select_columns = c('Marital_Status',
                      'Application_Type', 'Application_Order', 'Academic_Path',
                      'Previous_Qualification', 'Nationality', 'Gender'), 
                     remove_selected_columns = TRUE, remove_first_dummy = TRUE)

#' ## Training and Testing Data Split
#' The dataset will be split into a training and testing dataset to evaluate each
#' of the regression models. A 75:25 ratio will be used for the split.


# split the dataset
dataSplit <- sample.split(studentsData$Dropped_Out, SplitRatio = 0.75)

# define the training and testing dataset
trainingData <- subset(studentsData, dataSplit == TRUE)
testingData <- subset(studentsData, dataSplit == FALSE)

# a vector that contains the names of the models
modelNames <- c("Full Model", "Reduced Model", "Forward Selection", 
                "Backward Selection", "Ridge", "Lasso")

#' ## Full Logistic Model  
# consider a full logistic regression model
fullModel <- glm(Dropped_Out ~., family = "binomial", data = trainingData)
print(summary(fullModel))

#' ### Model Evaluation
#' It is not likely that the full logistic model will be the best model to predict
#' a student dropping out. But, we will evaluate this model so we can compare the results
#' with the other proposed models.
#' 
#' #### Training Data

# predicted outcome for dropout
full.prob.train <- predict(fullModel, trainingData, type = "response")
full.pred.train <- ifelse(full.prob.train >= 0.5, 1, 0)

# error rate for full model
print(fullModel.trainingError <- 1- mean(full.pred.train == trainingData[, "Dropped_Out"]))

# confusion matrix
print(matrix.confusion.full.train <- table(full.pred.train, trainingData[, "Dropped_Out"]))


#' #### Testing Data

# predicted outcome for dropout
full.prob.test <- predict(fullModel, testingData, type = "response")
full.pred.test <- ifelse(full.prob.test >= 0.5, 1, 0)

# error rate for full model
print(fullModel.testingError <- 1- mean(full.pred.test == testingData[, "Dropped_Out"]))

# confusion matrix
print(matrix.confusion.full.test <- table(full.pred.test, testingData[, "Dropped_Out"]))


# compute the accuracy, precision, and recall for the testing data
fulltestingMetrics <- testingMetrics(matrix.confusion.full.test)

# print the results
## accuracy
print(fullAccuracy <- as.numeric(fulltestingMetrics[1]))
## precision
print(fullPrecision <- as.numeric(fulltestingMetrics[2]))
## recall
print(fullRecall <- as.numeric(fulltestingMetrics[3]))

# compute the ROC and AUC
rocAUC.full <- ROCplot(testingData$Dropped_Out, full.pred.test, modelNames[1])

# plot the ROC
print(rocAUC.full[1])

#' ## Reduced Model
#' We will consider a reduced logistic regression model. Starting from
#' the full logistic model already defined, we will drop the predictors that are not significant.
#' The process will continue until the model only contains predictors that are significant.

reducedModel <- glm(Dropped_Out ~ Displaced + Tuition_Current + 
                      X1st_Sem_Units_Approved + X2nd_Sem_Units_Enrolled + 
                      X2nd_Sem_Units_Approved +
                        Unemployment_Rate + `Academic_Path_Social Service`, 
                      family = "binomial", data = trainingData)
print(summary(reducedModel))

# drop the predictors that aren't significant
reducedModel <- glm(Dropped_Out ~ Tuition_Current + 
                        X1st_Sem_Units_Approved + X2nd_Sem_Units_Enrolled + 
                        X2nd_Sem_Units_Approved, 
                         family = "binomial", data = trainingData)
print(summary(reducedModel))

# compare the two models
anova(reducedModel, fullModel, test = "Chisq")



#' ### Model Evaluation
#' #### Training Data

# predicted outcome for dropout
reducedModel.prob.train <- predict(reducedModel, trainingData, type = "response")
reducedModel.pred.train <- ifelse(reducedModel.prob.train >= 0.5, 1, 0)

# error rate for reduced model
print(reducedModel.trainingError <- 1 - 
        mean(reducedModel.pred.train == trainingData[, "Dropped_Out"]))

# confusion matrix
print(matrix.confusion.reducedModel.train <- table(reducedModel.pred.train, 
                                                  trainingData[, "Dropped_Out"]))


#' #### Testing Data

# predicted outcome for dropout
reducedModel.prob.test <- predict(reducedModel, testingData, type = "response")
reducedModel.pred.test <- ifelse(reducedModel.prob.test >= 0.5, 1, 0)

# error rate for reduced model
print(reducedModel.testingError <- 1- mean(reducedModel.pred.test == testingData[, "Dropped_Out"]))

# confusion matrix
print(matrix.confusion.reducedModel.test <- table(reducedModel.pred.test, testingData[, "Dropped_Out"]))

# compute the accuracy, precision, and recall for the testing data
reducedtestingMetrics <- testingMetrics(matrix.confusion.reducedModel.test)

# print the results
## accuracy
print(reducedAccuracy <- as.numeric(reducedtestingMetrics[1]))
## precision
print(reducedPrecision <- as.numeric(reducedtestingMetrics[2]))
## recall
print(reducedRecall <- as.numeric(reducedtestingMetrics[3]))

# compute the ROC and AUC
rocAUC.reduced <- ROCplot(testingData$Dropped_Out, reducedModel.pred.test, modelNames[2])

# plot the ROC
print(rocAUC.reduced[1])

#' ## Forward Selection

# model only with the intercept
interceptModel <- glm(Dropped_Out ~ 1, family = "binomial", data = trainingData)

# define logistic regression model using forward stepwise selection
forwardSelection <- step(interceptModel, direction = "forward", 
                         scope = list(upper = fullModel, lower = ~1), trace = 0)

# view the model
print(summary(forwardSelection))

#' ### Model Evaluation
#' #### Training Data

# predicted outcome for dropout
forwardSelection.prob.train <- predict(forwardSelection, trainingData, type = "response")
forwardSelection.pred.train <- ifelse(forwardSelection.prob.train >= 0.5, 1, 0)

# error rate for forward selection model
print(forwardSelection.trainingError <- 1 - 
        mean(forwardSelection.pred.train == trainingData[, "Dropped_Out"]))

# confusion matrix
print(matrix.confusion.forwardSelection.train <- table(forwardSelection.pred.train, 
                                                   trainingData[, "Dropped_Out"]))

#' #### Testing Data

# predicted outcome for dropout
forwardSelection.prob.test <- predict(forwardSelection, testingData, type = "response")
forwardSelection.pred.test <- ifelse(forwardSelection.prob.test >= 0.5, 1, 0)

# error rate for forward selection model
print(forwardSelection.testingError <- 1 - 
        mean(forwardSelection.pred.test == testingData[, "Dropped_Out"]))

# confusion matrix
print(matrix.confusion.forwardSelection.test <- table(forwardSelection.pred.test, 
                                                        testingData[, "Dropped_Out"]))


# compute the accuracy, precision, and recall for the testing data
forwardTestingMetrics <- testingMetrics(matrix.confusion.forwardSelection.test)

# print the results
## accuracy
print(forwardSelectionAccuracy <- as.numeric(forwardTestingMetrics[1]))
## precision
print(forwardSelectionPrecision <- as.numeric(forwardTestingMetrics[2]))
## recall
print(forwardSelectionRecall <- as.numeric(forwardTestingMetrics[3]))

# compute the ROC and AUC
rocAUC.forward <- ROCplot(testingData$Dropped_Out, forwardSelection.pred.test, modelNames[3])

# plot the ROC
print(rocAUC.forward[1])



#' ## Backward Selection
#' Note that it takes a few minutes for the backwards selection model to compile.


# define logistic regression model using backward stepwise selection
backwardSelection <- step(fullModel, direction = "backward", trace = 0)

# view the model
print(summary(backwardSelection))


#' ### Model Evaluation
#' #### Training Data

# predicted outcome for dropout
backwardSelection.prob.train <- predict(backwardSelection, trainingData, type = "response")
backwardSelection.pred.train <- ifelse(backwardSelection.prob.train >= 0.5, 1, 0)

# error rate for backward selection model
print(backwardSelection.trainingError <- 1 - 
        mean(backwardSelection.pred.train == trainingData[, "Dropped_Out"]))

# confusion matrix
print(matrix.confusion.backwardSelection.train <- table(backwardSelection.pred.train, 
                                                        trainingData[, "Dropped_Out"]))

#' #### Testing Data

# predicted outcome for dropout
backwardSelection.prob.test <- predict(backwardSelection, testingData, type = "response")
backwardSelection.pred.test <- ifelse(backwardSelection.prob.test >= 0.5, 1, 0)

# error rate for forward selection model
print(backwardSelection.testingError <- 1 - 
        mean(backwardSelection.pred.test == testingData[, "Dropped_Out"]))

# confusion matrix
print(matrix.confusion.backwardSelection.test <- table(backwardSelection.pred.test, 
                                                       testingData[, "Dropped_Out"]))


# compute the accuracy, precision, and recall for the testing data
BackwardTestingMetrics <- testingMetrics(matrix.confusion.backwardSelection.test)

# print the results
## accuracy
print(backwardSelctionAccuracy <- as.numeric(BackwardTestingMetrics[1]))
## precision
print(backwardSelctionPrecision <- as.numeric(BackwardTestingMetrics[2]))
## recall
print(backwardSelctionRecall <- as.numeric(BackwardTestingMetrics[3]))

# compute the ROC and AUC
rocAUC.backward <- ROCplot(testingData$Dropped_Out, backwardSelection.pred.test, modelNames[4])

# plot the ROC
print(rocAUC.backward[1])


#' ## Ridge Regression

# set up a grid of lambda values (from 10^4 to 10^(-2)) in decreasing sequence
grid <- 10 ^ seq(4, -2, length=100)

# define the design matrix
X <- model.matrix(Dropped_Out~., data = trainingData)[, -1]
y <- trainingData$Dropped_Out

testX <- model.matrix(Dropped_Out~., data = testingData)[, -1]
testY <- testingData$Dropped_Out


# fit ridge regression for each lambda on the grid
ridge.glm <- cv.glmnet(X, y, alpha=0, lambda=grid, thresh=1e-12)

# the best lambda value
print(lambda.best <- ridge.glm$lambda.min)

# define the ridge regression model
ridgeModel <- glmnet(X, y, alpha = 0, family = "binomial")

# get the coefficients
print(predict(ridgeModel, type = "coefficients", s = lambda.best)[, ])


#' ### Model Evaluation
#' #### Training Data

# predicted outcome for dropout
ridge.prob.train <- predict(ridgeModel, type = "response", 
                            s = lambda.best, newx = X)
ridge.pred.train <- ifelse(ridge.prob.train >= 0.5, 1, 0)

# error rate for ridge model
print(ridge.trainingError <- 1 - 
        mean(ridge.pred.train == trainingData[, "Dropped_Out"]))

# confusion matrix
print(matrix.confusion.ridge.train <- table(ridge.pred.train, 
                                                trainingData[, "Dropped_Out"]))


#' #### Testing Data

# predicted outcome for dropout
ridge.prob.test <- predict(ridgeModel, type = "response", 
                           s = lambda.best, newx = testX)
ridge.pred.test <- ifelse(ridge.prob.test >= 0.5, 1, 0)

# error rate for ridge model
print(ridge.testingError <- 1 - 
        mean(ridge.pred.test == testingData[, "Dropped_Out"]))

# confusion matrix
print(matrix.confusion.ridge.test <- table(ridge.pred.test, 
                                                testingData[, "Dropped_Out"]))


# compute the accuracy, precision, and recall for the testing data
ridgeTestingMetrics <- testingMetrics(matrix.confusion.ridge.test)

# print the results
## accuracy
print(ridgeAccuracy <- as.numeric(ridgeTestingMetrics[1]))
## precision
print(ridgePrecision <- as.numeric(ridgeTestingMetrics[2]))
## recall
print(ridgeRecall <- as.numeric(ridgeTestingMetrics[3]))

# compute the ROC and AUC
rocAUC.ridge <- ROCplot(testingData$Dropped_Out, ridge.pred.test, modelNames[5])

# plot the ROC
print(rocAUC.ridge[1])

#' ## Lasso Regression

lasso.glm <- cv.glmnet(X, y, alpha=1, lambda=grid, thresh=1e-12)

print(lambda.best <- lasso.glm$lambda.min)

# define the lasso regression model
lassoModel <- glmnet(X, y, alpha = 1, family = "binomial")

# examine the coefficients
lasso.coef <- predict(lassoModel, type = "coefficients", s = lambda.best)[, ]
print(lasso.coef[lasso.coef != 0])

#' ### Model Evaluation
#' #### Training Data

# predicted outcome for dropout
lasso.prob.train <- predict(lassoModel, type = "response", 
                            s = lambda.best, newx = X)
lasso.pred.train <- ifelse(lasso.prob.train >= 0.5, 1, 0)

# error rate for lasso model
print(lasso.trainingError <- 1 - 
        mean(lasso.pred.train == trainingData[, "Dropped_Out"]))

# confusion matrix
print(matrix.confusion.lasso.train <- table(lasso.pred.train, 
                                            trainingData[, "Dropped_Out"]))



#' #### Testing Data

# predicted outcome for dropout
lasso.prob.test <- predict(lassoModel, type = "response", 
                           s = lambda.best, newx = testX)
lasso.pred.test <- ifelse(lasso.prob.test >= 0.5, 1, 0)

# error rate for lasso model
print(lasso.testingError <- 1 - 
        mean(lasso.pred.test == testingData[, "Dropped_Out"]))

# confusion matrix
print(matrix.confusion.lasso.test <- table(lasso.pred.test, 
                                            testingData[, "Dropped_Out"]))


# compute the accuracy, precision, and recall for the testing data
lassoTestingMetrics <- testingMetrics(matrix.confusion.lasso.test)

# print the results
## accuracy
print(lassoAccuracy <- as.numeric(lassoTestingMetrics[1]))
## precision
print(lassoPrecision <- as.numeric(lassoTestingMetrics[2]))
## recall
print(lassoRecall <- as.numeric(lassoTestingMetrics[3]))


# compute the ROC and AUC
rocAUC.lasso <- ROCplot(testingData$Dropped_Out, lasso.pred.test, modelNames[6])

# plot the ROC
print(rocAUC.lasso[1])


#' ## Model Selection

# create a vector that contains the training errors
trainingErrors <- c(fullModel.trainingError, reducedModel.trainingError, 
                    forwardSelection.trainingError, backwardSelection.trainingError,
                    ridge.trainingError, lasso.trainingError)

# create a vector that contains the testing errors
testingErrors <- c(fullModel.testingError, reducedModel.testingError, 
                    forwardSelection.testingError, backwardSelection.testingError,
                    ridge.testingError, lasso.testingError)

# create a vector that contains the accuracy values
accuracy <- c(fullAccuracy, reducedAccuracy, forwardSelectionAccuracy, 
              backwardSelctionAccuracy, ridgeAccuracy, lassoAccuracy)

# create a vector that contains the precision values
precision <- c(fullPrecision, reducedPrecision, forwardSelectionPrecision, 
              backwardSelctionPrecision, ridgePrecision, lassoPrecision)

# create a vector that contains the recall values
recall <- c(fullRecall, reducedRecall, forwardSelectionRecall, 
            backwardSelctionRecall, ridgeRecall, lassoRecall)

# create a vector that contains the auc values
auc <- c(as.numeric(rocAUC.full[2]), as.numeric(rocAUC.reduced[2]),
         as.numeric(rocAUC.forward[2]), as.numeric(rocAUC.backward[2]), 
         as.numeric(rocAUC.ridge[2]), as.numeric(rocAUC.lasso[2]))


# create a dataframe that will contain all of the metric values
ModelMetrics <- data.frame(modelNames, trainingErrors, testingErrors, accuracy,
                           precision, recall, auc)

names(ModelMetrics) <- c("Model", "Training Error",
                  "Testing Error", "Accuracy", "Precision", "Recall", "AUC")

print(ModelMetrics)

#' From the table above, the full model has the lowest training error while the
#' ridge model has the lowest testing error. The ridge model also had the highest
#' value for accuracy and precision. The backward selection model had the highest
#' value for recall and the forward selection model had the highest AUC value. 
#' Overall, the best regression model would be the ridge model as it had the best
#' values for the majority of the metrics. 