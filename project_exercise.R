## Practical Machine Learning
## Author: Tomas Mawyin
## Project - Prediction of how exercise is performed based on data.

## The goal of your project is to predict the manner in which they did the exercise. 
## This is the "classe" variable in the training set. You may use any of the 
## other variables to predict with. You should create a report describing how 
## you built your model, how you used cross validation, what you think the 
## expected out of sample error is, and why you made the choices you did. 
## You will also use your prediction model to predict 20 different test cases. 

# Loading libraries
library(ggplot2)
library(caret)
library(rattle)

# URL for the training and testing sets
train.url <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
test.url <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"

# DOWNLOADING & READING FILES
# Downloading the files
download.file(train.url, destfile="pml-training.csv", method="libcurl")
download.file(test.url, destfile="pml-testing.csv", method="libcurl")

# Now we can read the files
# This will take care of missing and unimportant information when reading files
train.data <- read.csv("pml-training.csv", stringsAsFactors=FALSE, na.strings=c("","NA","#DIV/0"))
test.data <- read.csv("pml-testing.csv", stringsAsFactors=FALSE, na.strings=c("","NA","#DIV/0"))

# CLEANING DATA SETS
# We know a lot of columns that contain NAs. Let's separate the NA columns
na.cols <- sapply(train.data, function(x) any(is.na(x)))
colnames(train.data)[!complete.cases(t(train.data))]

# We know that the test and the training set have the same variables (except 
# for the last one).
train.data <- train.data[,!na.cols]
test.data <- test.data[,!na.cols]

# We can also remove the first 7 columns, since they are not predictors
train.data <- train.data[,-seq(1,7)]
test.data <- test.data[,-seq(1,7)]
testing <- test.data[,-length(test.data)]

# PRE-PROCESS DATA
# The first thing to do is to investigate zero variance predictors 
# We look at the 3rd column to see any variables that have zero variance
sum(nearZeroVar(train.data,saveMetrics = TRUE)[,3])

# EXPLORATORY DATA ANALYSIS - only on training set
dim(train.data)
str(train.data)

# Let's explore some of the predictors. It turns out it is hard to classify
# just by looking at two predictors at a time.
ggplot(data=train.data, aes(total_accel_forearm, total_accel_dumbbell)) +
    geom_point(aes(color=classe)) +
    labs(title = "Exploration of predictors") +
    labs(x = "Total Accel. Forearm", y = "Total Accel. Dumbbell")

ggplot(data=train.data, aes(roll_arm, yaw_dumbbell)) +
    geom_point(aes(color=classe)) +
    labs(title = "Exploration of predictors") +
    labs(x = "Roll Forearm", y = "Yaw Dumbbell")

# Based on the above plots, we know that a multivariate regression algorithm 
# will not produce optimal results, the data does not present a linear trend. 
# Since we have labeled data, we can construct algorithms that make use of all 
# the predictors we have such as decision trees and random forest

# MACHINE LEARNING PREDICTIONS
# Let's start by splitting the training set to get a validation set. We will do 
# a 60/40 split
set.seed(159753)
data.split <- createDataPartition(train.data$classe, p=0.60, list=FALSE)
training <- train.data[data.split,]
cross.val <- train.data[-data.split,]

# -- DESICION TREES
# Let's see if there is an unbalance in the classes
table(training$classe)

# Let's train the model using a decision tree. We will use tuneLength=30 to
# investigate more models
model.tree <- train(classe ~ ., data = training, method = "rpart", 
                    preProcess=c("center", "scale"), tuneLength = 30)
model.tree$finalModel
fancyRpartPlot(model.tree$finalModel)

# How do we do on the cross-validation set
predict.cv <- predict(model.tree, newdata = cross.val)
confusionMatrix(predict.cv, cross.val$classe)


# The accuracy of the decision tree is approximately 50%. There is a large 
# error in the cross-validation set. This will imply a larger error in the
# testing set.

# -- RANDOM FOREST
model.forest <- train(classe ~ ., data = training, method = 'rf',
                      trControl=trainControl(method="cv",number=5),
                      preProcess=c("center", "scale"))

predict.forest <- predict(model.forest, newdata = cross.val)
confusionMatrix(predict.forest, cross.val$classe)

# Since this algorithm is 98% accurate, we will use it on the testing set.
final.prediction <- predict(model.forest, newdata = testing)

# SAVING PREDICTIONS
pml_write_files = function(x){
    n = length(x)
    for(i in 1:n){
        filename = paste0("problem_id_",i,".txt")
        write.table(x[i],file=filename,quote=FALSE,row.names=FALSE,col.names=FALSE)
    }
}
pml_write_files(final.prediction)

# OPTIONAL
# BOOSTING
model.boosting <- train(classe~., method="gbm",data=training, verbose=FALSE)
predict.boost <- predict(model.boosting, newdata = cross.val)
confusionMatrix(predict.boost, cross.val$classe)

# Using PCA

