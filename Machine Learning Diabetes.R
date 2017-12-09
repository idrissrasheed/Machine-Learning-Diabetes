#Accessing packages for this project
library(corrplot)
library(e1071)
library(RSNNS)
library(randomForest)
library(mice)
library(kernlab)
library(caret)
library(AppliedPredictiveModeling)
library(VIM)
#Create a vector of the feature names
headers <- c("TotalPregnancies", "PlasmaGlucose", "DiastolicPressure", "FoldThickness",
             "Insulin", "BMI", "PedigreeFunction", "Age", "Diagnosis")

#Import data
url <- paste0("http://archive.ics.uci.edu/ml/machine-learning-databases/",
              "pima-indians-diabetes/pima-indians-diabetes.data")
diabetes <- read.csv(url(url),header = FALSE, col.names = headers)
str(diabetes)
diabetes$Diagnosis <- as.factor(ifelse(diabetes$Diagnosis == 0, "NotDiabetic", "Diabetic"))
pairs(diabetes)
#Creates loops to make all 0's NA
for (i in 2:6){
  for (n in 1:nrow(diabetes)){
    if (diabetes[n, i] == 0){
      diabetes[n, i] <- NA
    }
  }
}
table(is.na(diabetes))
aggr(diabetes[,2:6], cex.lab=1, cex.axis = .4, numbers = T, gap = 0)
#visualises a scatterplot of missing values
scattmatrixMiss(diabetes)
com.diabetes <- mice(diabetes, m = 3, method = 'pmm', seed = 125)
#Density plot original vs imputed dataset
densityplot(com.diabetes)
diabetes <- complete(com.diabetes)
corrplot(cor(diabetes[,-9]),type = "lower", method = "number")
diabetes[, 1:8] <- scale(diabetes[, 1:8], center = TRUE, scale = TRUE)
Folds <- trainControl(method = "repeatedcv",
                      number = 10,
                      repeats = 10,
                      classProbs=TRUE,
                      summaryFunction=twoClassSummary)
prop.table(table(diabetes$Diagnosis))
sampleSize <- floor(.7 * nrow(diabetes))
set.seed(125)
Ind <- sample(seq_len(nrow(diabetes)), size = sampleSize)

XTrain <- diabetes[Ind, 1:8]
XTest <- diabetes[-Ind, 1:8]

YTrain <- diabetes[Ind, 9]
YTest <- diabetes[-Ind, 9]
rf.expand <- expand.grid(mtry = 2:8)
set.seed(125)
diabetes.rf <- train(XTrain,
                     YTrain,
                     method = "rf",
                     metric = "ROC",
                     trControl = Folds,
                     tuneGrid = rf.expand)
diabetes.rf
varImpPlot(diabetes.rf$finalModel, main = "Diabetes Random Forest")
#Creates Linear SVM
linear.tune <- expand.grid(C = c(seq(.5, 5, by=.5)))
set.seed(125)
lsvm <- train(XTrain,
              YTrain,
              method = "svmLinear",
              metric = "ROC",
              trControl = Folds,
              tuneLength = 10,
              tuneGrid = linear.tune)
lsvm
lsvm.fit = ksvm(Diagnosis~DiastolicPressure+PlasmaGlucose, data = diabetes,type = 'C-svc', kernel = 'vanilladot')
plot(lsvm.fit, data=diabetes)
radial.svm.expand <- expand.grid(sigma = c(2,3,4,5),
                                 C = c(.2,.4,.6,.8))
set.seed(125)
rsvm <- train(XTrain,
              YTrain,
              method = "svmRadial",
              metric = "ROC",
              trControl = Folds,
              tuneGrid = radial.svm.expand)
rsvm
rsvm.fit = ksvm(Diagnosis~DiastolicPressure+PlasmaGlucose, data = diabetes,type = 'C-svc', kernel = 'rbfdot')
plot(rsvm.fit, data=diabetes)
confusionMatrix(diabetes.rf)
confusionMatrix(lsvm)
confusionMatrix(rsvm)
