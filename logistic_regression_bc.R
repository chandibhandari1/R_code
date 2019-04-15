####Application of Logistic regression from the  breast cancer data set which can be found on the package "mlbench"

####@Created by Chandi Bhandari#######on 2/23/2019################

#Target=to classify the given spicemen is benign or malignant


#adding the required packages
install.packages("mlbench")
library(healthcareai)
library(dplyr)
library(plyr)
library(sqldf)
library(ggplot2)
#installing the package for data diagnosis and munging
installed.packages("dlookr")
library(dlookr)
library(caret)

#installing packages for spliting set 
install.packages("caTools")
library(caTools)

# Loading the data
data(BreastCancer, package = "mlbench")
Bcancer <- BreastCancer

# Looking out the missing values
missingness(Bcancer)

# There is missing values in Bare nuclei feature lets remove the na values
BC <- na.omit(Bcancer)

#Checking out the structure of data
str(BC)

#Application of logistic regression "glm" without preprocessing data
glm(Class ~ Cell.shape, family = "binomial", data = BC)

#now removing the id from the dataset 
BC <- BC[, -1]

# From the above structure we saw all the values are the factors convert them to numeric:except for the class
for (i in 1: 9){
  BC[, i] <-as.numeric(as.character(BC[,i]))
}
#varifying the result
str(BC)

#Encoding the response variable into factor variables of 1s and 0s

##BC$Class <- ifelse(BC$Class == "malignant", 1, 0)
##BC$Class <- factor(BC$Class, levels = c(0,1))   #Or the folling shortCut method
BC['Class'] <- lapply(BC['Class'], factor, levels=c("benign", "malignant"), labels = c(0, 1))

# Looking at the data and checking for the balancing
table(BC$Class)

#this show there are 444= benign and 239=malignant, we need to have balacning them
#Use the SMOTE techniques

#adding the library for Linear regression and Classfication: split, pre-processing, feature selection
#model tuning using resampling, variable importance estimate
library(caret)
set.seed(100)
# Splitting the data into training and test set at the rate of 70-30 rate
trainIndex <- createDataPartition(BC$Class, p=0.7, list = FALSE)
TrData <- BC[trainIndex, ]
TeData <- BC[-trainIndex, ]

#Checking the data balancing in training and test
table(TrData$Class) #results shows 311=belign, 168 =malignant
table(TeData$Class) #results shows 133 =belign, 71= malignant

#Balancing the training data set either by under sampling or over sampling techniques (SMOTE techniques)
# Under sampling techniques
#Define the "not in the function"
'%notin%' <- Negate('%in%')
options(scipen=999)
Down_train <- downSample(x = TrData[, colnames(TrData)  %notin% "Class"], y =TrData$Class)

#Up sampling techniques
Up_train <- upSample(x=TrData[, colnames(TrData) %notin% "Class"], y=TrData$Class)

#Checking the balance in both up_train and down_train
table(Down_train$Class) # both have 168 how cutting down
table(Up_train$Class) # both have 311 getting up

#Fitting the logistic regression 
colnames(Down_train)
LogisticReg <- glm(Class ~ Cl.thickness + Cell.size + Cell.shape, family = "binomial", data = Down_train)
summary(LogisticReg)

#LogisticReg is build: now turn to make prediction
prediction <- predict(LogisticReg, newdata =TeData, type = "response" )

#Here the prediction contains the probability that the observation is =malignant 
#Now separate the data as if prob >= 0.5 then 1= malignant otherwise Benign=0
#Assigning values 1 or 0 based on their probability >= 0.05

Y_predictedno <- ifelse(prediction >= 0.5, 1, 0)
Y_predicted <- factor(Y_predictedno, levels = c(0,1))
#Looking our actual 
Y_act <- TeData$Class

#Testing the accuracy
mean(Y_predicted == Y_act) #gives 0.941 =94% accurate prediction

######################################UP Sampling #####################################################
###Working on Up_sampling on
LogisticReg1 <- glm(Class ~ Cl.thickness + Cell.shape+ Cell.size, family ="binomial", data = Up_train)
summary(LogisticReg1)

prediction1 <- predict(LogisticReg1, newdata = TeData, type = "response")
Y_predictednumber <- ifelse(prediction1 >= 0.5, 1,0)
Y_predicted1 <- factor(Y_predictednumber, levels = c(0,1))
Y_actual <- TeData$Class
#cheking the accuracy in upsampling
mean(Y_predicted1 == Y_actual) #giving the same amount of accuracy =94%

##################### Using all predictor variables
###Working on Up_sampling on
LogisticReg2 <- glm(Class ~ Cl.thickness + Cell.shape+ Cell.size + Marg.adhesion + Epith.c.size + Bare.nuclei + Bl.cromatin +Normal.nucleoli + Mitoses , family ="binomial", data = Up_train)
summary(LogisticReg2)

prediction2 <- predict(LogisticReg2, newdata = TeData, type = "response")
Y_predictednumber2 <- ifelse(prediction2 >= 0.5, 1,0)
Y_predicted2 <- factor(Y_predictednumber2, levels = c(0,1))
Y_actual2 <- TeData$Class
#cheking the accuracy in upsampling
mean(Y_predicted2 == Y_actual2) #giving the little better prediction =96% which means we leftout one important variable
