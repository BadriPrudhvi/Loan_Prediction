library(dplyr)
library(caret)
library(xgboost)
library(tidyr)
library(lubridate)
library(dummies)
library(VIM)
library(caret)

train_file <- read.csv("~/Documents/Analytics_Vidhya/Loan_Prediction/train.csv", na.strings=c("","NA"))
test_file <- read.csv("~/Documents/Analytics_Vidhya/Loan_Prediction/test.csv", na.strings=c("","NA"))

test_file$Loan_Status <- "Y"
Full_Data <- rbind(train_file,test_file)

Full_Data$Education <- ifelse(Full_Data$Education == "Graduate",1,0)
Full_Data$Gender <- ifelse(Full_Data$Gender == "Male",1,0)
Full_Data$Married <- ifelse(Full_Data$Married == "Yes",1,0)
Full_Data$Self_Employed <- ifelse(Full_Data$Self_Employed == "Yes",1,0)
Full_Data$Dependents <- ifelse(Full_Data$Dependents == "3+",3,
                               ifelse(Full_Data$Dependents == "2",2,
                                      ifelse(Full_Data$Dependents == "1",1,0)))
Full_Data <- dummy.data.frame(Full_Data, names = c('Property_Area'),  sep='_')


preProcValues <- preProcess(Full_Data, method = c("bagImpute"))
Full_Data <- predict(preProcValues, Full_Data)

#### Feature Engineering ####

Full_Data <- Full_Data %>%
  mutate(Total_Income = ApplicantIncome + CoapplicantIncome,
         Simple_Interest = LoanAmount * (1 + (0.09 * (Loan_Amount_Term/12/12))),
         Savings_Money = Total_Income - (Simple_Interest),
         ratio_income_loan = LoanAmount/Total_Income,
         ratio_term_loan = LoanAmount/Loan_Amount_Term,
         ratio_saving_income = Savings_Money/Total_Income)

Full_Data[,c("ApplicantIncome","CoapplicantIncome")] <- NULL
Full_Data$Loan_Status <- as.factor(Full_Data$Loan_Status)


### Seperating into Train and Test sets ####

train_data <- subset(Full_Data,Loan_ID %in% train_file$Loan_ID)
test_data <- subset(Full_Data,Loan_ID %in% test_file$Loan_ID)

train_data$Loan_ID <- NULL
test_data <- test_data %>%
  arrange(Loan_ID)
test_data[,c("Loan_ID","Loan_Status")] <- NULL

#### Removing Outliers ####
train_data <- subset(train_data, LoanAmount < quantile(train_data$LoanAmount,0.75,na.rm = T) + 1.5*IQR(train_data$LoanAmount,na.rm = T))
train_data <- subset(train_data, Total_Income < quantile(train_data$Total_Income,0.75,na.rm = T) + 1.5*IQR(train_data$Total_Income,na.rm = T))

#### Building Models ####
control <- trainControl(method = "cv",
                        number = 10,
                        classProbs = TRUE,
                        verboseIter = TRUE)

# Fit GBM
modelGbm <- train(Loan_Status~., 
                  data=train_data, 
                  method="gbm", 
                  trControl=control, 
                  preProcess = c("center","scale"),
                  verbose= TRUE)
print(modelGbm)
plot(modelGbm)
gbmPred <- predict(modelGbm, test_data)
gbm_output <- data.frame(Loan_ID = test_file$Loan_ID,Loan_Status = gbmPred)
write.csv(gbm_output,row.names = FALSE,"~/Documents/Analytics_Vidhya/Loan_Prediction/gbm_output.csv")

# Fit random forest
modelRF <- train(Loan_Status~., 
                  data=train_data, 
                  tuneLength = 3,
                  method = "ranger",
                  tuneGrid = data.frame(mtry=c(1,2,3,7,9,15)),
                 preProcess = c("center","scale"),
                  trControl = control)

# Print model to console
print(modelRF)
plot(modelRF)
RFPred <- predict(modelRF, test_data)
RF_output <- data.frame(Loan_ID = test_file$Loan_ID,Loan_Status = RFPred)
write.csv(RF_output,row.names = FALSE,"~/Documents/Analytics_Vidhya/Loan_Prediction/RF_output.csv")


# Fit Decision Tree
modelTree <- train(Loan_Status~., 
                   data=train_data, 
                   tuneLength = 3,
                   method = "rpart",
                   preProcess = c("center","scale"),
                   trControl = control)

# Print model to console
print(modelTree)
plot(modelTree)
treePred <- predict(modelTree, test_data)
Tree_output <- data.frame(Loan_ID = test_file$Loan_ID,Loan_Status = treePred)
write.csv(Tree_output,row.names = FALSE,"~/Documents/Analytics_Vidhya/Loan_Prediction/Tree_output.csv")

# Fit SVM
modelSVM <- train(Loan_Status~., 
                   data=train_data, 
                   tuneLength = 3,
                   method = "svmRadial",
                  preProcess = c("center","scale"),
                   trControl = control)

# Print model to console
print(modelSVM)
plot(modelSVM)
svmPred <- predict(modelSVM, test_data)
svm_output <- data.frame(Loan_ID = test_file$Loan_ID,Loan_Status = svmPred)
write.csv(svm_output,row.names = FALSE,"~/Documents/Analytics_Vidhya/Loan_Prediction/svm_output.csv")


## Fit XGBOOST

train_xgb <- train_data %>%
  mutate(Loan_Status = ifelse(Loan_Status =="Y",1,0))

xgb_params_1 = list(
  objective = "binary:logistic",                                               # binary classification
  eta = 0.01,                                                                  # learning rate
  max.depth = 6,                                                               # max tree depth
  eval_metric = "auc"                                                          # evaluation/loss metric
)

# cross-validate xgboost to get the accurate measure of error
xgb_cv_1 = xgb.cv(params = xgb_params_1,
                  data = as.matrix(train_xgb %>%
                                     select(-Loan_Status)),
                  label = train_xgb$Loan_Status,
                  nrounds = 1000, 
                  nfold = 10,                                                   # number of folds in K-fold
                  prediction = TRUE,                                           # return the prediction using the final model 
                  showsd = TRUE,                                               # standard deviation of loss across folds
                  stratified = TRUE,                                           # sample is unbalanced; use stratified sampling
                  verbose = TRUE,
                  print.every.n = 1,
                  metrics = "map"
)

nround = which(xgb_cv_1$dt$test.auc.mean == max(xgb_cv_1$dt$test.auc.mean))[1]
xgb_cv_1$dt[nround,]

XGBModel <- xgboost(param=xgb_params_1, 
                    data = as.matrix(train_xgb %>%
                                       select(-Loan_Status)),
                    label = train_xgb$Loan_Status,
                    nrounds=nround)

xgb_predict <- predict(XGBModel, as.matrix(test_data))
xgb_prediction <- as.numeric(xgb_predict > 0.3)
xgb_output <- data.frame(Loan_ID = test_file$Loan_ID,
                         Loan_Status = ifelse(xgb_prediction==1,"Y","N"))
write.csv(xgb_output,row.names = FALSE,"~/Documents/Analytics_Vidhya/Loan_Prediction/xgb_output.csv")
var.names = colnames(train_data)

Imp <- xgb.importance(feature_names = var.names, model = XGBModel)
Imp[1:17]

# plot the AUC for the training and testing samples
xgb_cv_1$dt %>%
  select(-contains("std")) %>%
  dplyr::mutate(IterationNum = 1:n()) %>%
  gather(TestOrTrain, AUC, -IterationNum) %>%
  ggplot(aes(x = IterationNum, y = AUC, group = TestOrTrain, color = TestOrTrain)) + 
  geom_line() + 
  theme_bw()

#********************************************************
#********************************************************
#********************************************************

# Create model_list
model_list <- list(gbm = modelGbm, rf = modelRF,tree = modelTree, svm = modelSVM)

# Pass model_list to resamples(): resamples
resamples <- resamples(model_list)

# Summarize the results
summary(resamples)
bwplot(resamples,metric="Accuracy")
varImp(modelGbm)
