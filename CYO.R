# Analysis of the Adult Income Dataset
# Libraries loaded using if(!require) function
if (!require(formatR)) 
  install.packages("formatR", repos = "http://cran.us.r-project.org")
library(formatR)
if (!require(tidyverse)) 
  install.packages("tidyverse", repos = "http://cran.us.r-project.org")
library(tidyverse)
if (!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
library(caret)
if (!require(RCurl)) install.packages("RCurl",repos = "http://cran.us.r-project.org")
library(RCurl)
if (!require(ggplot2)) install.packages("ggplot2", repos = "http://cran.us.r-project.org")
library(ggplot2)
if (!require(dplyr)) install.packages("dplyr", repos = "http://cran.us.r-project.org")
library(dplyr)
if (!require(randomForest)) 
  install.packages("randomForest", 
                   repos = "http://cran.us.r-project.org")
library(randomForest)
if (!require(e1071)) install.packages("e1071", repos = "http://cran.us.r-project.org")
library(e1071)
if (!require(rpart)) install.packages("rpart", repos = "http://cran.us.r-project.org")
library(rpart)
if (!require(rpart.plot)) install.packages("rpart", repos = "http://cran.us.r-project.org")
library(rpart.plot)
if (!require(ROSE)) install.packages("ROSE", repos = "http://cran.us.r-project.org")
library(ROSE)
# Dataset loaded from Github (requires internet connection)
options(timeout = 120)
x <- getURL("https://raw.githubusercontent.com/bkhooze/MovieLens/refs/heads/main/adult.csv")
salary <- read.csv(text = x)
# Gain an overview of the dataset
head(salary)
glimpse(salary)
# Noted in this dataset missing values coded as ?, recode this to NA
salary[salary == "?"] <- NA 
salary %>% summarise_all(~ sum(is.na(.)))
# Most columns have no missing values except workclass, occupation and native.country
sum(is.na(salary$occupation))/length(salary$occupation)*100
table(salary$workclass)
table(salary$occupation)
# Following code changes the NA values in the column workclass to "Private", 
# which is the most common observation.
salary <- salary %>% mutate(workclass = ifelse(is.na(workclass), "Private", workclass))
# Decision for complete case analysis and NA values omitted
salary <- na.omit(salary)
salary %>% summarise_all(~ sum(is.na(.)))
glimpse(salary)
# The following code detects and transforms the columns to the appropriate type
summary(salary)
salary[]<- lapply(salary,trimws)
num <- c(1,3,5,11,12,13)
salary[num] <- sapply(salary[num], as.numeric)
# Transform appropriate columns to numeric type
cat <- c(2,4,6,7,8,9,10,14)
salary[,cat] <- lapply(salary[,cat],factor)
# Transform appropriate columns to factor type
# Explore columns with multiple repeated values
str(salary)
table(salary$capital.gain)
table(salary$capital.loss)
table(salary$fnlwgt)
# Removed columns fnlwgt, capital.gain and capital.loss in view of multiple repeated values, with more than 20000 values are "0". Also uncertain of how these affects the outcome variable, income.
salary <- salary [-c(3,11,12)]
str(salary)
# Data visualisation - presenting the following as bar graphs
# Income grouped by sex and education level.
salary %>%
  group_by(sex, income) %>%
  summarise(count = n(), .groups = 'drop') %>%
  ggplot(aes(sex, count, fill = income)) +
  geom_bar(stat="identity", position = "dodge") + xlab("Sex") + ylab("Count")
salary %>%
  group_by(education.num, income) %>%
  summarise(count = n(), .groups = 'drop') %>% ggplot(aes(education.num, count, fill = income)) +  geom_bar(stat="identity", position = "dodge") + xlab("Education Number") + ylab("Count")
# Marital status has multiple categories - aim to recode as binary
table(salary$marital.status)
# Noted marital status consists of multiple values, to convert this to binary outcome - married vs not
salary <- salary %>% mutate(marriage_binary = ifelse(marital.status %in% c("Married-civ-spouse", 
                                                                           "Married-AF-spouse", "Married-spouse-absent"), 1, 0))
# Visualise income by married vs not married
salary %>%
  group_by(marriage_binary, income) %>% 
  summarise(count = n(), .groups = 'drop') %>% ggplot(aes(marriage_binary, count, fill = income)) + geom_bar(stat="identity", position = "dodge") + xlab("Not Married                                 Married") + ylab("Count") + scale_x_discrete(labels = NULL, breaks = NULL)
# Visualise income by race
salary %>% group_by(race, income) %>% summarise(count = n(), .groups = 'drop') %>%
  ggplot(aes(race, count, fill = income)) + geom_bar(stat="identity", position = "dodge") + xlab("Race") + ylab("Count") + scale_x_discrete(labels = abbreviate)
table(salary$occupation)
# Visualise income grouped by occupation
salary %>% group_by(occupation, income) %>% summarise(count = n(), .groups = 'drop') %>%
  ggplot(aes(occupation, count, fill = income)) + geom_bar(stat="identity", position = "dodge") + scale_x_discrete(labels = abbreviate)
# The abbreviations for occupation are listed in the table above. 
# Executive, professional and sales jobs seem to have the highest proportion of income earners > $50k.
# Visualise income by age
salary %>% group_by(age, income) %>% summarise(count = n(), .groups = 'drop') %>%
  ggplot(aes(age, count, fill = income)) + geom_bar(stat="identity", position = "stack") 
# Individuals in the 30 to 50 age range have the highest proportion of people earning >$50k a year.
# Recode income into binary outcomes, 0 if income <$50K and 1 if income >$50K. Drop marital status column as this has been coded into binary. Drop education column as this is coded in education.num.
salary <- salary %>% mutate(income = ifelse(income == c("<=50K"), 0, 1))
salary <- salary [-c(3,5)]
str(salary)
# Check if there is correlation between columns which are numeric with each other
correlation_var <- c("age", "education.num", "hours.per.week", "marriage_binary", "income")
correlation <- round(cor(salary[correlation_var]),2)
correlation
# In the numeric variables in the dataset for analysis, there are no variables that are highly correlated with each other and thus all variables were included for the analysis.
# Partition into test and train set. Decision to use 0.3 for test set as outcome is somewhat unbalanced.
set.seed(1, sample.kind = "Rounding")
test_index <- createDataPartition(salary$income, times = 1, p = 0.3, list = FALSE)
salary_train <- salary[-test_index,]
salary_test <- salary[test_index,]
# To deal with unbalanced dataset, oversampling strategy using ROSE selected
oversampled_data <- ovun.sample(income ~ ., data = salary_train, method = "over", N = 31656)$data
table(oversampled_data$income)
# 4 models compared - logistic regression, random forest, support vector machines, decision tree
# Model 1: Logistic regression using oversampling to correct for imbalanced dataset
fit_glm <- glm(income ~ ., data = oversampled_data, family = binomial("logit"))
p_glm <- predict(fit_glm, salary_test, type = "response")
p_glm <- as.factor(ifelse(p_glm > 0.5, "1", "0"))
salary_test$income <- as.factor(salary_test$income)
m1 <- confusionMatrix(p_glm, salary_test$income)
m1
# Model 2: Random Forest
fit_rf <- randomForest(income~., data = oversampled_data, ntree = 500)
pred_rf <- predict(fit_rf, salary_test, type = "response")
pred_rf <- as.factor(ifelse(pred_rf > 0.5, "1", "0"))
m2 <- confusionMatrix(pred_rf, salary_test$income)
m2
# Model 3: Support Vector Machines
fit_svm <- svm(income ~ ., data = oversampled_data)
pred_svm <- predict(fit_svm, newdata = salary_test, type = "response")
pred_svm <- as.factor(ifelse(pred_svm > 0.5, "1", "0"))
m3 <- confusionMatrix(pred_svm, salary_test$income)
m3
# Model 4: Decision tree
fit_dectree <- rpart(income~.,data=oversampled_data,method = "class")
rpart.plot(fit_dectree, box.col=c("red","blue"))
pred_dectree <- predict(fit_dectree, newdata = salary_test, type = "class")
m4 <- confusionMatrix(pred_dectree, salary_test$income,positive="1")
m4
# F1 score was also calculated as this helps when dataset is imbalanced and also produces a more stable model performance. 
f1_score <- function(model)
{
  precision <- model$byClass["Pos Pred Value"]
  recall <- model$byClass["Sensitivity"]
  f1 <- 2 * (precision * recall) / (precision + recall)
}
f1m1 <- f1_score(m1)
f1m2 <- f1_score(m2)
f1m3 <- f1_score(m3)
f1m4 <- f1_score(m4)
# Results presented graphically as a tibble
options(digits = 3)
results <- tibble(Model = c("Logistic Regression Balanced", "Random Forest", 
                            "Support Vector Machines", "Decision Tree"),
                  Accuracy = c(m1$overall["Accuracy"],m2$overall["Accuracy"],
                               m3$overall["Accuracy"],m4$overall["Accuracy"]),
                  Sensitvity = c(m1$byClass["Sensitivity"],m2$byClass["Sensitivity"],
                                 m3$byClass["Sensitivity"],m4$byClass["Sensitivity"]),
                  Specificity = c(m1$byClass["Specificity"],m2$byClass["Specificity"],
                                  m3$byClass["Specificity"],m4$byClass["Specificity"]),
                  F1score = c(f1m1,f1m2,f1m3,f1m4)
)
results
# The accuracy, sensitivity and specificity and F1 score of the various models to predict whether an individual earns $50k or more is displayed in the table above. These results were derived using the oversampling method to deal with an unbalanced dataset. Overall, the random forest model had the best accuracy of 0.810 and F1 score of 0.868 to predict the outcome.
