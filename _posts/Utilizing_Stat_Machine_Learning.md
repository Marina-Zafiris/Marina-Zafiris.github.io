---
title: "Utilizing Statistical and Machine Learning Models to Best Predict Graduate Level Admittance"
date: 2019-05-01T15:34:30-04:00
categories:
  - blog
tags:
  - decision tree
  - linear regression
  - neural network
---

The purpose of this research is to compare and find out
which statistical/machine learning model is the best at
predicting graduate admittance. The models, which were
used and compared, were multiple linear regression, neural networks, and decision trees. Linear regression is based on the method of least squares that reduces the Mean Squared Error (MSE), which is attributed to Gauss and Legendre. Multiple linear regression involves using more than one independent variable to estimate the coefficients of the predictors for the function by best reducing error. From this, we can use those coefficients for our future models. One of
the methods employed to best predict admittance is a Neural Network. Artificial neural networks are a composition of the processes of finding the parameter and the utilization of an error function, usually a sigmoid function. Decision Trees use a probabilistic modeling approach to classify predictors which results in one of the categories of outcomes.

Our objective is to examine and analyze student graduate admission based on GRE Scores, TOEFL Scores, University Ranking, Statement of Purpose (SOP), Letter of Recommendation (LOR), Academic Performance (GPA) and lastly, Projects and Research. The dependent variable (y) being Chance of Admission.

```r
install.packages("MASS")
install.packages("fitdist")
library(fitdist)
library(olsrr)
library(MPV)
library(car)
library(psych)
dat = read.csv("C:/Users/Vasilis/Desktop/Spring 2019 UHD/Linear Regression/Admission.csv", header = TRUE)
head(dat)
class(dat)
```

The dataset must first be restructured in order to be easily manipulated. We first eliminate "Serial No." category due to it's redundancy, as it does not act as a regressor variable in our analysis, but rather identification column. The response variable  is identified as the Chance of Admission column. And the remaining columns are assigned to the following regressor variables:
x1 - GRE Score
x2 - TOEFL Score
x3 - University Ranking
x4 - Statement of Purpose (SOP)
x5 - Letter of Recommendation (LOR)
x6 - Academic Performance (GPA)
x7 - Projects and Research

```r
y = dat[,9]
x1 = dat[,2]
x2 = dat[,3]
x3 = dat[,4]
x4 = dat[,5]
x5 = dat[,6]
x6 = dat[,7]
x7 = dat[,8]

data.1 = data.frame(y, x1, x2, x3, x4, x5, x6, x7)

# Ensuring the data is all numeric
data.2 = as.data.frame(sapply(data.1, as.numeric))

# Better Look at our Data (Less Crowded Compared to Normal Plotting)
pairs(data.2, pch=16, cex= .1)

# Linear Model of entire Data Set
lm.2 = lm(y~ .,data= data.2)
lm.2
plot(lm.2)
```

#Data Cleaning
Most of the time, the dataset will have lot of anomalies, such as outliers. In order to ensure, accurate analysis we must conduct outlier treatment and feature treatment.
An outlier is an observation that lies an abnormal distance from other values in a random sample. Bellow we display how to eliminatemate certain outliers utilizing Cooks Distance Method. Any points that lay above the approximate 3rd Quartile Range of Distance from the median will be eliminated, in essences deleting the rows that contain anomaly data.

```r
# Eliminating Outliers with Cooks
cooks <- cooks.distance(lm.2)
summary(cooks)
data.3 <- data.2[cooks < 0.002, ]
```

Feature engineering consists of combining, adding, deleting and/or scaling certain features of the dataset in order to increase the accuracy of the model. Below, we are utilizing Power Transform tool in order to transform Y to better scale our model.

```r
# Building a new dataframe with Transformed Y

powerTransform(y ~., data=data.3)

y.trans1 <- ((data.3[,1])^1.29158 )

x.new <- data.3[,2:8]

DataNew = cbind.data.frame(y.trans1, x.new)
```

This reduces our new dataset to 387 observations.

# Variable Selection
We are utilizing ols_step_best_subset in order to select the "best" or most important" subset from a large pool of candidate regressor variables. Our main purpose is to remove redundant regressors.

```{r,echo=FALSE}

ols_step_best_subset(lm( y.trans1 ~ . , data= DataNew) )
#plot(ols_step_best_subset(lm(y.trans1~., data.new1) ) )
```

We must construct a model that removes redundant regressors without oversimplifying the data. Therefore we must have more than one regressor, with the most optimal Standard Error and R^2 (goodness of fit/Coefficient of determination) score. In essence, whichever accounts for a greater scope of the data, that is which explains more of the data.

```r
lm(y.trans1~x1+x6, data=DataNew)
lm(y.trans1~x2+x6+x7, data=DataNew)
lm(y.trans1~x2+x3+x6+x7, data=DataNew)
```

The above regressor test displays a model (y~x1+x6) with regressors x1, x6. The linear model is as follows:
$\hat{y} =  -2.068708 + 0.003742x_{1} + 0.181215x_{6}$

The above regressor test displays a model (y~x2+x6+x7) with regressors x1, x6. The linear model is as follows:
$\hat{y} =  -1.421259 + 0.005138x_{2} + 0.176941x_{6} + 0.043022x_{7}$

The above regressor test displays a model (y~x2+x3+x6+x7) with regressors x1, x6. The linear model is as follows:
$y =  -1.273251 + 0.004552x_{2} + 0.015771x_{3} + 0.161662x_{6} + 0.037628x_{7}$  

#Comparing Models

Original Model
```r

PRESS(lm( y.trans1~ . , data= DataNew))
vif(lm(y.trans1~ . , data=DataNew))

ols_test_normality(lm(y.trans1~ . , data=DataNew))

ols_plot_resid_fit(lm(y.trans1~ . , data=DataNew))
```
The VIF (variance inflation factor) of the adjusted all regressor model displays multicollinearityfor X1 ad X6. The bellow models will account for this problem.

Model 1
```r

PRESS(lm(y.trans1~x1+x6, data=DataNew))

vif(lm(y.trans1~x1+x6, data=DataNew))

ols_test_normality(lm(y.trans1~x1+x6, data=DataNew))

ols_plot_resid_fit(lm(y.trans1~x1+x6, data=DataNew))
```

Model 2
```r

PRESS(lm(y.trans1~x2+x6+x7, data=DataNew))

vif(lm(y.trans1~x2+x6+x7, data=DataNew))

ols_test_normality(lm(y.trans1~x2+x6+x7, data=DataNew))

ols_plot_resid_fit(lm(y.trans1~x2+x6+x7, data=DataNew))
```

Model 3
```r

PRESS(lm(y.trans1~x2+x3+x6+x7, data=DataNew))

vif(lm(y.trans1~x2+x3+x6+x7, data=DataNew))

ols_test_normality(lm(y.trans1~x2+x3+x6+x7, data=DataNew))

ols_plot_resid_fit(lm(y.trans1~x2+x3+x6+x7, data=DataNew))
```

All of our models look exceptionally good. The VIF's of all three models state there is no multi-collinearity present amongst the regressors of each model;
Lets compare the R^2 values given by the ols_step_best_subset:
The R^2 of Model 1 is 0.9249.
The R^2 of Model 2 is 0.9359.
The R^2 of Model 3 is 0.9421.

The greatest R^2 value is of Model 3, however the difference is almost negligible. And our goal is to almost always find the most simple model for our dataset, usually leading us to the model with the least amount of regressors. Hence, why the OLS listed model one as the best choice

  In comparing, our models to outside datasets its safe to assume that GRE Scores is of more importance to U.S based schools for graduate admission than TOEFL schools. The GRE is a graduate school entrance exam while the TOEFL is a test of your English language skills. Schools want to see GRE scores to ensure proper handling of graduate-level coursework. In contrast, TOEFL scores reflect your English skills and your ability to perform at an English-speaking school. The two scores could display similar trends and display multicollinearity. Hence, why none of the three models contain both X1 (GRE) and X2 (TOEFL).

Due to this assumption, the most comprisable model is Model 2, so our final linear model is
$\hat{y} =  -2.068708 + 0.003742x_{1} + 0.181215x_{6}$

For every unit change in GRE Score $x_{1}$ leads to a positive 0.003742 change in $\hat{y}$, holding all other variables constant. For every unit change in Cumulative GPA $x_{6}$ leads to a positive 0.181215 change in y, holding all other variables constant.

```r
fit = lm(y.trans1~x1+x6, data=DataNew)
summary(fit)
```

The summary of the fitted linear model displays all the regressors having a p-value of <2e-16, which is smaller than α of 0.05. We reject Ho and conclude that there is a significant linear relationship between Chance of Admission and GRE Scores and Cumulativelitive GPA.
In addition, with p-value = $<2*10^{-16}$, we reject H0 : β0 = 0(or no-intercept model).
Our Residual standard error -  σˆ = 0.04121.
The model displays a Multiple R^2 value of 0.9249, reflecting that approximately 92.5% of the variance in Admission Rate is explained by Gre Scores and Cumulative GPA.

```r
#install.packages("psych")

corr.test(DataNew[1:7])
pairs.panels(DataNew[1:7])
```

# Confidence Intervals of Regressors
Now we will find the 95% confidence interval for x1 (GRE Scores) and x6 (Cumulative GPA for our multiple linear regression model, y~x1+x6

```r
newconfint <- confint(fit)
newconfint
```

With our new model only containing x1 and x6, we are 95% confident that our coefficient for our x1 will lie within the interval (0.003035546, 0.004447458) and x6 will lie within the interval (0.168130089  0.194300508).



# Neural Network
```r
#install.packages("caret")
#install.packages("lattice")
library(MASS)
library(neuralnet)
library(ggplot2)
library(caTools)
library(caret)
#------------------------------------------------------------------------------------------------------------#
#                                                 Neural Network                                             #
#------------------------------------------------------------------------------------------------------------#


#start with Graduate Admission Dataset


dat = read.csv("C:/Users/Vasilis/Desktop/Spring 2019 UHD/Linear Regression/Admission.csv", header = TRUE)
class(dat)


str(dat)
# predicting chance low birth weight with the other predictors

Binom_Admit = ifelse(dat$Chance.of.Admit > 0.72, 1, 0)

head(Binom_Admit)



dat_bw=cbind(dat[ , -9], Binom_Admit)

#### All are integer, if not then we would have to convert
#### with the as. function
str(dat_bw)



#-------------------------------------------------------------------------------#
####  To run a neural network it is best to normalize the data
#### normalization with min max
#### [x - min(x)]/[max(x) - min(x)] for all x in X
#### making a function to do that


#normalize = function(x)
#{
#  return( (x - min(x))/(max(x) - min(x)) )
#}


#sdata = as.data.frame(apply( X = dat_bw, MARGIN = 2, FUN = normalize ))

#head(sdata)
#-------------------------------------------------------------------------------#
for (i in 1 : ncol(dat_bw) )
{
  dat_bw[i] = (dat_bw[,i] - min(dat_bw[,i]))/ ( max(dat_bw[,i]) - min(dat_bw[i]) )
}


#### Data Partition
set.seed(500)

ind = sample.split(Y = dat_bw$Binom_Admit, SplitRatio = 0.7)

training = dat_bw[ ind, ]

testing = dat_bw[ !ind, ]

#### Neural Network model
#### first we make a formula
n  =  names(dat_bw)

f = as.formula(paste("Binom_Admit ~", paste(n[!n %in% "Binom_Admit"], collapse = " + ")))

####Neural Network model

nn = neuralnet(formula = f,
               data = training,
               hidden = c(3, 2),
               linear.output = FALSE )



#### plotting te neural network

plot(nn)


#### Prediction
out1 = compute(nn, training[ ,-9])  # all rows but not the response variable/column

head(out1)  # the whole output

p1 = out1$net.result   # probability of predicted values

head(out1$net.result)


pred1 = ifelse(p1 > 0.5, 1 , 0)

tab1 = table(pred1, training$Binom_Admit)

tab1

# Accuracy
a1 = sum(diag(tab1))/sum(tab1)
a1

#Mean Squred Error  (MSE)
e = 1 - a1
e


#Confusion Matrix and Misclassification Error - testing data


#### Prediction
out2 = compute(nn, testing[,-9])  # all rows but not the response variable/column

out2  # the whole output

p2 = out2$net.result   # probability of predicted values

head(out2$net.result)


pred2 = ifelse(p2 > 0.5, 1 , 0)

tab2 = table(pred2, testing$Binom_Admit)

head(tab2)

# Accuracy
a1 = sum(diag(tab2))/sum(tab2)
a1

#Mean Squred Error  (MSE)
e = 1 - a1
e

```

#Decision Tree

```r

# Decision Tree for Model 1 (all regressors)
library(C50)
# Shuffle data
set.seed(9850)
dat_bw$Binom_Admit <- as.factor(dat_bw$Binom_Admit)
g <- runif(nrow(dat_bw))
DataR <- dat_bw[order(g),]
str(DataR)
# Classification Tree with Number of samples: 400 and Number of predictors: 9
m1 <- C5.0(DataR[1:400, -9], DataR[1:400, 9])
summary(m1)
# Predicting
p1 <- predict(m1, DataR[401:500,])
table(DataR[401:500, 9], predicted = p1)
# Plotting M1
plot(m1)

```



x2 - TOEFL Score
x6 - Academic Performance (GPA)
x7 - Projects and Research

```r

# Decision tree for model 2 (x2 - TOEFL Score, x6 - Academic Performance (GPA)x7 - Projects and Research)

library(C50)
# # Shuffle data
set.seed(9850)

dat_bw$Binom_Admit <- as.factor(dat_bw$Binom_Admit)
new_dat <- dat_bw[ , c(3, 7, 8, 9)]
head(new_dat)
g <- runif(nrow(new_dat))
DataR <- new_dat[order(g),]
str(DataR)

# Classification Tree with Number of samples: 400 and Number of predictors: 4
m1 <- C5.0(DataR[1:400, -4], DataR[1:400, 4])
summary(m1)


# Predicting
p1 <- predict(m1, DataR[401:500,])
table(DataR[401:500, 4], predicted = p1)


# Plotting M1
plot(m1)
```
