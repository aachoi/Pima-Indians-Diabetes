# Predicting Affordabilitty of Houses in Ames, Iowa
## Abstract
The purpose of this project was to utilize advanced classification techniques like
logistic regression, bagging, and random forest in order to build a model to predict the
affordability of residential homes in Ames, Iowa. We applied rigorous data cleaning, in
particular to deal with missing values and outliers in the data set. In order to further
reduce the complexity of the data, we utilized dimension reduction methods like feature
engineering and principal component analysis. Our concluding clean data set had
dimensions of 5,000 rows and 65 predictors. Our approach placed a heavy emphasis on
data cleaning rather than modeling; thus our time was spent mostly on data cleaning.
The models we used were logistic regression, tree, bagging, and random forest. We
found random forest to yield the highest accuracy. From there, our various attempts
focused on fine-tuning different parameters like the number of trees and the number of
predictors sampled for each split (mtry). We attribute our high accuracy results to the
meticulous amount of data cleaning conducted. Prior to data cleaning, models on the
uncleaned dataset yielded an accuracy to be 84% or lower. With data cleaning, the accuracy
jumped to 94% or higher.

In conclusion, our best model was a random forest with 10 predictors. The model
yielded a 98% validation set accuracy and a 97.98% Kaggle accuracy.
## Introduction
The dataset describes the sale of individual residential properties in Ames, Iowa from
2006 to 2010 (5,000 rows) as well as explanatory variables about different features of the
properties (79 columns). The variables detail different aspects of the property that
interested home buyers would want to know about a potential property. Our response
variable _affordabilitty_ was defined as anything that fell above or below the median price of
houses in Ames, Iowa. We aim to identify the important features of a property to build a
highly accurate classification model to predict the affordability of residential homes in
Ames, Iowa.
## Methodology
### Initial Attempt: Logistic Regression
We believed features such as _OverallQual_, _OverallCond_, _YearBuilt_, _TotalBsmtSF_,
_BsmtFullBath_, and more would be important features when predicting the affordability of
houses. Prior to any data cleaning, we wanted to see how a rudimentary logistic model with
our assumptions would fare. After splitting our training data into a 90% training and 10%
validation set, we used our subset of predictors to create a logistic model and estimated
a misclassification rate of around 16%, which was pretty high. As a result, our next step
was to clean the data which contained a lot of NA’s and misclassified values.
### Data Cleaning
Before performing any further data analysis, we wanted to thoroughly clean our data
by dealing with missing observations and bringing more value to existing columns - by
creating new columns, removing redundant predictors and reducing the levels in a
categorical variable. In order to keep changes and transformations that we make consistent
between both the training and testing set, we combined the two datasets and created a
column _affordabilitty_ with placeholder values of NA for all the observations from the
testing set.

Immediately, we recognized that _MSSubClass_ was incorrectly imported as a numeric
variable, since they are encodings for the different types of dwellings. Therefore, we
changed it to the proper type from integer to factor.

Our first goal was to deal with the missing values in the data set, so we obtained the
overall number of missing observations in each feature (column). For any predictors that
had more than 60% of the data missing like _Utilities_, _MiscFeature_, and _MiscVal_, we dropped
the variables. For the remaining predictors, we want to identify why the observations were
missing so that we can place it in one of two cases: missing values due misclassification or
truly missing values. Our solutions for the two cases respectively were to relabel the
predictors levels or imputation.

In order to identify the misclassified data, we cross-referenced all of the categorical
predictors with missing observations with the data description sheet to see which
predictors should be relabeled as “None”. The missingness signified the lack of a feature, as
opposed to missing information. Therefore, we created a new factor level “None” for the
following 8 predictors: _Alley_, _BsmtQual_, _BsmtCond_, _BsmtExposure_, _BsmtFinType1_,
_BsmtFinType2_, _FireplaceQu_, and _Fence_.

According to the data description, if _GarageType_, _GarageFinish_, _GarageQual_,
_GarageCond_, or _GarageYrBlt_ have an NA input, then it indicates that the property does not
have a garage. (Although the data description did not specify this for _GarageYrBlt_, we found
this to be true since the missing rows for _GarageYrBlt_ were the same as _GarageFinish_,
_GarageQual_, and _GarageCond_.) For this reason, we expected the 5 predictors to have the
same number of missing values. However, _GarageType_ only had 252 missing values and the
other predictors had 254 missing values. We found that there were 2 observations that
were categorized as having a “Detchd” garage, even though the other corresponding 4
garage predictors indicated that a garage did not exist on the property. Therefore, for these
two unique cases, we relabelled the _GarageType_ from “Detchd” to NA. Since the number of
missing values is now consistent between the predictors, we created a new factor level
“None” for the missing values for _GarageType_, _GarageFinish_, _GarageQual_, and _GarageCond_.
For the _GarageYrBlt_, we simply assigned the missing values to 2018 as a placeholder value
to indicate the lack of garage on the property. We noted that one observation had a
_GarageYrBlt_ value of 2207, which is impossible since it is currently 2018. (Figure 1A) Since
other garage features for this observation indicated that a garage does exist on the
property, we simply set the _GarageYrBlt_ value equivalent to the year the home was built.
(Figure 1B)

According to the data description, if _PoolQC_ has a missing value, then it is equivalent to
having no pool. We expected all properties with missing values for _PoolQC_ would have 0 as
their _PoolArea_. However, upon further inspection, we found that there were 6 cases where
a property had a missing value for _PoolQC_, but a numeric value for the _PoolArea_. (Figure
2A) For these unique cases, we looked at the median _PoolArea_ for each type of _PoolQC_.
Based on the observations’ proximity to the median for the corresponding classes, we
assigned the missing observations a _PoolQC_. After dealing with the 6 outlier cases, the
remaining properties with missing data for _PoolQC_ had a corresponding _PoolArea_ equal to
0. We promptly relabeled these observations’ _PoolQC_ value to “None”. (Figure 2B)
For _MasVnrType_, there is already a designated “None” level for the factor. For any
missing values, we designated it to “None”. Similarly for any missing values for
_MasVnrArea_, we set it equal to 0 in order to signify that it does not have a masonry veneer.
We looked at our boxplots to confirm that all MasVnrTypes with “None” had a _MasVnrArea_
with 0. (Figure 3A) However, we identified 16 cases where a property had a number for a
_MasVnrArea_, but was labeled as having no _MasVnrType_. In 2 of these cases, the _MasVnrArea_
was equivalent to 1. Since we know that it is impossible to have a masonry veneer area of 1
square foot, we set them to 0. For the remaining 14 properties that had a _MasVnrArea_, but
was classified as having no masonry veneer, we relabelled them to the “Other” category for
the _MasVnrType_. (Figure 3B)

It was easier to identify the categorical predictors that were improperly labeled
because it was explicitly stated on the data description sheet. For the numerical predictors,
we had to investigate whether or not NA indicated a lack of a feature by cross-referencing
them with a similar and related predictor and inferring their value. For _BsmtFinSF1_,
_BsmtFinSF2_, _BsmtUnfSF_, and _TotalBsmtSF_ we discovered that the same two observations
were missing across these predictors. The corresponding _BsmtQual_ for these two
observations indicated the property didn’t have a basement. Therefore, we set the NA’s for
the 4 predictors to 0. Similarly, for _BmtFullBath_ and _BsmtHalfBath_, we found that the same
4 observations had missing values and their corresponding _BsmtQual_ denoted that a
basement didn’t exist. So we set the NA’s into 0. Additionally, _GarageCars_ and _GarageArea_
have the same two observations missing from their column and the corresponding
_GarageQual_ indicated that a garage doesn’t exist on the property. Therefore, we set the 2
NA’s for those 2 predictors to 0.

The remaining predictors _MSZoning_, _LotFrontage_, _Electrical_, _KitchenQual_, _Functional_,
_Exterior1st_, _Exterior2nd_, and _SaleType_, fall into our second case and meant that the missing
observations were truly missing and did not represent a lack of a feature. For our
numerical variable _LotFrontage_, our initial approach was to impute with the median value
of the predictor. For the categorical variables, our initial approach was to randomly sample
from the levels of the factor to fill the missing values. However, as we were conducting our
exploratory analysis, we noticed that the majority of our observations fell into one level of
that factor. (Figure 4) Therefore, for predictors like _Electrical_, _KitchenQual_, _Functional_,
_Exterior1st_, _Exterior2nd_, and _SaleType_ that had less than 5 observations missing, we simply
assigned the missing value to the mode of that predictor. Our imputation process to replace
missing observations with the mode introduces bias into our data. For predictors with 5 or
more observations missing like _MSZoning_ and _LotFrontage_, we developed a prediction
model based on the existing observations. For _MSZoning_, we first created a new dataframe
consisting of columns without missing observations and attached the _MSZoning_ column on
this dataframe. We then split this dataframe into rows that contained no NA’s and rows
that did. With the data that did not contain any NA’s we split into 90% training and 10%
testing. We then created a random forest model using all the predictors and the train data
to predict the testing data and we got an accuracy of 96% which was significantly high. As a
result, we created a Random Forest model using all the predictors and the data that did not
contain any NA’s to predict the NA values of _MSZoning_. Similarly, we used the same
methodology to predict the NA’s in _LotFrontage_, but instead of using Random Forest, we
used Generalized Boosted Regression Models with our distribution set to Gaussian. We
used Generalized Boosted Regression Model due to getting a low MSE after predicting the
testing data using the training data.

After handling the missing data in the data set, we conducted feature engineering in
order to extract more value from the existing predictors. We created a new categorical
variable _Remodeled_ that described if a house has been remodeled and a new numerical
variable _HouseAge_ that calculated how old the house based on the difference between the
current year and the year the house was built. If the house was remodeled, then we
calculated the difference between the current year and the year it was remodeled. In
addition, we combined _BsmtFinSF1_ and _BsmtFinSF2_ in order to calculate the _BsmtTotalFin_,
which is a numerical variable for the total square feet of the finished basement area. Since
_YearBuilt_, _YearRemodAdd_, _BsmtFinSF1_, and _BsmtFinSF2_ became redundant features, we
dropped them from our dataset. For _LotShape_, rather than distinguishing between the
different types of irregular property shapes, we simplified and condensed our levels to Reg
vs. IR (irregular).

After cleaning our data, we perform some exploratory analysis. From our correlation
matrix, we see three distinct highly correlated pairs of variable, which are _GarageCars_ and
_GarageArea_, _GrLivArea_ and _TotRmsAbv_, _X1stFlrSF_ and _TotalBsmtSF_. (Figure 5) Based on the
density plots, we select the predictors (_GarageArea_, _GrLivArea_, and _X1stFlrSF_) that have the
least amount of overlap in classes for _affordabilitty_. We drop these highly correlated
variables, as well as redundant variables like _Exterior2nd_ from our dataset. To further
minimize the dimensionality of our data, we conduct a principal component analysis on the
numeric variables. In order to retain at least 95% of the original data, we need to include
23 principal components into our model. (Figure 7) We then combine our principal
components dataframe with our categorical variables and then proceed to split our data
into the original training and testing data.
### Modeling
We further segment our training data into a training (90%) and validation (10%) set
so that we can estimate our misclassification rate for different models. Our first attempt
was to run a full logistic regression model, and we received an accuracy rate of 92%. We
can see that data cleaning was successful as our accuracy rose by nearly 10%.
Our next approach was to use the full tree model. The full tree model gave us an
accuracy of 96.2%, which was higher than the logistic model. In addition, we used cross
validation to find the best tree to use for pruning. Although we pruned the tree, we ended
up with the same results. We knew that using trees resulted in a lot of correlated branches,
so our next approach was bagging.
Setting mtry to 64 and ntree=900 based on the model plot (Figure 8) lead us to a
accuracy of 97%. Based on this bagging model, we created a Variance Importance Plot
(figure 10) that lead us to create random forest models that consisted of the ten most
important predictors and the thirteen most important predictors. Comparing these two
models we could see that the smaller model had better results which was 98%. We
hypothesized that smaller models did better in predicting as it led to less of an overfit. As a
result, we tested this hypothesis with models with different amount predictors and found
that models with more than 13 and less than 7 predictors gave us less accuracy than that of
ten predictors.

To further improve our results, we ran the model with the ten most important
predictors multiple times and saved the resulting _affordabilitty_ responses in a dataframe.
We then used majority vote in order to predict the final testing _affordabilitty_ response and
ended up with an accuracy of 98%. Since our accuracy stayed consistent across the
repeated trials, we know that the model success can be attributed to the our reduced model
of top 10 predictors rather than the random sampling of a subset of predictors at each split
of the tree.
## Main Results
Our first attempt at prediction using the clean data set was through Logistic
Regression. We used full logistic model and received an accuracy of 96% which was a
drastic improvement from the full logistic model using the uncleaned dataset. We then
used forward and backward selection to decrease the number of predictors we used but
ended up with getting a lower accuracy. As a result, we had concluded that the full logistic
regression model would give us the best result.

In addition to Logistic Regression, we used the full tree method. We found that the
full tree model did best with mindev = 0 and minsize = 2. We got an accuracy of 96.2% with
the full tree model. Also, we looked into pruning the tree, but found that the accuracy
decreased so we decided the full tree model would be the best fit. Since the Tree model is
known to be really correlated, we decided to then use Bagging method.
With Bagging, we achieved an accuracy of 97.4% which was an improvement from
Logistic Regression and the full tree model. Since Bagging did well, we used Variance
Importance Plot (Figure 10) to find the most important predictors to use in Random Forest
models.

With the random forest model, we were able to tune different aspects of our model like
mtry, ntree, strata, nodesize, and maxnodes. For our main result which got us 97.98%
accuracy on Kaggle, we used a random forest model with the ten most important predictors
(_Principle Component 1_, _Principle Component 3_, _Principle Component 4_, _Principle Component
5_, _Principle Component 9_, _Principle Component 12_, _Neighborhood_, _Principle Component 16_,
_Principle Component 19_, and _Principle Component 20_, figure 10) that we ran ten times and
found the most common result. We set mtry to each iteration so that it went from one to
ten and set ntree to 1100 as the plot began to stabilize around 1100 (figure 9). Previous to
using this model on the actual testing dataset, we had tested it on the training data set and
ended with 98% accuracy. We can see that through our result from Kaggle and this result,
that there was a difference of 0.02% in accuracy which demonstrated that our model did
not overfit.
## Limitations
By applying principal component analysis on all the numerical variables prior to any
data modeling, we heavily sacrificed the interpretability of our final results. Since each
principal component is a linear combinations of all of the numeric predictors, we were not
able to properly identify which numeric predictors would be an important feature in
predicting the affordability of houses. In addition, with random forest there is tree
correlation and it is easy to overfit to the training data. There is usually high variance
without much regularization, and we combatted this by performing cross validation.
## Recommendations
Our approach placed a heavy focus on data cleaning, particularly with handling missing
observations. Although we employed a bit of feature engineering with the creation of new
variables _Remod_, _HouseAge_, and _BsmtTotalFin_, further exploration of new potential
predictors would have been highly beneficial. For example, creating variables like
_TotalBath_, _TotalSF_, _Conditions_, _BsmtFinTypes_, _IndoorPatioSF_ would enable us to condense
and drop predictors _BsmtFullBath_, _BsmtHalfBath_, _FullBath_, _HalfBath_, _GrLivArea_,
_TotalBsmtSF_, _Condition1_, _Condition2_, _BsmtFinType1_, _BsmtFinType2_, _EnclosedPorch_,
_X3SsnPorch_, and _ScreenPorch_. Reducing the number of redundant predictors in the data
eliminates collinear variables and reduces multicollinearity between predictors. We also
recommend others to reduce the levels for categorical variables for more than 10 levels, in
particular _MSSubClass_ and _Neighborhood_. Collapsing the levels in a categorical variable and
eliminating redundant variables can successfully reduce the high dimensionality of the data
set.

Since the majority of our time was focused on data cleaning, we were hesitant to drop
much of the variables from our model. We recommend that moving forward, others build
models from a smaller subset of really important predictors in order to further reduce the
dimensionality and complexity of the model.
## Conclusions
We realized that data cleaning was very important after multiple, unsuccessful attempts at
predicting the testing data using the training dataset that had all its rows with NA’s
removed. Through data cleaning we saw a large, significant increase in accuracy just
through the Logistic Regression Model alone. As a result, we put much of our effort and
time in cleaning and condensing the training dataset whether it be combining variables
through PCA to removing highly correlated variables. However, due to spending a lot of our
time and effort on cleaning the data, we were not able to fully explore the different
methods we had used whether it be altering the mtry’s to changing ntrees multiple times.
Since we had done a lot of data cleaning, we believe that we could have gotten better
results if we had more time to effectively explore and use random forest. In addition, we
could not explore the new methods that we had learned like xgboost and neural network.
We would have experimented with these news methods and compared the results with our
current best, which is random forest. Also, If these three methods had significant results,
we would have ran all three methods multiple times and used majority vote to find the best
test prediction.
