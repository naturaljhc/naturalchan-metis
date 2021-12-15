After collecting our data, we did some surface data analysis on the numeric variables and noticed a few things:
* The distribution on many variables are non-uniform, in fact, most are heavily skewed.
* Since most distributions of variables were skewed, it is slightly hard to tell if some variables were correlated with each other from just a pair plot.
* All numeric variables seemed correlated with the price per night, but this is still hard to tell without looking deeper. For example, because ratings were heavily skewed towards 5 stars, we notice that no high priced listings were associated with a low rating. However, when we observe listings with a high rating, the price varies heavily so rating might not be too correlated with the price.

We then ran a very simple OLS model and filled all missing values with 0 for now. To get an idea of what variables to keep, we first used statsmodels.api to get some statistics. Overall, the R2 values were surprisingly high - consistently scoring over 0.9 after randomizing our training set multiple times. We found that only one of our numeric variables, ratings, had a P value of over 0.05. This may be due to the ratings not actually being correlated with the price as we observed before.

We tried OLS again with sklearn and found that the R2 was still pretty high using the same randomized training sets with a fairly consistent value of over 0.8. When scoring on the test sets, the R2 fluctuates, never below 0.7 and sometimes quite higher than the R2 on the training set.

When plotting our residuals, we observe strong heteroskedasticity so we may need to make adjustments to our model. Our first step was to take a log transformation of target and upon doing so, we see a slight quadratic relationship between our residuals and our predictions. However, our R2 decreses dramatically to below .6 for both the training and the test set.

Next Steps:
It is quite clear that there are some heavy outliers within our dataset. We will want to observe their importance in our data and if they are erroneous or not. To do so, we may need to observe some other features of those rows. For example, some data points have titles indicating they are erroneous. Some outliers may be real and simply have high prices due to the number of guests/bedrooms/baths listed.
We will also want to start using more complex models by including the other features - to this end, we will need to split some columns into dummy or numerical variables. After we sort out the errors from our base model, we will compare these more complex models and see if they offer a relatively significant improvement to the base model. We may also want to create dummy variables for the date as they are categorical in nature.
