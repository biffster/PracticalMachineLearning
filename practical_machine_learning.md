Practical Machine Learning  
June 7, 2016  

# Predictors
## Components of a predictor:
- Question - what are you trying to predict?
- Input data - As much data as available
  - Easiest to predict if you already have the data you are trying to predict on.
  - Garbage in, garbage out
- Features
  - Leads to data compression
  - Retain relevant information
  - Based on expert application knowledge
  - Common mistakes:
    - Trying to automate feature selection without explaining/understanding what is happening
    - Not paying attention to data-specific quirks
    - Throwing away information unnecessarily
- Algorithm - random forest, decision trees
  - Interpretable
  - Simple
  - Accurate
  - Fast to train and test
  - Scalable
- Parameters - estimate parameters to apply to algorithm
- Evaluation - test algorithm and parameters against test data

## Relative order of importance
1. Question
2. Data
3. Features
4. Algorithms

# Errors
## In-sample error
- Error rate on the same data set you used to build predictor
- Resubstitution Error

## Out of sample error
- Error rate on new data set
- Sometimes called Resubstitution Error

## Overfitting
- In-sample errors are always less than out of sample
- Matching algorithm to the data we have
- Data contains
  - Signal
  - Noise
- Goal of predictor is to find the signal and ignore noise
- Don't tailor predictor too strongly to sample data

## Basic terms:

- Positive = identified
- Negative = rejected
- Sensitivity
  - Positive test and has the disease
- Specificity
  - Negative test and don't have the disease
- Mean squared error
- Root mean squared error
- Median absolute deviation


# Prediction study design
## Sample steps
1. Define error rate
2. Split data into:
  - Training
  - Testing
  - Validation
3. Pick features on the training set using cross-validation
4. Pick the prediction function
5. If there's no validation set, apply a model only 1 time.
6. If we do have a validation set:
  - Apply prediction models to test set
  - Redefine
  - Apply only once

## Misc:
- **Know the benchmarks**

# Size of sample sizes
-  Avoid small sample sizes
-  For a large sample size:
  - 60% Training
  - 20% Test
  - 20% Validation
- For a medium sample size:
  - 60% Training
  - 40% Testing
- For a small sample size:
  - Do cross Validation
  - Give caveat that this was never tried outside of sample set  

# Principles to remember:
- Set test/validation data aside and __don't__ look at it
- Generate randomly sampled training and test data
- Data sets must reflect structure of the problem
  - Back testing, split train/test data into time chunks
- All subsets should reflect as much diversity as possible
  - Can also try to balance by features, but that's tricky

# ROC Curves
## Why a curve?
- Binary classiciations you are predicting has one of two categories
- Predictions are often quantitative instead of binary
- Cutoff you choose gives different results

## Area under the curve
- AUC = 0.5 is the same as guessing
- AUC = 1 is the perfect classifier
- AUC >= 0.8 is considered "good enough"

# Cross validation
## Method
- Use the training set
- Split this into training/test sets
- Build a model on this new training set
- Evaluate on the test set
- Repeated and average estimated errors

## Used for:
- Picking variables to include in a model
- Picking the type of predicition function to use
- Picking parameters in the prediction function
- Comparing different predictors

## Types of cross-validation methodologies:
- Random subsampling
  - __without replacement__
- K-fold cross validation
  - Subset data in equal blocks
  - Larger k = low bias, more variance
  - Smaller k = more bias, less variance
- Leave one out cross validation
- For time series data, data must be used in "chunks"

# Preprocessing
- When creating a standardization in the training set, the smoother you choose  (e.g. mean(train)) are based off the training set, __not the test set__ when you actually run the data through.
- Training and test msut be processed the exact same way
- Test transformations will likely be imperfect
- Be wary of transforming factor variables
- Do all pre-processing with caret

# Covariates
- New Covariates
  - Only done on the training set
  - Best approach is through exploratory analysis
  - new Covariates should be *added* to data frames

# Predicting with trees
- Iteratively split variables into groups
- Evaluate homogeneity within each group
  - If necessary, split again

## Pros
- Easy to interpret
- Better performance in non-linear settings

## Cons
- Without pruning/cross validation can lead to Overfitting
- Harder to estimate uncertainty
- Results may be variable

## Basic algorithm
1. Start with all variables in one group
2. Find variable/split that best separates the outcomes
3. Divide the data into two groups (leaves) on that split (node)
4. Within each split, find the best variable/split that separates the outcomes
5. Continue until the groups are too small or sufficiently "pure"

## Measures of impurity
- Numeric value between 0 and 1
- 0 = perfect purity
- 0.5 = no purity

## Notes
- Classiciation trees are non-linear models
  - They use interactions between variables
  - Data transformations may be less important (monotone transformations)
  - Trees can also be used for regression problems (continous outcome)
- Multiple tree building options in R

# Boot-strap aggregating (bagging)
- Resample cases and recalculate predictions
- Average or majority vote
- Similar bias
- Reduced variance
- Most useful for non-linear functions

## Bagging in caret
- bagEarth
- treebag
- bagFDA

## Notes
- Most useful for nonlinear models
- Often used with trees as an extension in random forests
- Several models use bagging in caret's *train* function

# Random forests
1. Bootstrap samples
2. At each split, bootstrap variables
3. Grow multiple trees and then vote

## Pros and cons

- Pros
  - Accuracy
- Cons
  - Slow
  - Hard to interpret
  - Subject to overfitting

## Notes
- Usually one of the two top performing algorithms along with boosting
- Difficult to interpret because of multiple trees but normaly very accurate
- Care should be taken to avoid overfitting
  - rfcv function

# Boosting
1. Take lots of possibly weak predictors
2. Weight them and add them
3. Equals a stronger predictor

## Basics
- Start with a set of classifiers
  - e.g. All possible trees, all possible regression models
- Create a classifier that combines classification functions
- Goal is to minimize error on training set

## Boosting in R
- Boosting can be used with any subset of classifiers
- One large subclass is *gradient boosting*
- R has multiple boosting libraries. Differences include the choice of basic classification functions and combination rules
  - gbm = boosting with trees
  - mboost = model-based boosting
  - ada = statistical boosting based on *additive logistic regression*
  - gamBoost = boosting generalized additive models

# Model-based prediction

## Basic idea
1. Assume the data follows a probablistic model
2. Use Baye's theorem to identify optimal classifiers

### Pros
- Can take advantage of structure of the data
- May be computationally convenient
- Are reasonably accurate on real problems

### Cons
- You have to make additional assumptions about data
- When the model is incorrect, you get reduced accuracy

## Classifying using the model
- Linear discriminant analysis assumes multi-variate Gaussian with the same covariances
- Quadratic discrimant analysis assumes multi-variage Gaussian with different covariances
- Model-based prediction assumes more complicated versions for the covariance matrix
- Naive Bayes assumes independence between features (predictors) for model building

# Regularized regression
- Fit a regression model
- Shrink/penalize large coefficients

## Pros
- Delivers a good estimation of Y
- Can help with the bias/variance tradeoff
- Can help with model selection

## Cons
- May be computationally demanding on large data sets
- Deos not perform as well as random forests and boosting

# Model selection approach: split samples
- Errors in test always increase at a certain point with model complexity.

## Approach
1. Divide data into training/test/validation
2. Treat validation as test data
3. train all competing models on the train data,
4. pick the best one on the validation set
5. Assess performance on new data, apply to test set
6. Re-split and re-perform steps 1-3 if necessary

## Common problems
- Not enough data to split as many times as needed
- Computational complexity for running all possible models.

# Combining predictors (ensemble)
- Combine classifiers by averaging/voting
- Combine classifiers improves accuracy
- Reduces interpretability
- Boosting, bagging and random forests are examples.

## Basic intuition: majority vote
- e.g. 5 independent identifiers at 70% accuracy for each equals 83.7% majority vote accuracy
- with 101 identifiers, majority vote = 99.9%

## Approaches
- Bagging, boosting, random forests combining similar classifiers
- Combine different classifiers using model stacking or model ensembling

## Notes:
- Even simple blending can be useful
- Typical model for binary/multiclass data
  - Build an odd number of models
  - Predict with each model
  - Predict the class by majority vote
- This can get dramatically more complicated

# Forecasting
- Typically used for time series data.
- Specific pattern types
  - Trends - long term increase or decrease
  - Seasonal patterns
  - Cycles
- Subsampling into training/test is more complicated
- Similar issues arise in spatial data
  - Dependency between nearby observations
  - Location-specific effects
- Typically goal is to predict one or more observations in the future
- All standard predictions can be used with caution

## Possible problems:
- Spurious correlations
- Beware extrapolating too far into the future
- Beware dependencies over time

# Unsupervised prediction
## Key ideas
- Sometimes you don't know labels for prediction
- To build a predictor
  1. Create clusters
  2. Name clusters
  3. Build predictor for clusters
- Predict clusters in a new dataset
- This is mainly to be used for data exploration
 
