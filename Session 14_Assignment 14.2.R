# Use the below given data set
# Problem - prediction of the number of comments in the upcoming 24 hours on
# those blogs, The train data was generated from different base times that may
# temporally overlap. Therefore, if you simply split the train into disjoint partitions,
# the underlying time intervals may overlap. Therefore, the you should use the
# provided, temporally disjoint train and test splits to ensure that the evaluation is
# fair.

# a. Read the dataset and identify the right features
# b. Clean dataset, impute missing values and perform exploratory data analysis.
# c. Visualize the dataset and make inferences from that
# d. Perform any 3 hypothesis tests using columns of your choice, make conclusions
# e. Create a linear regression model to predict the number of comments in the next 24 hours
# (relative to basetime)
# f. Fine tune the model and represent important features
# g. Interpret the summary of the linear model
# h. Report the test accuracy vs. the training accuracy
# i. Interpret the final model coefficients
# j. Plot the model result and compare it with assumptions of the model


library(data.table)
library(MatrixModels)

library(e1071)
library(FNN)
library(glmnet)
library(ranger)
library(xgboost)

# load and combine dataset
train = fread("BlogFeedback-Train.csv")
test = fread("BlogFeedback-Test.csv")

# error measure
mse = function(y_hat, y) {
  mse = mean((y - y_hat)^2)
  
  return(mse)
}

# create design matrices
train_x = model.Matrix(V281 ~ . - 1, data = train, sparse = F)
train_x_sparse = model.Matrix(V281 ~ . - 1, data = train, sparse = T)
train_y = train$V281

test_x = model.Matrix(V281 ~ . - 1, data = test, sparse = F)
test_y = test$V281

train_xgb = xgb.DMatrix(data = as.matrix(train_x), label = train_y)
test_xgb = xgb.DMatrix(data = as.matrix(test_x), label = test_y)

# try kNN
pred_knn = knn.reg(train_x, test_x, train_y, k = 19)$pred
mse(pred_knn, test_y)

# try LASSO
mdl_lasso = cv.glmnet(train_x_sparse, train_y, family = "gaussian", alpha = 1)
pred_lasso = predict(mdl_lasso, newx = test_x)
mse(pred_lasso, test_y)

# try random forest
mdl_rf = ranger(V281 ~ ., data = train, num.trees = 1000, mtry = 120, write.forest = T)
pred_rf = predict(mdl_rf, test)
mse(pred_rf$predictions, test_y)

# try SVM
mdl_svm = svm(V281 ~ V52 + V55 + V61 + V51 + V54 + V21 + V6 + V10, data = train, kernel = "radial", cost = 2, gamma = 0.25)
pred_svm = predict(mdl_svm, test)
mse(pred_svm, test_y)

# try XGboost
mdl_xgb = xgboost(data = train_xgb, nround = 750, nthread = 4, max_depth = 6, eta = 0.025, subsample = 0.7, gamma = 3)
pred_xgb = predict(mdl_xgb, test_xgb)
mse(pred_xgb, test_y)
