library(deepnet)

hidden_rbm <- 1
numepochs <- 3
batchsize <- 100
learningrate_rbm <- 0.8
learningrate_scale_rbm <- 1
cd <- 1

hidden_rbm <- c(10)
learningrate_nn <- 0.001
learningrate_scale_nn <- 1
numepochs_nn <- 3
batchsize_nn <- 100


X_train <- read.csv(file="../X_train.csv", header=T, sep=",", row.names=1)
X_test <- read.csv(file="../X_test.csv", header=T, sep=",", row.names=1)
y_train <- read.csv(file="../y_train.csv", header=T, sep=",", row.names=1)
y_test <- read.csv(file="../y_test.csv", header=T, sep=",", row.names=1)

X_train <- X_train[1:939,]
X_test <- X_test[1:313,]

X_train <- as.matrix(X_train)
X_test <- as.matrix(X_test)
y_train <- as.matrix(y_train)
y_test <- as.matrix(y_test)

rbm <- rbm.train(x=X_train, hidden=1, numepochs = 3, batchsize = 100, learningrate = 0.8, learningrate_scale = 1, momentum = 0.5, visible_type = "bin", hidden_type = "bin", cd = 1)

transformed_train <- rbm.up(rbm, X_train)
transformed_test <- rbm.up(rbm, X_test)

nn = nn.train(x=transformed_train, y_train, initW = NULL, initB = NULL, hidden = c(10,10), activationfun = "sigm", learningrate = 0.001, momentum = 0.5, learningrate_scale = 1, output = "sigm", numepochs = 3, batchsize = 100, hidden_dropout = 0, visible_dropout = 0)

nn.test(nn, transformed_test, y_test, t = 0.5)