library(deepnet)

# run this only once
# configs = data.frame(hidden_rbm=integer(), numempochs_rbm=integer(), batchsize_rbm=integer(), lr_rbm=numeric(), cd=integer(), hidden_nn=integer(), lr_nn=numeric(), numepochs_nn=integer(), batchsize_nn=numeric(), score=numeric())

hidden_rbm <- 1000
numepochs_rbm <- 10
batchsize_rbm <- 90
learningrate_rbm <- 0.01
learningrate_scale_rbm <- 1
cd <- 10

hidden_nn <- c(10,10)
learningrate_nn <- 0.001
learningrate_scale_nn <- 1
numepochs_nn <- 10
batchsize_nn <- 90


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

rbm <- rbm.train(x=X_train, hidden=hidden_rbm, numepochs = numepochs_rbm, batchsize = batchsize_rbm, learningrate = learningrate_rbm, learningrate_scale = learningrate_scale_rbm, momentum = 0.5, visible_type = "bin", hidden_type = "bin", cd = cd)

transformed_train <- rbm.up(rbm, X_train)
transformed_test <- rbm.up(rbm, X_test)

nn = nn.train(x=transformed_train, y_train, initW = NULL, initB = NULL, hidden = hidden_nn, activationfun = "sigm", learningrate = learningrate_nn, momentum = 0.5, learningrate_scale = learningrate_scale_nn, output = "sigm", numepochs = numepochs_nn, batchsize = batchsize_nn, hidden_dropout = 0, visible_dropout = 0)

score <- nn.test(nn, transformed_test, y_test, t = 0.5)
configs = rbind(configs,data.frame(hidden_rbm=hidden_rbm, numempochs_rbm = numepochs_rbm, batchsize_rbm=batchsize_rbm, lr_rbm=learningrate_rbm, cd=cd, hidden_nn=hidden_nn, lr_nn=learningrate_nn, numepochs_nn=numepochs_nn, batchsize_nn=batchsize_nn, score=score))
configs
score

