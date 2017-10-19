# run this only once
if(exists("configs") == F){
	library(deepnet)
	
	#Creating list to store the differents outputs and information
	configs = data.frame(hidden_rbm=integer(), numempochs_rbm=integer(), batchsize_rbm=integer(), lr_rbm=numeric(), cd=integer(), hidden_nn=character(), lr_nn=numeric(), numepochs_nn=integer(), batchsize_nn=numeric(), onehot=integer(), err_score=numeric(), err_score_norm=numeric())
	predict_list <- list()
	predict_norm_list <- list()
	nn_list <- list()
}

#in order to work with the labels coded as onehot vector or not, leave it 1
onehot <- 1

#Hyperparameters for the rbm
hidden_rbm <- 100
numepochs_rbm <- 10
batchsize_rbm <- 100
learningrate_rbm <- 0.1
learningrate_scale_rbm <- 1
cd <- 3

#Hyperparameters for the neural net
hidden_nn <- c(100)
learningrate_nn <- 0.1
learningrate_scale_nn <- 1
numepochs_nn <- 10
batchsize_nn <- 100

#Loading the files
X_train <- read.csv(file="../csv/X_train_AAL.csv", header=T, sep=",", row.names=1)
X_test <- read.csv(file="../csv/X_test_AAL.csv", header=T, sep=",", row.names=1)
y_train_org <- read.csv(file="../csv/y_train_AAL.csv", header=T, sep=",", row.names=1)
y_test_org <- read.csv(file="../csv/y_test_AAL.csv", header=T, sep=",", row.names=1)

#Removing the first row because it has no return
X_train <- X_train[2:nrow(X_train),]
X_test <- X_test[2:nrow(X_test),]

#Onehot vector encoding
onehot_test <- matrix(0L, nrow=dim(y_test_org)[1], ncol=max(y_test_org)+1)
counter <- 1
for(y in 1:dim(y_test_org)[1]){
	onehot_test[counter,y_test_org[y,]+1] <- 1
	counter <- counter+1
}
onehot_train <- matrix(0L, nrow=dim(y_train_org)[1], ncol=max(y_train_org)+1)
counter <- 1
for(y in 1:dim(y_train_org)[1]){
	onehot_train[counter, y_train_org[y,]+1] <- 1
	counter <- counter+1
}

#Coverting dataframes to matrices
X_train <- as.matrix(X_train)
X_test <- as.matrix(X_test)
y_train <- as.matrix(y_train_org)
y_test <- as.matrix(y_test_org)

#Asigning onehot vecto labels to the variables used in the model
if(onehot){
	y_test <- onehot_test
	y_train <- onehot_train
}

#Training the rbm
rbm <- rbm.train(x=X_train, hidden=hidden_rbm, numepochs = numepochs_rbm, batchsize = batchsize_rbm, learningrate = learningrate_rbm, learningrate_scale = learningrate_scale_rbm, momentum = 0.5, visible_type = "bin", hidden_type = "bin", cd = cd)

#Transforming input values
transformed_train <- rbm.up(rbm, X_train)
transformed_test <- rbm.up(rbm, X_test)

#Training the neural net
nn = nn.train(x=transformed_train, y_train, initW = NULL, initB = NULL, hidden = hidden_nn, activationfun = "sigm", learningrate = learningrate_nn, momentum = 0.5, learningrate_scale = learningrate_scale_nn, output = "sigm", numepochs = numepochs_nn, batchsize = batchsize_nn, hidden_dropout = 0, visible_dropout = 0)


#Calculating the score normalized
score <- 0
score <- nn.test(nn, transformed_test, y_test, t = 0.5)

pred <- nn.predict(nn, transformed_test)
pred_norm <- matrix(0L, nrow=dim(pred)[1], ncol=dim(pred)[2])
for(i in 1:dim(pred)[1]){
	max_row <- which.max(pred[i,])
	pred_norm[i,max_row] <- 1
}
score_norm <- 0
for(i in 1:dim(pred_norm)[1]){
	if(which.max(pred_norm[i,]) == which.max(y_test[i,])){
		score_norm <- score_norm +1
	}
}
score_norm <- 1- score_norm/dim(pred)[1]

#Saving data into lists
configs <- rbind(configs,data.frame(hidden_rbm=hidden_rbm, numempochs_rbm = numepochs_rbm, batchsize_rbm=batchsize_rbm, lr_rbm=learningrate_rbm, cd=cd, hidden_nn=paste(hidden_nn,collapse=" "), lr_nn=learningrate_nn, numepochs_nn=numepochs_nn, batchsize_nn=batchsize_nn, onehot=onehot, err_score=score, err_score_norm = score_norm))
predict_list[[dim(configs)[1]]] <- pred
predict_norm_list[[dim(configs)[1]]] <- pred_norm
nn_list[[dim(configs)[1]]] <- nn

# Histogram of y_train
sort(table(y_train_org),decreasing=TRUE)

#Printing information
configs