# run this only once
if(exists("configs") == F){
	library(deepnet)
	library(darch)
	
	#Creating list to store the differents outputs and information
	configs = data.frame(hidden_rbm=integer(), numempochs_rbm=integer(), batchsize_rbm=integer(), lr_rbm=numeric(), cd=integer(), hidden_nn=character(), lr_nn=numeric(), numepochs_nn=integer(), batchsize_nn=numeric(), err_score=numeric(), err_score_norm=numeric())
	predict_list <- list()
	predict_norm_list <- list()
	nn_list <- list()
}

#Hyperparameters for the rbm
hidden_rbm <- 100
numepochs_rbm <- 10
batchsize_rbm <- 100
learningrate_rbm <- 0.01
learningrate_scale_rbm <- 1
cd <- 10

#Hyperparameters for the neural net
hidden_nn <- c(200,50)
learningrate_nn <- 0.01
learningrate_scale_nn <- 0.8
numepochs_nn <- 10
batchsize_nn <- 500

#################################
dataFolder = "mnist/"
downloadMNIST = F
# Make sure to prove the correct folder if you have already downloaded the
# MNIST data somewhere, or otherwise set downloadMNIST to TRUE
provideMNIST(dataFolder, downloadMNIST)
  
# Load MNIST data
load(paste0(dataFolder, "train.RData")) # trainData, trainLabels
load(paste0(dataFolder, "test.RData")) # testData, testLabels

# only take 1000 samples, otherwise training takes increasingly long
set.seed(123)
chosenRowsTrain <- sample(1:nrow(trainData), size=2000)
trainDataSmall <- trainData[chosenRowsTrain,]
trainLabelsSmall <- trainLabels[chosenRowsTrain,]
#################################
trainDataSmall_norm <- matrix(0L, nrow=nrow(trainDataSmall), ncol=ncol(trainDataSmall))
for(i in 1:nrow(trainDataSmall)){
	max_row <- which.max(trainDataSmall[i,])
	pred_norm[i,max_row] <- 1
}

#Training the rbm
rbm <- rbm.train(x=trainDataSmall, hidden=hidden_rbm, numepochs = numepochs_rbm, batchsize = batchsize_rbm, learningrate = learningrate_rbm, learningrate_scale = learningrate_scale_rbm, momentum = 0.5, visible_type = "bin", hidden_type = "bin", cd = cd)

#Transforming input values
#transformed_train <- rbm.up(rbm, X_train)
#transformed_test <- rbm.up(rbm, X_test)

#Training the neural net
nn = nn.train(x= trainDataSmall, y= trainLabelsSmall, initW = rbm$W, initB = NULL, hidden = hidden_nn, activationfun = "sigm", learningrate = learningrate_nn, momentum = 0.5, learningrate_scale = learningrate_scale_nn, output = "sigm", numepochs = numepochs_nn, batchsize = batchsize_nn, hidden_dropout = 0, visible_dropout = 0)


#Calculating the score normalized
score <- 0
score <- nn.test(nn, testData, testLabels, t = 0.5)

pred <- nn.predict(nn, testData)
pred_norm <- matrix(0L, nrow=dim(pred)[1], ncol=dim(pred)[2])
for(i in 1:dim(pred)[1]){
	max_row <- which.max(pred[i,])
	pred_norm[i,max_row] <- 1
}
score_norm <- 0
for(i in 1:dim(pred_norm)[1]){
	if(which.max(pred_norm[i,]) == which.max(testLabels[i,])){
		score_norm <- score_norm +1
	}
}
score_norm <- 1- score_norm/dim(pred)[1]

#Saving data into lists
configs <- rbind(configs,data.frame(hidden_rbm=hidden_rbm, numempochs_rbm = numepochs_rbm, batchsize_rbm=batchsize_rbm, lr_rbm=learningrate_rbm, cd=cd, hidden_nn=paste(hidden_nn,collapse=" "), lr_nn=learningrate_nn, numepochs_nn=numepochs_nn, batchsize_nn=batchsize_nn, err_score=score, err_score_norm = score_norm))
predict_list[[dim(configs)[1]]] <- pred
predict_norm_list[[dim(configs)[1]]] <- pred_norm
nn_list[[dim(configs)[1]]] <- nn

# Histogram of y_train
#sort(table(df[,'Return_to_pred']),decreasing=TRUE)

#Printing information
configs