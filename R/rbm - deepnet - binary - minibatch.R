# run this only once
if(exists("configs") == F){
	library(deepnet)
	
	#Creating list to store the differents outputs and information
	configs = data.frame(hidden_rbm=integer(), numempochs_rbm=integer(), batchsize_rbm=integer(), lr_rbm=numeric(), lr_scale_rbm=numeric(), cd=integer(), hidden_nn=character(), lr_nn=numeric(),lr_scale_nn=numeric(), numepochs_nn=integer(), batchsize_nn=numeric(), using_rbm=logical(), err_score_norm=numeric())
	predict_list <- list()
	predict_norm_list <- list()
	nn_list <- list()
}

#General hyperparameters
OneClassOutput <- T
UseRandomSampling <- F
use_rbm <- T
pseudominibatch <- 100

#Hyperparameters for the rbm
hidden_rbm <- 50
numepochs_rbm <- 20
batchsize_rbm <- 50
learningrate_rbm <- 0.1
learningrate_scale_rbm <- 1
cd <- 10

#Hyperparameters for the neural net
hidden_nn <- c(50, 5)
learningrate_nn <- 0.1
learningrate_scale_nn <- 0.5
numepochs_nn <- 5
batchsize_nn <- 10

#Loading the files
if(OneClassOutput){
	file <- "../csv/A_binary_1-class-output.csv"
}else{
	file <- "../csv/A_binary.csv"
}
df <- read.csv(file=file, header=T, sep=",", row.names=1, colClasses=c("numeric", "character"))

#Creating X
X <- matrix(0L, nrow=dim(df)[1], ncol=nchar(df[1,"All_values"]))
for(n in 1:nrow(df)){
	for(i in 1:nchar(df[1,"All_values"])){
		X[n,i] <- strtoi(substr(df[n, "All_values"],i,i))
	}
}

#Onehot vector encoding
counter <- 1
if(OneClassOutput){
	y <- df$Return_binary
}else{
	y <- matrix(0L, nrow=dim(df)[1], ncol=max(df['Return_to_pred'])+1)
	for(row in 1:dim(df)[1]){
		y[counter, (df[row,'Return_to_pred']+1)] <- 1
		counter <- counter+1
	}
}

#Train test split
smp_size = floor(0.75*nrow(df))

set.seed(123)
train_ind <- sample(seq_len(nrow(df)), size = smp_size)

#Coverting dataframes to matrices
X_train_org <- as.matrix(X[train_ind,])
X_test_org <- as.matrix(X[-train_ind,])
y_train_org <- as.matrix(y[train_ind])
y_test_org <- as.matrix(y[-train_ind])

if(UseRandomSampling == F){
	X_train_org <- as.matrix(X[1:smp_size,])
	X_test_org <- as.matrix(X[(smp_size+1):nrow(X),])
	y_train_org <- as.matrix(y[1:smp_size])
	y_test_org <- as.matrix(y[(smp_size+1):nrow(X)])	
}

#Creating pred matrix
pred <- matrix(0L, nrow=(nrow(X)-pseudominibatch-1), ncol=2)

#Running several nn
#starting <- 1
for(starting in 1:(nrow(X)-pseudominibatch-1)){
	#Getting x and y sets
	X_train <- as.matrix(X[starting:(starting+pseudominibatch),])
	y_train <- as.matrix(y[starting:(starting+pseudominibatch)])
	X_test <- X[(starting+pseudominibatch+1),,drop=FALSE]
	y_test <- y[(starting+pseudominibatch+1),drop=FALSE]
	
	#Training the rbm
	rbm <- rbm.train(x=X_train, hidden=hidden_rbm, numepochs = numepochs_rbm, batchsize = batchsize_rbm, learningrate = learningrate_rbm, learningrate_scale = learningrate_scale_rbm, momentum = 0.5, visible_type = "bin", hidden_type = "bin", cd = cd)

	#Transforming input values
	#transformed_train <- rbm.up(rbm, X_train)
	#transformed_test <- rbm.up(rbm, X_test)

	#Getting initW
	if(use_rbm){
		initW <- rbm$W
	}else{
		initW <- NULL
	}

	#Training the neural net
	nn = nn.train(x=X_train, y=y_train, initW = initW, initB = NULL, hidden = hidden_nn, activationfun = "sigm", learningrate = learningrate_nn, momentum = 0.5, learningrate_scale = learningrate_scale_nn, output = "sigm", numepochs = numepochs_nn, batchsize = batchsize_nn, hidden_dropout = 0, visible_dropout = 0)
	
	pred[starting,1] <- nn.predict(nn, X_test)
	pred[starting,2] <- y_test
}

#Calculating the score normalized
pred_norm <- matrix(0L, nrow=dim(pred)[1], ncol=dim(pred)[2])
pred_norm[,2] <- pred[,2]
for(i in 1:dim(pred)[1]){
	if(pred[i,1] >= 0.5) pred_norm[i,1] <- 1
	else pred_norm[i,1] <- 0
}
score_norm <- 0
for(i in 1:dim(pred_norm)[1]){
	if(pred_norm[i,1] == pred_norm[i,2]){
		score_norm <- score_norm +1
	}
}
score_norm <- 1- score_norm/dim(pred)[1]

#Saving data into lists
configs <- rbind(configs,data.frame(hidden_rbm=hidden_rbm, numempochs_rbm = numepochs_rbm, batchsize_rbm=batchsize_rbm, lr_rbm=learningrate_rbm, lr_scale_rbm=learningrate_scale_rbm, cd=cd, hidden_nn=paste(hidden_nn,collapse=" "), lr_nn=learningrate_nn, lr_scale_nn=learningrate_scale_nn, numepochs_nn=numepochs_nn, batchsize_nn=batchsize_nn, using_rbm=use_rbm, err_score_norm = score_norm))
predict_list[[dim(configs)[1]]] <- pred
predict_norm_list[[dim(configs)[1]]] <- pred_norm
nn_list[[dim(configs)[1]]] <- nn

# Histogram of y_train
if(OneClassOutput){
	sort(table(df[,'Return_binary']),decreasing=TRUE)
	sort(table(y_test),decreasing=TRUE)
	sort(table(y_train),decreasing=TRUE)
}else{
	sort(table(df[,'Return_to_pred']),decreasing=TRUE)
}

#Printing information
configs