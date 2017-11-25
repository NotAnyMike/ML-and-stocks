# run this only once
if(exists("configs") == F){
	library(deepnet)
	
	#Creating list to store the differents outputs and information
	configs = data.frame(hidden_rbm=integer(), numempochs_rbm=integer(), batchsize_rbm=integer(), lr_rbm=numeric(), cd=integer(), hidden_nn=character(), lr_nn=numeric(), numepochs_nn=integer(), batchsize_nn=numeric(), using_rbm=logical(), err_score=numeric(), err_score_norm=numeric())
	predict_list <- list()
	predict_norm_list <- list()
	nn_list <- list()
}

#General hyperparameters
use_rbm <- T

#Hyperparameters for the rbm
hidden_rbm <- 50
numepochs_rbm <- 200
batchsize_rbm <- 100
learningrate_rbm <- 0.1
learningrate_scale_rbm <- 1
cd <- 10

#Hyperparameters for the neural net
hidden_nn <- c(50,5)
learningrate_nn <- 0.1
learningrate_scale_nn <- 0.5
numepochs_nn <- 5
batchsize_nn <- 200

#Loading the files
df <- read.csv(file="../csv/A_binary.csv", header=T, sep=",", row.names=1, colClasses=c("numeric", "character"))

#Creating X
X <- matrix(0L, nrow=dim(df)[1], ncol=nchar(df[1,"All_values"]))
for(n in 1:nrow(df)){
	for(i in 1:nchar(df[1,"All_values"])){
		X[n,i] <- strtoi(substr(df[n, "All_values"],i,i))
	}
}

#Onehot vector encoding
y <- matrix(0L, nrow=dim(df)[1], ncol=max(df['Return_to_pred'])+1)
counter <- 1
for(row in 1:dim(df)[1]){
	y[counter, (df[row,'Return_to_pred']+1)] <- 1
	counter <- counter+1
}

#Train test split
smp_size = floor(0.75*nrow(df))

set.seed(123)
train_ind <- sample(seq_len(nrow(df)), size = smp_size)

#Coverting dataframes to matrices
X_train <- as.matrix(X[train_ind,])
X_test <- as.matrix(X[-train_ind,])
y_train <- as.matrix(y[train_ind,])
y_test <- as.matrix(y[-train_ind,])

X_train <- as.matrix(X[1:smp_size,])
X_test <- as.matrix(X[(smp_size+1):nrow(X),])
y_train <- as.matrix(y[1:smp_size,])
y_test <- as.matrix(y[(smp_size+1):nrow(X),])

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


#Calculating the score normalized
score <- 0
score <- nn.test(nn, X_test, y_test, t = 0.5)

pred <- nn.predict(nn, X_test)
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
configs <- rbind(configs,data.frame(hidden_rbm=hidden_rbm, numempochs_rbm = numepochs_rbm, batchsize_rbm=batchsize_rbm, lr_rbm=learningrate_rbm, cd=cd, hidden_nn=paste(hidden_nn,collapse=" "), lr_nn=learningrate_nn, numepochs_nn=numepochs_nn, batchsize_nn=batchsize_nn, using_rbm=use_rbm, err_score=score, err_score_norm = score_norm))
predict_list[[dim(configs)[1]]] <- pred
predict_norm_list[[dim(configs)[1]]] <- pred_norm
nn_list[[dim(configs)[1]]] <- nn

# Histogram of y_train
sort(table(df[,'Return_to_pred']),decreasing=TRUE)

#Printing information
configs