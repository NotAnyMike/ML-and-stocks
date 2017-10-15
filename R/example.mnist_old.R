# Loading library
library(darch)

# MNIST example with pre-training
dataFolder = "data/"
downloadMNIST = T
# Make sure to prove the correct folder if you have already downloaded the
# MNIST data somewhere, or otherwise set downloadMNIST to TRUE
provideMNIST(dataFolder, downloadMNIST)
  
# Load MNIST data
load(paste0(dataFolder, "train.RData")) # trainData, trainLabels
load(paste0(dataFolder, "test.RData")) # testData, testLabels

# Load csv

X_train <- read.csv(file="../X_train.csv", header=T, sep=",", row.names=1)
X_test <- read.csv(file="../X_test.csv", header=T, sep=",", row.names=1)
y_train <- read.csv(file="../y_train.csv", header=T, sep=",", row.names=1)
y_test <- read.csv(file="../y_test.csv", header=T, sep=",", row.names=1)
  
# only take 1000 samples, otherwise training takes increasingly long
chosenRowsTrain <- sample(1:nrow(trainData), size=1000)
trainDataSmall <- trainData[chosenRowsTrain,]
trainLabelsSmall <- trainLabels[chosenRowsTrain,]
  
darch  <- darch(trainDataSmall, trainLabelsSmall,
  rbm.numEpochs = 5,
  rbm.consecutive = F, # each RBM is trained one epoch at a time
  rbm.batchSize = 100,
  rbm.lastLayer = -1, # don't train output layer
  rbm.allData = T, # use bootstrap validation data as well for training
  rbm.errorFunction = rmseError,
  rbm.initialMomentum = .5,
  rbm.finalMomentum = .7,
  rbm.learnRate = .1,
  rbm.learnRateScale = .98,
  rbm.momentumRampLength = .8,
  rbm.numCD = 2,
  rbm.unitFunction = sigmoidUnitRbm,
  rbm.weightDecay = .001,
  layers = c(784,100,10),
  darch.batchSize = 100,
  darch.dither = T,
  darch.initialMomentum = .4,
  darch.finalMomentum = .9,
  darch.momentumRampLength = .75,
  bp.learnRate = 1,
  bp.learnRateScale = .99,
  darch.unitFunction = c(tanhUnit, softmaxUnit),
  bootstrap = T,
  darch.numEpochs = 20,
  gputools = T, # try to use gputools
  gputools.deviceId = 0
)

predictions <- predict(darch, newdata=testData, type="class")
  
labels <- cbind(predictions, testLabels)
numIncorrect <- sum(apply(labels, 1, function(i) { any(i[1:10] != i[11:20]) }))
cat(paste0("Incorrect classifications on test data: ", numIncorrect,
           " (", round(numIncorrect/nrow(testLabels)*100, 2), "%)\n"))
  
 darch
