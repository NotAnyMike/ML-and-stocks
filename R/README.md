# Using R

## RBM + ANN in `R` using `deepnet` package

These are the values of the different configurations tested here and the score of the prediction (the error in prediction):

![configs][configs]

The main problem here is that the model predicts (in the best case) the mode, the same result is obtained using the darch library and with the same `A` stock. The documentation of the package can be found [here](https://cran.r-project.org/web/packages/deepnet/deepnet.pdf), see the section on `rbm.train` the attribute `visible_type`, they talk about the sigmoid function when they should be talking about the type of the variables, which is set `binary` by default, and no other value works here.

Changing the stock used in the last results, the clasification errors are the following:

![deepnet and AAL stock][deepnet_AAL]

## Shallow Neural net with a weight initialization given by a RBM using `deepnet`

### Important
The input for this model is different of the rest. Instead of trying to use the average price (a continue value) as inputs (with some lags) here the inputs are the color of the candlestick, so the codifications were the following:
* `00` there was no change
* `01` the candlestick was green (close $>$ open)
* `10` the candlestick was red (close $< $ open)

this inputs contained some numeber of lags

### Results
Running a RBM and then using the weights learned from that generative model to initilize the weights of a Shallow Neural Net the results are the following

![binary_deepnet][binary_deepnet]

## Deep Belief Nets using `darch` package

These are the different hyperparameters used and the classification error:

![configs_darch][configs_darch]

You can see [here](https://cran.r-project.org/web/packages/darch/darch.pdf) their documentation, and [here](https://github.com/maddin79/darch/blob/master/examples/example.mnist.R) they clearly use the library to model continous input. The problem with this library is that accessing only the rmb class is not possible (at least I coudln't find a way to do it)

Using the input from the `A` stock have the same behaviour of the model using the `deepnet` library (i.e. always predicts the mode) but it allows a more complex predictions using a different stock (the `AAL` stock), changing the stock makes the prediction depend on the value and not only predicts the mode, using the stock `AAL` the results are the following

![configs using another stock][configs_darch_AAL_1]

Using the ReLU activation function does not improve significantly the model

![using relu][relu]

[configs_darch]: img/2.png "different hyperparameters for dbn using darch library"
[configs]: img/1.png "different configs"
[configs_darch_AAL_1]: img/AAL_4_darch_tanh_pred_compl_20lags.png "Config using AAL stock"
[relu]: img/AAL_darch_relu_pred-comp.png "Using relu activation function for the first layer"
[deepnet_AAL]: img/AAL_3_deepnet_20lags.png "Using a different stock and the package deepnet"
[binary_deepnet]: img/deepnet_rbm_W_init.png "Using rbm as a initialization parameter to a shallow neuralnet"
