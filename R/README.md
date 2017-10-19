# Using R

## RBM + ANN in `R` using `deepnet` package

These are the values of the different configurations tested here and the score of the prediction (the error in prediction):

![configs][configs]

the documentation of the package can be found [here](https://cran.r-project.org/web/packages/deepnet/deepnet.pdf), see the section on `rbm.train` the attribute `visible_type`, they talk about the sigmoid function when they should be talking about the type of the variables, which is set `binary` by default, and no other value works here.

## Deep Belief Nets using `darch` package

These are the different hyperparameters used and the classification error:

![configs_darch][configs_darch]

You can see [here](https://cran.r-project.org/web/packages/darch/darch.pdf) their documentation, and [here](https://github.com/maddin79/darch/blob/master/examples/example.mnist.R) they clearly use the library to model continous input. The problem with this library is that accessing only the rmb class is not possible (at least I coudln't find a way to do it)

[configs_darch]: img/2.png "different hyperparameters for dbn using darch library"
[configs]: img/1.png "different configs"
