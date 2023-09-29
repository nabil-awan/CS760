############################################################
### Q-4
############################################################

ff <- function(x.input, y.input){
  library(polynom)
  return(as.function(poly.calc(x.input, y.input)))
}

sigma <- seq(0, 100, by=1)
log.train.error <- NULL 
log.test.error <- NULL

# test set
set.seed(123)
x.test <- runif(100, min = 100, max = 200)
y.test <- sin(x.test)

for(i in 1:length(sigma)){
  set.seed(i)
  # training set
  x.train <- runif(100, min = 100, max = 200) + rnorm(100, mean = 0, sd = sigma[i]) 
  y.train <- sin(x.train)
  log.train.error <- c(log.train.error, log(mean(y.train-ff(x.train, y.train)(x.train))^2))
  log.test.error <- c(log.test.error, log(mean(y.test-ff(x.train, y.train)(x.test))^2))
}

dat.err <- data.frame(sigma=sigma, log.train.error=log.train.error, log.test.error=log.test.error)

library(ggplot2)
ggplot(dat.err, aes(x=sigma, y=log.train.error)) +
  geom_line(size=1, col='blue') +
  geom_line(aes(x=sigma, y=log.test.error), size=1, col='red') + 
  ylab("Log MSE")
