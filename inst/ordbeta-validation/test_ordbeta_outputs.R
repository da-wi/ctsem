# Test C: filtered / smoothed predictions on the observation scale, and data generation from a fitted model.
# Also a mixed model (ordered beta + gaussian indicator) to exercise the combined update path.
args <- commandArgs(trailingOnly = TRUE)
lib <- args[1]; rec <- readRDS(args[2])
suppressPackageStartupMessages(library(ctsem, lib.loc = lib))
fit <- rec$fit; sim <- rec$sim
fit$stanmodel <- ctsem:::stanmodels$ctsm # reattach compiled model after readRDS

k <- ctStanKalman(fit, subjects = 1:3)
first <- function(a){ a <- drop(a); if(is.matrix(a)) a[, 1] else a }
yp <- first(k$yprior); ys <- first(k$ysmooth); yobs <- first(k$y)
cat('dims yprior', dim(k$yprior), ' rows', length(yobs), '\n')
cat('prior predictions within [0,1]:', all(yp >= 0 & yp <= 1, na.rm = TRUE),
  ' smoothed within [0,1]:', all(ys >= 0 & ys <= 1, na.rm = TRUE), '\n')
cat('mean observed', round(mean(yobs, na.rm = TRUE), 3), ' mean one-step prediction', round(mean(yp, na.rm = TRUE), 3),
  ' cor(observed, smoothed)', round(cor(c(yobs), c(ys), use = 'complete'), 3), '\n')

# data generation through the generated quantities model (ordbeta_rng path)
g <- ctStanGenerateFromFit(fit, nsamples = 20, fullposterior = FALSE, cores = 1)
yg <- g$generated$Y[, , 1]
cat('generated: exact zeros', round(mean(yg == 0, na.rm = TRUE), 3), 'observed', round(mean(sim$NA_vas == 0), 3),
  '| exact ones', round(mean(yg == 1, na.rm = TRUE), 3), 'observed', round(mean(sim$NA_vas == 1), 3),
  '| mean', round(mean(yg, na.rm = TRUE), 3), 'observed', round(mean(sim$NA_vas), 3), '\n')

# mixed: ordered beta negative affect + gaussian positive affect sharing dynamics
set.seed(11)
sim2 <- sim
lat <- qlogis(pmin(pmax(sim$NA_vas, .01), .99))
sim2$PA <- -0.5 * lat + rnorm(nrow(sim), 0, 1)
m2 <- ctModel(type = 'ct', manifestNames = c('NA_vas', 'PA'), latentNames = c('na', 'pa'), manifesttype = c(2, 0),
  LAMBDA = diag(2), MANIFESTMEANS = 0, CINT = 'auto', silent = TRUE)
m2$pars$indvarying <- FALSE
f2 <- ctStanFit(sim2, m2, cores = 1, verbose = 0)
s2 <- summary(f2, digits = 4)
cat('mixed model fitted, loglik', round(s2$loglik, 2), '\n')
print(s2$popmeans[grepl('drift|ordbeta|mvar', rownames(s2$popmeans)), 1:2])
