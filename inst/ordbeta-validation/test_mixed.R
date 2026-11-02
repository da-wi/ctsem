# Mixed indicators of one latent process: ordered beta VAS + gaussian item, same time points (combined update path)
args <- commandArgs(trailingOnly = TRUE); lib <- args[1]
suppressPackageStartupMessages(library(ctsem, lib.loc = lib))
set.seed(21)
tr <- c(drift = -0.6, cint = -0.3, diffusion = 1.1, k1 = -0.4, k2 = 2.6, phi = 12, lambda2 = 0.8, mm2 = 1, sd2 = 0.6)
sim <- do.call(rbind, lapply(1:80, function(i){
  t <- unlist(lapply(0:5, function(d) sort(d * 24 + 9 + runif(7, 0, 13))))
  x <- numeric(length(t)); a <- tr['drift']; q <- tr['diffusion']; c0 <- tr['cint']
  x[1] <- rnorm(1, -c0 / a, sqrt(q^2 / (-2 * a)))
  for(j in 2:length(t)){ e <- exp(a * (t[j] - t[j - 1]))
    x[j] <- e * x[j - 1] + (e - 1) / a * c0 + rnorm(1, 0, sqrt(q^2 * (1 - e^2) / (-2 * a))) }
  data.frame(id = i, time = t, NA_vas = rordbeta(length(t), x, tr['k1'], tr['k2'], tr['phi']),
    NA_item = tr['mm2'] + tr['lambda2'] * x + rnorm(length(t), 0, tr['sd2']))
}))
sim$NA_item[sample(nrow(sim), 600)] <- NA  # some rows with ordered beta only
m <- ctModel(type = 'ct', manifestNames = c('NA_vas', 'NA_item'), latentNames = 'negaff', manifesttype = c(2, 0),
  LAMBDA = matrix(c(1, 'lambda2'), 2, 1), MANIFESTMEANS = c(0, 'mm2'), CINT = 'cint', DRIFT = 'drift',
  DIFFUSION = 'diffusion', MANIFESTVAR = c('sigma', 0, 0, 'sd2'), T0MEANS = 't0m', T0VAR = 't0var', silent = TRUE)
m$pars$indvarying <- FALSE
f <- ctStanFit(sim, m, cores = 1, verbose = 0)
pm <- summary(f, digits = 6)$popmeans
est <- c(drift = pm['drift', 'mean'], cint = pm['cint', 'mean'], diffusion = pm['diffusion', 'mean'],
  k1 = pm['ordbetacut1_NA_vas', 'mean'], k2 = pm['ordbetacut1_NA_vas', 'mean'] + pm['ordbetacutgap_NA_vas', 'mean'],
  phi = 1 / pm['sigma', 'mean']^2, lambda2 = pm['lambda2', 'mean'], mm2 = pm['mm2', 'mean'], sd2 = pm['sd2', 'mean'])
print(round(rbind(true = tr, est = est), 3))
