# Test B: parameter recovery for the ordered beta measurement model (manifesttype = 2) in a group study.
# Data are simulated independently of ctsem (exact OU discretisation + rordbeta), mimicking ESM
# negative affect on a VAS: strong floor, rare ceiling, irregular beeps, person-specific mean levels.
# Usage: Rscript test_ordbeta_recovery.R <library path> <seed> <out.rds>
args <- commandArgs(trailingOnly = TRUE)
lib <- args[1]; seed <- as.integer(args[2]); out <- args[3]
suppressPackageStartupMessages(library(ctsem, lib.loc = lib))
set.seed(seed)

true <- c(drift = -0.6, cint = -0.3, diffusion = 1.1, cintsd = 0.25,
  k1 = -0.4, k2 = 2.6, phi = 12)
N <- 80; days <- 6; beepsperday <- 7
sim <- do.call(rbind, lapply(1:N, function(i){
  t <- unlist(lapply(0:(days - 1), function(d) sort(d * 24 + 9 + runif(beepsperday, 0, 13))))
  cinti <- rnorm(1, true['cint'], true['cintsd'])
  x <- numeric(length(t))
  a <- true['drift']; q <- true['diffusion']
  x[1] <- rnorm(1, -cinti / a, sqrt(q^2 / (-2 * a)))
  for(j in 2:length(t)){
    dt <- t[j] - t[j - 1]; e <- exp(a * dt)
    x[j] <- e * x[j - 1] + (e - 1) / a * cinti + rnorm(1, 0, sqrt(q^2 * (1 - e^2) / (-2 * a)))
  }
  y <- rordbeta(length(t), x, true['k1'], true['k2'], true['phi'])
  data.frame(id = i, time = t, NA_vas = y)
}))
cat(sprintf('N=%d, obs=%d, exact zeros %.0f%%, exact ones %.0f%%\n', N, nrow(sim),
  100 * mean(sim$NA_vas == 0), 100 * mean(sim$NA_vas == 1)))

m <- ctModel(type = 'ct', manifestNames = 'NA_vas', latentNames = 'negaff', manifesttype = 2,
  LAMBDA = matrix(1), MANIFESTMEANS = 0, CINT = 'cint', DRIFT = 'drift', DIFFUSION = 'diffusion',
  MANIFESTVAR = 'sigma', T0MEANS = 't0m', T0VAR = 't0var', silent = TRUE)
m$pars$indvarying <- FALSE
m$pars$indvarying[m$pars$param %in% 'cint'] <- TRUE

t0 <- Sys.time()
fit <- ctStanFit(sim, m, cores = 1, verbose = 0)
el <- as.numeric(difftime(Sys.time(), t0, units = 'mins'))
s <- summary(fit, digits = 6)
pm <- s$popmeans
get <- function(n) pm[n, 'mean']
est <- c(drift = get('drift'), cint = get('cint'), diffusion = get('diffusion'),
  cintsd = s$popsd['cint', 'mean'],
  k1 = get('ordbetacut1_NA_vas'), k2 = get('ordbetacut1_NA_vas') + get('ordbetacutgap_NA_vas'),
  phi = 1 / get('sigma')^2)
se <- c(drift = pm['drift', 'sd'], cint = pm['cint', 'sd'], diffusion = pm['diffusion', 'sd'])
cat(sprintf('fit time %.1f min, loglik %.2f\n', el, s$loglik))
print(round(rbind(true = true, est = est), 3))

# naive gaussian model on the same data, for comparison of the dynamics
mg <- ctModel(type = 'ct', manifestNames = 'NA_vas', latentNames = 'negaff',
  LAMBDA = matrix(1), MANIFESTMEANS = 0, CINT = 'cint', DRIFT = 'drift', DIFFUSION = 'diffusion',
  MANIFESTVAR = 'sigma', T0MEANS = 't0m', T0VAR = 't0var', silent = TRUE)
mg$pars$indvarying <- FALSE
mg$pars$indvarying[mg$pars$param %in% 'cint'] <- TRUE
fitg <- ctStanFit(sim, mg, cores = 1, verbose = 0)
driftg <- summary(fitg, digits = 6)$popmeans['drift', 'mean']
cat(sprintf('gaussian model drift %.3f (true %.3f)\n', driftg, true['drift']))

saveRDS(list(true = true, est = est, se = se, driftg = driftg, minutes = el,
  zeros = mean(sim$NA_vas == 0), fit = if(seed == 1) fit else NULL, sim = if(seed == 1) sim else NULL), out)
