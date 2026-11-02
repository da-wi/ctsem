# Test A: a Gaussian ctsem model must give identical results with the original and the extended package.
# Usage: Rscript test_regression.R <library path>   (prints loglik and estimates, saved to rds)
args <- commandArgs(trailingOnly = TRUE)
lib <- args[1]; out <- args[2]
suppressPackageStartupMessages(library(ctsem, lib.loc = lib))
set.seed(1)
gm <- ctModel(LAMBDA = diag(2), DRIFT = 'auto', MANIFESTMEANS = 0, CINT = 'auto',
  manifestNames = c('Y1', 'Y2'), latentNames = c('eta1', 'eta2'), type = 'ct', silent = TRUE)
gm$pars$indvarying <- FALSE
gm$pars$indvarying[gm$pars$matrix %in% 'T0MEANS'] <- TRUE
d <- as.data.frame(ctstantestdat)
d <- d[d$id %in% 1:20, c('id', 'time', 'Y1', 'Y2')]
fit <- ctStanFit(d, gm, cores = 1, verbose = 0)
s <- summary(fit, digits = 10)
res <- list(ll = s$loglik, est = fit$stanfit$rawest, pop = s$popmeans)
cat('loglik', format(res$ll, digits = 12), '\n')
print(round(s$popmeans[, 1:3], 5))

# binary + gaussian mix (existing manifesttype 1 path)
d2 <- d
d2$Y2 <- as.numeric(d2$Y2 > stats::median(d2$Y2, na.rm = TRUE))
bm <- gm
bm$manifesttype <- c(0, 1)
fitb <- ctStanFit(d2, bm, cores = 1, verbose = 0)
res$llbin <- summary(fitb)$loglik
res$estbin <- fitb$stanfit$rawest
cat('loglik binary mix', format(res$llbin, digits = 12), '\n')
saveRDS(res, out)
