#' Gauss-Hermite nodes and weights for a standard normal
#'
#' Probabilists' Gauss-Hermite rule via the Golub-Welsch algorithm, used for
#' ordered beta measurement updates (manifesttype = 2). Weights sum to one, so that
#' \code{sum(w * f(x))} approximates \code{E[f(Z)]} for \code{Z ~ N(0,1)}.
#'
#' @param n number of nodes.
#' @return list with nodes \code{x} and weights \code{w}.
#' @keywords internal
ctOrdBetaGaussHermite <- function(n = 15){
  if(n < 2) stop('n must be at least 2')
  J <- matrix(0, n, n)
  off <- sqrt(seq_len(n - 1))
  J[cbind(1:(n - 1), 2:n)] <- off
  J[cbind(2:n, 1:(n - 1))] <- off
  e <- eigen(J, symmetric = TRUE)
  o <- order(e$values)
  x <- e$values[o]
  w <- e$vectors[1, o]^2
  list(x = x, w = w / sum(w))
}

#' Ordered beta distribution helpers
#'
#' Density, random generation and moments of the ordered beta distribution
#' (Kubinec, 2023) as used for \code{manifesttype = 2} in \code{\link{ctModel}}.
#' \code{eta} is the linear predictor on the logit scale, \code{k1 < k2} are the cutpoints
#' and \code{phi} is the beta precision.
#'
#' @param y observations in the closed interval 0 to 1.
#' @param n number of draws.
#' @param eta linear predictor (logit scale).
#' @param k1 lower cutpoint.
#' @param k2 upper cutpoint, greater than k1.
#' @param phi beta precision, positive.
#' @param log logical, return log density?
#' @return \code{dordbeta}: density (mass at 0 and 1, density in between);
#' \code{rordbeta}: random draws; \code{mordbeta}: matrix with expected value and variance.
#' @references Kubinec, R. (2023). Ordered beta regression: A parsimonious, well-fitting model
#' for continuous data with lower and upper bounds. Political Analysis, 31(4), 519-536.
#' @examples
#' y <- rordbeta(1000, eta = 0, k1 = -1, k2 = 2, phi = 10)
#' mean(y == 0); mean(y == 1)
#' @export
rordbeta <- function(n, eta, k1, k2, phi){
  eta <- rep_len(eta, n)
  u <- stats::runif(n)
  p0 <- 1 - stats::plogis(eta - k1)
  p1 <- stats::plogis(eta - k2)
  mu <- pmin(pmax(stats::plogis(eta), 1e-10), 1 - 1e-10)
  y <- stats::rbeta(n, mu * phi, (1 - mu) * phi)
  y <- pmin(pmax(y, 1e-8), 1 - 1e-8)
  y[u < p0] <- 0
  y[u >= p0 & u < p0 + p1] <- 1
  y
}

#' @rdname rordbeta
#' @export
dordbeta <- function(y, eta, k1, k2, phi, log = FALSE){
  n <- max(length(y), length(eta))
  y <- rep_len(y, n)
  eta <- rep_len(eta, n)
  out <- numeric(n)
  z <- y <= 0
  o <- y >= 1
  m <- !z & !o
  out[z] <- stats::plogis(k1 - eta[z], log.p = TRUE)
  out[o] <- stats::plogis(eta[o] - k2, log.p = TRUE)
  if(any(m)){
    mu <- pmin(pmax(stats::plogis(eta[m]), 1e-10), 1 - 1e-10)
    out[m] <- stats::plogis(eta[m] - k1, log.p = TRUE) + stats::plogis(k2 - eta[m], log.p = TRUE) +
      log(-expm1(k1 - k2)) + stats::dbeta(y[m], mu * phi, (1 - mu) * phi, log = TRUE)
  }
  if(log) out else exp(out)
}

#' @rdname rordbeta
#' @export
mordbeta <- function(eta, k1, k2, phi){
  p1 <- stats::plogis(eta - k2)
  pmid <- stats::plogis(eta - k1) - p1
  mu <- stats::plogis(eta)
  ey <- p1 + pmid * mu
  ey2 <- p1 + pmid * (mu * (1 - mu) / (1 + phi) + mu^2)
  cbind(mean = ey, var = ey2 - ey^2)
}
