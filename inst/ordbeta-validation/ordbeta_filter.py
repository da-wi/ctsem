"""Validation of the ordered-beta measurement update used in the ctsem extension.

Measurement model (Kubinec 2023, ordered beta), eta = linear predictor:
  P(y = 0)       = 1 - logistic(eta - k1)
  P(y = 1)       = logistic(eta - k2)
  p(y | 0<y<1)   = [logistic(eta - k1) - logistic(eta - k2)] * Beta(y | mu*phi, (1-mu)*phi),  mu = logistic(eta)

Filter step for one scalar observation, eta ~ N(m, s2) from the Kalman prediction:
  Z      = integral p(y|eta) N(eta|m,s2) d eta   (one-step predictive likelihood)
  E, V   = posterior mean / variance of eta given y
computed by adaptive Gauss-Hermite quadrature (nodes centred on the posterior mode,
scaled by the curvature there). The state moments then follow exactly because the
state is Gaussian given eta (linear predictor), see update_state().
"""
import numpy as np
from scipy.special import expit, gammaln, betaln, logsumexp
from scipy import integrate, optimize


def log_expit(x):
    return -np.logaddexp(0.0, -x)


def ordbeta_logpdf(y, eta, k1, k2, phi):
    """Numerically stable: sigma(a)-sigma(b) = sigma(a)*sigma(-b)*(1-exp(b-a)), b-a = k1-k2."""
    eta = np.asarray(eta, dtype=float)
    if y == 0:
        return log_expit(-(eta - k1))
    if y == 1:
        return log_expit(eta - k2)
    mu = expit(eta)
    mu = np.clip(mu, 1e-12, 1 - 1e-12)
    a, b = mu * phi, (1 - mu) * phi
    lbeta = (a - 1) * np.log(y) + (b - 1) * np.log1p(-y) - betaln(a, b)
    lmid = log_expit(eta - k1) + log_expit(k2 - eta) + np.log(-np.expm1(k1 - k2))
    return lmid + lbeta


def ordbeta_rng(rng, eta, k1, k2, phi):
    u = rng.uniform()
    p0 = 1 - expit(eta - k1)
    p1 = expit(eta - k2)
    if u < p0:
        return 0.0
    if u < p0 + p1:
        return 1.0
    mu = expit(eta)
    return rng.beta(mu * phi, (1 - mu) * phi)


GH_X, GH_W = np.polynomial.hermite_e.hermegauss(15)  # probabilists' nodes, weight exp(-x^2/2)
GH_W = GH_W / np.sqrt(2 * np.pi)


def quad_update(y, m, s2, k1, k2, phi, adaptive=True, nnewton=4, fd=1e-4, x=GH_X, w=GH_W):
    """Return log Z, E[eta|y], Var[eta|y]."""
    s = np.sqrt(s2)
    def g(e):  # log of unnormalised posterior
        return ordbeta_logpdf(y, e, k1, k2, phi) - 0.5 * (e - m) ** 2 / s2 - 0.5 * np.log(2 * np.pi * s2)
    if adaptive:
        # start at the better of prior mean and (for continuous y) logit(y): the beta part is not
        # log-concave in its tails, so a far-away start can trap plain Newton.
        c = m
        if 0 < y < 1:
            c2 = np.log(y) - np.log1p(-y)
            if g(c2) > g(c):
                c = c2
        for _ in range(nnewton):  # damped Newton, finite-difference derivatives (as in the Stan code)
            g0, gp, gm = g(c), g(c + fd), g(c - fd)
            d1 = (gp - gm) / (2 * fd)
            d2 = (gp - 2 * g0 + gm) / fd ** 2
            d2 = min(d2, -1.0 / s2)  # posterior at least as concave as prior
            step = np.clip(-d1 / d2, -3 * s, 3 * s)
            for _ in range(6):  # backtracking: only accept ascent steps
                if g(c + step) >= g0:
                    break
                step *= 0.5
            else:
                step = 0.0
            c = c + step
        g0, gp, gm = g(c), g(c + fd), g(c - fd)
        d2 = min((gp - 2 * g0 + gm) / fd ** 2, -1.0 / s2)
        sh = 1.0 / np.sqrt(-d2)
        sh = min(sh, s)  # never wider than the prior
        nodes = c + sh * x
        lw = np.log(w) + g(nodes) - (-0.5 * x ** 2 - np.log(sh) - 0.5 * np.log(2 * np.pi))
    else:
        nodes = m + s * x
        lw = np.log(w) + ordbeta_logpdf(y, nodes, k1, k2, phi)
    logZ = logsumexp(lw)
    p = np.exp(lw - logZ)
    E = np.sum(p * nodes)
    V = max(np.sum(p * (nodes - E) ** 2), 1e-10)
    return logZ, E, V


def exact_update(y, m, s2, k1, k2, phi):
    """Reference: dense grid in log space (robust to underflow)."""
    grid = np.linspace(-40, 40, 400001)
    dx = grid[1] - grid[0]
    lg = ordbeta_logpdf(y, grid, k1, k2, phi) - 0.5 * (grid - m) ** 2 / s2 - 0.5 * np.log(2 * np.pi * s2)
    logZ = logsumexp(lg) + np.log(dx)
    p = np.exp(lg - logsumexp(lg))
    E = np.sum(p * grid)
    V = np.sum(p * (grid - E) ** 2)
    return logZ, E, V


def update_state(xm, P, a, b, Eeta, Veta):
    """State x ~ N(xm,P), eta = a'x + b. Given posterior moments of eta, return posterior moments of x.
    Exact for the first two moments because x | eta is Gaussian."""
    m = a @ xm + b
    s2 = a @ P @ a
    k = P @ a / s2
    xnew = xm + k * (Eeta - m)
    Pnew = P - np.outer(k, k) * s2 + np.outer(k, k) * Veta
    return xnew, Pnew


# ---------- 1. single-step accuracy -----------------------------------------------------------
def single_step_check():
    rng = np.random.default_rng(1)
    k1, k2 = -1.5, 2.0
    errs = {'prior-GH': [], 'adaptive-GH': []}
    worstcell = {'prior-GH': (0, None), 'adaptive-GH': (0, None)}
    for phi in [2, 5, 15, 40, 100, 250]:
        for sd in [0.2, 0.7, 1.5, 3.0, 5.0]:
            for m in [-4, -3, -1, 0, 1.5, 3, 4]:
                for y in [0.0, 1.0, 0.005, 0.03, 0.3, 0.5, 0.8, 0.97, 0.995]:
                    lz, E, V = exact_update(y, m, sd ** 2, k1, k2, phi)
                    for name, ad in [('prior-GH', False), ('adaptive-GH', True)]:
                        lz2, E2, V2 = quad_update(y, m, sd ** 2, k1, k2, phi, adaptive=ad)
                        e = (abs(lz2 - lz), abs(E2 - E) / np.sqrt(V), abs(V2 / V - 1))
                        errs[name].append(e)
                        if e[0] > worstcell[name][0]:
                            worstcell[name] = (e[0], dict(phi=phi, sd=sd, m=m, y=y, logZ=round(lz, 2)))
    print('Single step, 15 nodes; grid phi 2..250, prior sd .2..5, m -4..4, all y types (1890 cells)')
    for n, v in errs.items():
        v = np.array(v)
        print(f'  {n:12s} |err logZ| median {np.median(v[:,0]):.1e}  99% {np.quantile(v[:,0],.99):.1e}  max {v[:,0].max():.1e}'
              f' | err E (post. sd) max {v[:,1].max():.1e} | rel err Var max {v[:,2].max():.1e}')
        print(f'      worst cell: {worstcell[n][1]}')


# ---------- 2. full filter vs particle filter on a CT OU process ------------------------------
def simulate(rng, T, drift, cint, diff, k1, k2, phi, lam=1.0, mm=0.0):
    times = np.cumsum(rng.uniform(0.3, 3.0, T))  # irregular ESM-like intervals
    x = cint / -drift + rng.normal() * np.sqrt(diff ** 2 / (-2 * drift))
    ys = []
    prev = times[0]
    for t in times:
        dt = t - prev
        prev = t
        e = np.exp(drift * dt)
        x = e * x + (e - 1) / drift * cint + rng.normal() * np.sqrt(diff ** 2 * (1 - e ** 2) / (-2 * drift))
        ys.append(ordbeta_rng(rng, lam * x + mm, k1, k2, phi))
    return times, np.array(ys)


def adf_loglik(times, ys, drift, cint, diff, k1, k2, phi, lam=1.0, mm=0.0):
    xm = np.array([cint / -drift])
    P = np.array([[diff ** 2 / (-2 * drift)]])
    a = np.array([lam])
    ll = 0.0
    prev = times[0]
    for t, y in zip(times, ys):
        dt = t - prev
        prev = t
        e = np.exp(drift * dt)
        xm = e * xm + (e - 1) / drift * cint
        P = e * P * e + diff ** 2 * (1 - e ** 2) / (-2 * drift)
        m = a @ xm + mm
        s2 = a @ P @ a
        lz, E, V = quad_update(y, m, s2, k1, k2, phi)
        ll += lz
        xm, P = update_state(xm, P, a, mm, E, V)
    return ll


def pf_loglik(times, ys, drift, cint, diff, k1, k2, phi, lam=1.0, mm=0.0, n=40000, seed=0):
    rng = np.random.default_rng(seed)
    x = cint / -drift + rng.normal(size=n) * np.sqrt(diff ** 2 / (-2 * drift))
    ll = 0.0
    prev = times[0]
    for t, y in zip(times, ys):
        dt = t - prev
        prev = t
        e = np.exp(drift * dt)
        x = e * x + (e - 1) / drift * cint + rng.normal(size=n) * np.sqrt(diff ** 2 * (1 - e ** 2) / (-2 * drift))
        lw = ordbeta_logpdf(y, lam * x + mm, k1, k2, phi)
        lz = logsumexp(lw) - np.log(n)
        ll += lz
        p = np.exp(lw - logsumexp(lw))
        idx = rng.choice(n, n, p=p)
        x = x[idx]
    return ll


if __name__ == '__main__':
    single_step_check()

    rng = np.random.default_rng(42)
    true = dict(drift=-0.4, cint=-0.2, diff=0.9, k1=-1.0, k2=2.5, phi=12.0)
    times, ys = simulate(rng, 300, **true)
    print(f'\nSimulated 300 obs: {np.mean(ys == 0):.0%} exact zeros, {np.mean(ys == 1):.0%} exact ones')

    print('\nLog-likelihood ADF vs particle filter (40k particles), varying drift:')
    for d in [-0.2, -0.4, -0.8]:
        p = dict(true, drift=d)
        print(f'  drift {d:5.2f}:  ADF {adf_loglik(times, ys, **p):9.2f}   PF {pf_loglik(times, ys, **p):9.2f}')

    # ML recovery on a larger series using the ADF likelihood
    times, ys = simulate(np.random.default_rng(7), 2000, **true)
    def negll(th):
        return -adf_loglik(times, ys, drift=-np.exp(th[0]), cint=th[1], diff=np.exp(th[2]),
                           k1=th[3], k2=th[3] + np.exp(th[4]), phi=np.exp(th[5]))
    th0 = np.array([np.log(0.6), 0.0, np.log(0.6), -0.5, np.log(2.0), np.log(5.0)])
    res = optimize.minimize(negll, th0, method='L-BFGS-B')
    th = res.x
    est = dict(drift=-np.exp(th[0]), cint=th[1], diff=np.exp(th[2]), k1=th[3], k2=th[3] + np.exp(th[4]), phi=np.exp(th[5]))
    print('\nML recovery with ADF likelihood, single subject, T=2000:')
    for k in true:
        print(f'  {k:6s} true {true[k]:7.3f}   est {est[k]:7.3f}')
