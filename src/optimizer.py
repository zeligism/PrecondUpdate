
from loss import F, grad, hessian_diag, hvp
from plot import plot_hessian_acc, plot_hessian_approx

import numpy as np

DATA_FREQ_PER_EPOCH = 5  # TODO: add this param somewhere


def collect_data(ep,X,y,w, lam=0.0, D=None, D_ratio=0.0):
    loss = F(X,y,w)
    g_norm = np.linalg.norm(grad(X,y,w))**2
    error = np.mean(X.dot(w) * y < 0)  # wrong prediction -> 100% error
    error += 0.5 * np.mean(X.dot(w) * y == 0)  # ambiguous prediction -> 50% error
    H_diag = hessian_diag(X,y,w)
    H_diag_err = np.linalg.norm(D - H_diag) / np.linalg.norm(H_diag)
    return (ep, loss, g_norm, error, D_ratio, H_diag_err)


def sample_z(size):
    return 2 * np.random.randint(0,2,size) - 1


def init_diagonal(X,y,w, BS, precond="hutchinson",
                  precond_warmup=1, precond_zsamples=1, alpha=1e-7, full_batch_hvp=False,):
    if precond == "hessian":
        D = hessian_diag(X,y,w)
    elif precond == "hutchinson":
        # grad(w+z) - grad(w) approximates H(w)z
        D = 0.
        H_diag = hessian_diag(X,y,w)
        H_diag_errs = []
        N = precond_warmup * precond_zsamples
        for _ in range(precond_warmup):
            i = None if full_batch_hvp else np.random.choice(X.shape[0], BS)
            for _ in range(precond_zsamples):
                z = sample_z(w.shape)
                D += z * hvp(X,y,w,z,i) / N
                H_diag_errs.append(
                    np.linalg.norm(D - H_diag) / np.linalg.norm(H_diag))
        # plot_H_acc(H_diag, D)
        # plot_H_approx(H_diag_errs)
    else:
        D = np.ones_like(w)

    D = np.abs(D)
    D = np.maximum(D, alpha)
    return D


def update_diagonal(D, X, y, w, i, beta=0.999, lam=0.0, alpha=1e-5,
                    precond="hutchinson", precond_zsamples=1):
    # Diagonal preconditioning
    if precond == "hessian":
        # full hessian
        D = hessian_diag(X,y,w,i=None,lam=lam)
    elif precond == "hutchinson":
        # estimate hessian diagonal
        D_est = 0.0
        for _ in range(precond_zsamples):
            z = sample_z(w.shape)
            D_est += z * hvp(X,y,w,z,i) / precond_zsamples
        D = np.abs(beta * D + (1 - beta) * D_est) + lam
    else:
        D = np.ones_like(w)

    D = np.maximum(D, alpha)
    return D


def OASIS(X, y, T=10000, BS=1, gamma=1.0, beta=0.99, lam=0.0, alpha=1e-5, adaptive_gamma=True,
          precond="hutchinson", precond_warmup=10, precond_resample=False, precond_zsamples=1):
    ep = 0  # count effective (full) passes through dataset
    data = []
    w = np.zeros(X.shape[1])
    w_prev = w[:]
    theta = 1e100

    D = init_diagonal(X,y,w,BS, alpha=alpha, precond=precond,
                      precond_warmup=precond_warmup, precond_zsamples=precond_zsamples)
    D_ratio = np.mean(D > alpha)
    data.append(collect_data(ep,X,y,w,lam,D,D_ratio))

    for it in range(T * (X.shape[0] // BS)):
        # Calculate gradients
        i = np.random.choice(X.shape[0], BS)
        g = grad(X,y,w,i,lam=lam)
        ep += BS / X.shape[0]

        if precond == "hutchinson" and precond_resample:
            j = np.random.choice(X.shape[0], BS)
            ep += BS / X.shape[0]
        else:
            j = i

        D = update_diagonal(D,X,y,w,j, beta=beta, lam=lam, alpha=alpha,
                            precond=precond, precond_zsamples=precond_zsamples)
        D_ratio = np.mean(D > alpha)

        # adaptive learning rate
        if adaptive_gamma and it > 0:
            g_prev = grad(X,y,w_prev,i,lam=lam)
            gamma_prev = gamma
            gamma = gamma * np.sqrt(1 + theta)
            gamma_est = 0.5 * np.sqrt(
                np.sum(D * (w - w_prev)**2) / (np.sum(D**-1 * (g - g_prev)**2))
            )
            if gamma_est < gamma:
                gamma = gamma_est
            theta = gamma / gamma_prev

        # update
        w_prev = w[:]
        w -= gamma * D**-1 * g

        if it % (X.shape[0] // (BS * DATA_FREQ_PER_EPOCH)) == 0:
            data.append(collect_data(ep,X,y,w,lam,D,D_ratio))
    return w, np.array(data)


def SGD(X, y, T=10000, BS=1, gamma=0.0002, beta=0.999, lam=0.0, alpha=1e-5,
        precond="hutchinson", precond_warmup=10, precond_resample=False, precond_zsamples=1,
        momentum=0.0, adam=False):
    ep = 0  # count effective (full) passes through dataset
    data = []
    w = np.zeros(X.shape[1])
    if adam:
        m = w[:]
        v = w[:]

    D = init_diagonal(X,y,w,BS, alpha=alpha, precond=precond,
                      precond_warmup=precond_warmup, precond_zsamples=precond_zsamples)
    D_ratio = np.mean(D > alpha)
    data.append(collect_data(ep,X,y,w,lam,D,D_ratio))

    for it in range(T * (X.shape[0] // BS)):
        # Calculate gradients
        i = np.random.choice(X.shape[0], BS)
        g = grad(X,y,w,i,lam=lam)
        ep += BS / X.shape[0]

        if precond == "hutchinson" and precond_resample:
            j = np.random.choice(X.shape[0], BS)
            ep += BS / X.shape[0]
        else:
            j = i
        D = update_diagonal(D,X,y,w,j, beta=beta, lam=lam, alpha=alpha,
                            precond=precond, precond_zsamples=precond_zsamples)
        D_ratio = np.mean(D > alpha)

        if adam:
            m = momentum * m + (1 - momentum) * g
            v = beta * v + (1 - beta) * g**2
            m_corr = m / (1 - momentum**(it+1))
            v_corr = v / (1 - beta**(it+1))
            D_mix = np.sqrt(v_corr) + D
            w -= gamma * D_mix**-1 * m_corr
        else:
            w -= gamma * D**-1 * g

        # Update data
        if it % (X.shape[0] // (BS * DATA_FREQ_PER_EPOCH)) == 0:
            data.append(collect_data(ep,X,y,w,lam,D,D_ratio))

    return w, np.array(data)


def SARAH(X, y, T=10, BS=1, gamma=0.2, beta=0.999, lam=0.0, alpha=1e-5,
          precond="hutchinson", precond_warmup=10, precond_resample=False, precond_zsamples=1):
    ep = 0  # count effective (full) passes through dataset
    data = []
    wn = np.zeros(X.shape[1])

    D = init_diagonal(X,y,wn,BS, alpha=alpha, precond=precond,
                      precond_warmup=precond_warmup, precond_zsamples=precond_zsamples)
    D_ratio = np.mean(D > alpha)
    data.append(collect_data(ep,X,y,wn,lam,D,D_ratio))

    for epoch in range(T):
        g_full = grad(X,y,wn,lam=lam)
        g = g_full[:]
        ep += 1
        nv0 = np.linalg.norm(g)
        wp = wn[:]

        for it in range(10**10):
            # Calculate gradients
            i = np.random.choice(X.shape[0], BS)
            gn = grad(X,y,wn,i,lam=lam)
            gp = grad(X,y,wp,i,lam=lam)
            g += gn - gp
            nv = np.linalg.norm(g)
            ep += BS / X.shape[0]

            if precond == "hutchinson" and precond_resample:
                j = np.random.choice(X.shape[0], BS)
                ep += BS / X.shape[0]
            else:
                j = i
            D = update_diagonal(D,X,y,wn,j, beta=beta, lam=lam, alpha=alpha,
                                precond=precond, precond_zsamples=precond_zsamples)
            D_ratio = np.mean(D > alpha)

            # Update rule
            wp = wn[:]
            wn -= gamma * D**-1 * g

            # Update data
            freq = X.shape[0] // (BS * DATA_FREQ_PER_EPOCH)
            if freq > 0 and it % freq == 0:
                data.append(collect_data(ep,X,y,wn,lam,D,D_ratio))

            # Inner loop stopping criterion
            if nv < 0.1 * nv0 or it > X.shape[0] // BS:
                data.append(collect_data(ep,X,y,wn,lam,D,D_ratio))
                break

    return wn, np.array(data)


def SVRG(X, y, T=10, BS=1, gamma=0.2, beta=0.999, lam=0.0, alpha=1e-5,
         precond="hutchinson", precond_warmup=10, precond_resample=False, precond_zsamples=1):
    ep = 0  # count effective (full) passes through dataset
    data = []
    w_out = np.zeros(X.shape[1])

    D = init_diagonal(X,y,w_out,BS, alpha=alpha, precond=precond,
                      precond_warmup=precond_warmup, precond_zsamples=precond_zsamples)
    D_ratio = np.mean(D > alpha)
    data.append(collect_data(ep,X,y,w_out,lam,D,D_ratio))

    for epoch in range(T):
        g_full = grad(X,y,w_out,lam=lam)
        ep += 1
        nv0 = np.linalg.norm(g_full)
        w_in = w_out[:]

        for it in range(10**10):
            # Calculate gradients
            i = np.random.choice(X.shape[0], BS)
            g_in = grad(X,y,w_in,i,lam=lam)
            g_out = grad(X,y,w_out,i,lam=lam)
            g = g_full + g_in - g_out
            nv = np.linalg.norm(g)
            ep += BS / X.shape[0]

            if precond == "hutchinson" and precond_resample:
                j = np.random.choice(X.shape[0], BS)
                ep += BS / X.shape[0]
            else:
                j = i
            D = update_diagonal(D,X,y,w_in,j, beta=beta, lam=lam, alpha=alpha,
                                precond=precond, precond_zsamples=precond_zsamples)
            D_ratio = np.mean(D > alpha)

            # Update rule
            w_in -= gamma * D**-1 * g

            # Update data
            if it % (X.shape[0] // (BS * DATA_FREQ_PER_EPOCH)) == 0:
                data.append(collect_data(ep,X,y,w_in,lam,D,D_ratio))

            # Inner loop stopping criterion
            if nv < 0.1 * nv0 or it > X.shape[0] // BS:
                w_out = w_in[:]
                break

    return w_in, np.array(data)


def L_SVRG(X, y, T=10000, BS=1, gamma=0.2, beta=0.999, lam=0.0, alpha=1e-5, p=0.99,
           precond="hutchinson", precond_warmup=10, precond_resample=False, precond_zsamples=1):
    ep = 0  # count effective (full) passes through dataset
    data = []
    w_out = np.zeros(X.shape[1])
    w_in = w_out[:]
    g_full = grad(X,y,w_out,lam=lam)

    D = init_diagonal(X,y,w_out,BS, alpha=alpha, precond=precond,
                      precond_warmup=precond_warmup, precond_zsamples=precond_zsamples)
    D_ratio = np.mean(D > alpha)
    data.append(collect_data(ep,X,y,w_out,lam,D,D_ratio))

    ep += 1

    for it in range(T * (X.shape[0] // BS)):
        # Calculate gradients
        i = np.random.choice(X.shape[0], BS)
        g_in = grad(X,y,w_in,i,lam=lam)
        g_out = grad(X,y,w_out,i,lam=lam)
        g = g_full + g_in - g_out
        ep += BS / X.shape[0]

        if precond == "hutchinson" and precond_resample:
            j = np.random.choice(X.shape[0], BS)
            ep += BS / X.shape[0]
        else:
            j = i
        D = update_diagonal(D,X,y,w_in,j, beta=beta, lam=lam, alpha=alpha,
                            precond=precond, precond_zsamples=precond_zsamples)
        D_ratio = np.mean(D > alpha)

        # Inner loop stopping criterion is now for updating w_out
        # and comes before the update rule @XXX: move after update data?
        if np.random.rand(1)[0] > p:
            w_out = w_in[:]
            g_full = grad(X,y,w_out,lam=lam)
            ep += 1

        # Update rule
        w_in -= gamma * D**-1 * g

        # Update data
        if it % (X.shape[0] // (BS * DATA_FREQ_PER_EPOCH)) == 0:
            data.append(collect_data(ep,X,y,w_in,lam,D,D_ratio))

    return w_in, np.array(data)


def Adam(X, y, T=10000, BS=1, gamma=0.2, beta1=0.9, beta2=0.999, eps=1e-8, **_):
    ep = 0  # count effective (full) passes through dataset
    data = []
    w = np.zeros(X.shape[1])
    m = w[:]
    v = w[:]
    data.append(collect_data(ep,X,y,w,0.,0.,0.))

    for it in range(T * (X.shape[0] // BS)):
        # Calculate gradients
        i = np.random.choice(X.shape[0], BS)
        g = grad(X,y,w,i)
        ep += BS / X.shape[0]

        m = beta1 * m + (1 - beta1) * g
        v = beta2 * v + (1 - beta2) * g**2

        m_corr = m / (1 - beta1**(it+1))
        v_corr = v / (1 - beta2**(it+1))

        # Update rule
        w = w - gamma * (m_corr / (np.sqrt(v_corr) + eps))

        # Update data
        if it % (X.shape[0] // (BS * DATA_FREQ_PER_EPOCH)) == 0:
            data.append(collect_data(ep,X,y,w,0.,0.,0.))

    return w, np.array(data)
