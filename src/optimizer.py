
from loss import F, grad, hessian, hvp
import numpy as np

DATA_FREQ_PER_EPOCH = 5  # TODO: add this param somewhere


def collect_data(ep,X,y,w,lam=0.0):
    loss = F(X,y,w)
    g_norm = np.linalg.norm(grad(X,y,w))**2
    error = np.mean(X.dot(w)*y < 0)
    return (ep, loss, g_norm, error)

def sample_z(size):
    return 2 * np.random.randint(0,2,size) - 1


def init_diagonal(X,y,w, BS, precond="hutchinson",
                  precond_warmup=1, precond_zsamples=1, alpha=1e-7, full_batch_hvp=True,):
    if precond == "hessian":
        D = hessian(X,y,w).diagonal()
    elif precond == "hutchinson":
        # grad(w+z) - grad(w) approximates H(w)z
        D = 0.
        N = precond_warmup * precond_zsamples
        for _ in range(precond_warmup):
            i = None if full_batch_hvp else np.random.choice(X.shape[0], BS)
            for _ in range(precond_zsamples):
                z = sample_z(w.shape)
                D += z * hvp(X,y,w,z,i) / N
    else:
        D = np.ones_like(w)

    D[D < alpha] = alpha
    return D

def update_diagonal(D, X, y, w, i, beta=0.999, lam=0.0, alpha=1e-5,
                    precond="hutchinson", precond_zsamples=1):
    # Diagonal preconditioning
    if precond == "hessian":
        # full hessian
        D = hessian(X,y,w).diagonal() + lam
    elif precond == "hutchinson":
        # estimate hessian diagonal
        D_est = 0.0
        for _ in range(precond_zsamples):
            z = sample_z(w.shape)
            D_est += z * hvp(X,y,w,z,i) / precond_zsamples
        D = np.abs(beta * D + (1-beta) * D_est) + lam
    else:
        D = np.ones_like(w)

    D[D < alpha] = alpha
    return D


def OASIS(X, y, T=10000, BS=1, gamma=1.0, beta=0.99, lam=0.0, alpha=1e-5, adaptive_gamma=True,
          precond="hutchinson", precond_warmup=10, precond_resample=True, precond_zsamples=1):
    ep = 0  # count effective (full) passes through datatset
    data = []
    theta = 1e10
    w = np.zeros(X.shape[1])
    w_prev = w[:]
    D = init_diagonal(X,y,w,BS, alpha=alpha, precond=precond,
                      precond_warmup=precond_warmup, precond_zsamples=precond_zsamples)

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
        D = update_diagonal(D,X,y,w,j, beta=beta, lam=lam, alpha=0.5*np.linalg.norm(g)**2,
                            precond=precond, precond_zsamples=precond_zsamples)

        # adaptive learning rate @TODO
        if adaptive_gamma and it>0:
            g_prev = grad(X,y,w_prev,i,lam=lam)
            gamma_prev = gamma
            gamma = gamma * np.sqrt(1 + theta)
            gamma_est = 0.5 * np.sqrt(np.sum(D * (w-w_prev)**2) / (np.sum(D**-1 * (g-g_prev)**2)))
            if gamma_est < gamma:
                gamma = gamma_est
            theta = gamma / gamma_prev

        # update
        w_prev = w[:]
        w = w - gamma * D**-1 * g

        if it % (X.shape[0] // (BS * DATA_FREQ_PER_EPOCH)) == 0:
            data.append(collect_data(ep,X,y,w,lam))
    return w, np.array(data)


def SGD(X, y, T=10000, BS=1, gamma=0.0002, beta=0.999, lam=0.0, alpha=1e-5,
        precond="hutchinson", precond_warmup=10, precond_resample=True, precond_zsamples=1):
    ep = 0  # count effective (full) passes through datatset
    data = []
    w = np.zeros(X.shape[1])
    D = init_diagonal(X,y,w,BS, alpha=alpha, precond=precond,
                      precond_warmup=precond_warmup, precond_zsamples=precond_zsamples)

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
        D = update_diagonal(D,X,y,w,j, beta=beta, lam=lam, alpha=0.5*np.linalg.norm(g)**2,
                            precond=precond, precond_zsamples=precond_zsamples)

        # Update rule
        w = w - gamma * D**-1 * g

        # Update data
        if it % (X.shape[0] // (BS * DATA_FREQ_PER_EPOCH)) == 0:
            data.append(collect_data(ep,X,y,w,lam))

    return w, np.array(data)


def SARAH(X, y, T=10, BS=1, gamma=0.2, beta=0.999, lam=0.0, alpha=1e-5,
          precond="hutchinson", precond_warmup=10, precond_resample=True, precond_zsamples=1):
    ep = 0  # count effective (full) passes through datatset
    data = []
    wn = np.zeros(X.shape[1])
    D = init_diagonal(X,y,wn,BS, alpha=alpha, precond=precond,
                      precond_warmup=precond_warmup, precond_zsamples=precond_zsamples)

    for epoch in range(T):
        v = grad(X,y,wn,lam=lam)
        ep += 1
        nv0 = np.linalg.norm(v)
        wp = wn[:]

        for it in range(10**10):
            # Calculate gradients
            i = np.random.choice(X.shape[0], BS)
            gn = grad(X,y,wn,i,lam=lam)
            gp = grad(X,y,wp,i,lam=lam)
            v += gn - gp
            nv = np.linalg.norm(v)
            ep += BS / X.shape[0]

            if precond == "hutchinson" and precond_resample:
                j = np.random.choice(X.shape[0], BS)
                ep += BS / X.shape[0]
            else:
                j = i
            D = update_diagonal(D,X,y,wn,j, beta=beta, lam=lam, alpha=0.5*nv**2,
                                precond=precond, precond_zsamples=precond_zsamples)

            # Update rule
            wp = wn[:]
            wn = wn - gamma * D**-1 * v

            # Update data
            if it % (X.shape[0] // (BS * DATA_FREQ_PER_EPOCH)) == 0:
                data.append(collect_data(ep,X,y,wn,lam))

            # Inner loop stopping criterion
            if nv < 0.1*nv0 or it > X.shape[0] // BS:
                break

    return wn, np.array(data)


def SVRG(X, y, T=10, BS=1, gamma=0.2, beta=0.999, lam=0.0, alpha=1e-5,
         precond="hutchinson", precond_warmup=10, precond_resample=True, precond_zsamples=1):
    ep = 0  # count effective (full) passes through datatset
    data = []
    w_out = np.zeros(X.shape[1])
    D = init_diagonal(X,y,w_out,BS, alpha=alpha, precond=precond,
                      precond_warmup=precond_warmup, precond_zsamples=precond_zsamples)

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
            v = g_in - g_out + g_full
            nv = np.linalg.norm(v)
            ep += BS / X.shape[0]

            if precond == "hutchinson" and precond_resample:
                j = np.random.choice(X.shape[0], BS)
                ep += BS / X.shape[0]
            else:
                j = i
            D = update_diagonal(D,X,y,w_in,j, beta=beta, lam=lam, alpha=0.5*nv**2,
                                precond=precond, precond_zsamples=precond_zsamples)

            # Update rule
            w_in = w_in - gamma * D**-1 * v

            # Update data
            if it % (X.shape[0] // (BS * DATA_FREQ_PER_EPOCH)) == 0:
                data.append(collect_data(ep,X,y,w_in,lam))

            # Inner loop stopping criterion
            if nv < 0.1 * nv0 or it > X.shape[0] // BS:
                w_out = w_in[:]
                break

    return w_in, np.array(data)


def L_SVRG(X, y, T=10000, BS=1, gamma=0.2, beta=0.999, lam=0.0, alpha=1e-5, p=0.99,
           precond="hutchinson", precond_warmup=10, precond_resample=True, precond_zsamples=1):
    ep = 0  # count effective (full) passes through datatset
    data = []
    w_out = np.zeros(X.shape[1])
    D = init_diagonal(X,y,w_out,BS, alpha=alpha, precond=precond,
                      precond_warmup=precond_warmup, precond_zsamples=precond_zsamples)

    w_in = w_out[:]
    g_full = grad(X,y,w_out,lam=lam)
    ep += 1

    for it in range(T * (X.shape[0] // BS)):
        # Calculate gradients
        i = np.random.choice(X.shape[0], BS)
        g_in = grad(X,y,w_in,i,lam=lam)
        g_out = grad(X,y,w_out,i,lam=lam)
        v = g_in - g_out + g_full
        nv = np.linalg.norm(v)
        ep += BS / X.shape[0]

        if precond == "hutchinson" and precond_resample:
            j = np.random.choice(X.shape[0], BS)
            ep += BS / X.shape[0]
        else:
            j = i
        D = update_diagonal(D,X,y,w_in,j, beta=beta, lam=lam, alpha=0.5*nv**2,
                            precond=precond, precond_zsamples=precond_zsamples)

        # Inner loop stopping criterion is now for updating w_out
        # and comes before the update rule
        if np.random.rand(1)[0] > p:
            w_out = w_in[:]
            g_full = grad(X,y,w_out,lam=lam)
            ep += 1

        # Update rule
        w_in = w_in - gamma * D**-1 * v

        # Update data
        if it % (X.shape[0] // (BS * DATA_FREQ_PER_EPOCH)) == 0:
            data.append(collect_data(ep,X,y,w_in,lam))

    return w_in, np.array(data)


