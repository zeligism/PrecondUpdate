
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

def norm_scaled(x,D):
    return np.sqrt(np.sum(x * D * x))

def initialize_D(X,y,w, BS, precond="hutchinson", N=10, eps=1e-10):
    if precond == "hessian":
        D = np.diagonal(hessian(X,y,w)) + eps
        return D
    elif precond == "hutchinson":
        # grad(w+z) - grad(w) approximates H(w)z
        D = 0.
        for _ in range(N):
            z = sample_z(w.shape)
            i = np.random.choice(X.shape[0], BS)  # XXX
            D += z * hvp(X,y,w,z,i=None) / N
        D[D < eps] = eps
        return D
    else:
        return 1.


def OASIS(X, y, T=10000, BS=1, gamma=1.0, beta=0.99, lam=0.0, alpha=1e-5,
          precond="hutchinson", precond_warmup=10, precond_resample=True):
    ep = 0  # count effective (full) passes through datatset
    data = []
    theta = 1e10
    w = np.zeros(X.shape[1])
    D = initialize_D(X,y,w,BS, precond=precond, N=precond_warmup)

    # first step
    i = np.random.choice(X.shape[0], BS)
    g = grad(X,y,w,i,lam=lam)
    ep += BS / X.shape[0]
    # update
    w_prev = w[:]
    w = w - gamma * D**-1 * g

    for it in range(T * (X.shape[0] // BS)):
        # Calculate gradients
        i = np.random.choice(X.shape[0], BS)
        g_prev = grad(X,y,w_prev,i,lam=lam)
        g = grad(X,y,w,i,lam=lam)
        ep += BS / X.shape[0]

        # estimate hessian diagonal
        z = sample_z(w.shape)
        if precond_resample:
            j = np.random.choice(X.shape[0], BS)
            ep += BS / X.shape[0]
        else:
            j = i
        D_est = z * hvp(X,y,w,z,j) + lam
        D = np.abs(beta * D + (1-beta) * D_est)
        D[D < alpha] = alpha

        # adaptive learning rate @TODO
        """
        gamma_prev = gamma
        gamma_est = 0.5 * norm_scaled(w-w_prev,D) / norm_scaled(g-g_prev,D)
        gamma = np.minimum(gamma * np.sqrt(1 + theta), gamma_est)
        theta = gamma / gamma_prev
        """

        # update
        w_prev = w[:]
        w = w - gamma * D**-1 * g

        if it % (X.shape[0] // (BS * DATA_FREQ_PER_EPOCH)) == 0:
            data.append(collect_data(ep,X,y,w,lam))
    return w, np.array(data)


def SGD(X, y, T=10000, BS=1, gamma=0.0002, beta=0.999, lam=0.0, alpha=1e-5,
        precond="hutchinson", precond_warmup=10, precond_resample=True):
    ep = 0  # count effective (full) passes through datatset
    data = []
    w = np.zeros(X.shape[1])
    D = initialize_D(X,y,w,BS, precond=precond, N=precond_warmup)

    for it in range(T * (X.shape[0] // BS)):
        # Calculate gradients
        i = np.random.choice(X.shape[0], BS)
        g = grad(X,y,w,i,lam=lam)
        ep += BS / X.shape[0]

        # Diagonal preconditioning
        if precond == "hessian":
            # full hessian
            D = np.diagonal(hessian(X,y,w)) + lam + 1e-10
        elif precond == "hutchinson":
            # estimate hessian diagonal
            z = sample_z(w.shape)
            if precond_resample:
                j = np.random.choice(X.shape[0], BS)
                ep += BS / X.shape[0]
            else:
                j = i
            D_est = z * hvp(X,y,w,z,j) + lam
            D = np.abs(beta * D + (1-beta) * D_est)
            D[D < alpha] = alpha

        # Update rule
        w = w - gamma * D**-1 * g

        # Update data
        if it % (X.shape[0] // (BS * DATA_FREQ_PER_EPOCH)) == 0:
            data.append(collect_data(ep,X,y,w,lam))

    return w, np.array(data)


def SARAH(X, y, T=10, BS=1, gamma=0.2, beta=0.999, lam=0.0, alpha=1e-5,
          precond="hutchinson", precond_warmup=10, precond_resample=True):
    ep = 0  # count effective (full) passes through datatset
    data = []
    wn = np.zeros(X.shape[1])
    D = initialize_D(X,y,wn,BS, precond=precond, N=precond_warmup)

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
            wp = wn[:]
            ep += BS / X.shape[0]

            # Diagonal preconditioning
            if precond == "hessian":
                # full hessian
                D = np.diagonal(hessian(X,y,wn)) + lam + 1e-10
            elif precond == "hutchinson":
                # estimate hessian diagonal
                z = sample_z(wn.shape)
                if precond_resample:
                    j = np.random.choice(X.shape[0], BS)
                    ep += BS / X.shape[0]
                else:
                    j = i
                D_est = z * hvp(X,y,wn,z,j) + lam
                D = np.abs(beta * D + (1-beta) * D_est)
                D[D < alpha] = alpha

            # Update rule
            wn = wn - gamma * D**-1 * v

            # Update data
            if it % (X.shape[0] // (BS * DATA_FREQ_PER_EPOCH)) == 0:
                data.append(collect_data(ep,X,y,wn,lam))

            # Inner loop stopping criterion
            nv = np.linalg.norm(v)
            if nv < 0.1*nv0 or it > X.shape[0] // BS:
                break

    return wn, np.array(data)


def SVRG(X, y, T=10, BS=1, gamma=0.2, beta=0.999, lam=0.0, alpha=1e-5,
         precond="hutchinson", precond_warmup=10, precond_resample=True):
    ep = 0  # count effective (full) passes through datatset
    data = []
    w_out = np.zeros(X.shape[1])
    D = initialize_D(X,y,w_out,BS, precond=precond, N=precond_warmup)

    for epoch in range(T):
        g_full = grad(X,y,w_out,lam=lam)
        ep += 1
        gnorm0 = np.linalg.norm(g_full)
        w_in = w_out[:]

        for it in range(10**10):
            # Calculate gradients
            i = np.random.choice(X.shape[0], BS)
            g_in = grad(X,y,w_in,i,lam=lam)
            g_out = grad(X,y,w_out,i,lam=lam)
            v = g_in - g_out + g_full
            ep += BS / X.shape[0]

            # Diagonal preconditioning
            if precond == "hessian":
                # full hessian
                D = np.diagonal(hessian(X,y,w_in)) + lam + 1e-10
            elif precond == "hutchinson":
                # estimate hessian diagonal
                z = sample_z(w_in.shape)
                if precond_resample:
                    j = np.random.choice(X.shape[0], BS)
                    ep += BS / X.shape[0]
                else:
                    j = i
                D_est = z * hvp(X,y,w_in,z,j) + lam
                D = np.abs(beta * D + (1-beta) * D_est)
                D[D < alpha] = alpha

            # Update rule
            w_in = w_in - gamma * D**-1 * v

            # Update data
            if it % (X.shape[0] // (BS * DATA_FREQ_PER_EPOCH)) == 0:
                data.append(collect_data(ep,X,y,w_in,lam))

            # Inner loop stopping criterion
            gnorm = np.linalg.norm(g_full)
            if gnorm < 0.1 * gnorm0 or it > X.shape[0] // BS:
                w_out = w_in[:]
                break

    return w_in, np.array(data)


def L_SVRG(X, y, T=10000, BS=1, gamma=0.2, beta=0.999, lam=0.0, alpha=1e-5, p=0.99,
           precond="hutchinson", precond_warmup=10, precond_resample=True):
    ep = 0  # count effective (full) passes through datatset
    data = []
    w_out = np.zeros(X.shape[1])
    D = initialize_D(X,y,w_out,BS,precond=precond)

    w_in = w_out[:]
    g_full = grad(X,y,w_out,lam=lam)
    ep += 1

    for it in range(T * (X.shape[0] // BS)):
        # Calculate gradients
        i = np.random.choice(X.shape[0], BS)
        g_in = grad(X,y,w_in,i,lam=lam)
        g_out = grad(X,y,w_out,i,lam=lam)
        v = g_in - g_out + g_full
        ep += BS / X.shape[0]

        # Diagonal preconditioning
        if precond == "hessian":
            # full hessian
            D = np.diagonal(hessian(X,y,w_in)) + lam + 1e-10
        elif precond == "hutchinson":
            # estimate hessian diagonal
            z = sample_z(w_in.shape)
            if precond_resample:
                j = np.random.choice(X.shape[0], BS)
                ep += BS / X.shape[0]
            else:
                j = i
            D_est = z * hvp(X,y,w_in,z,j) + lam
            D = np.abs(beta * D + (1-beta) * D_est)
            D[D < alpha] = alpha

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

