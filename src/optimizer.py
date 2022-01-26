
from loss import F, grad, hessian, hvp
import numpy as np

def collect_data(ep,X,y,w,lam=0.0):
    loss = F(X,y,w)
    g_norm = np.linalg.norm(grad(X,y,w))**2
    error = np.mean(X.dot(w)*y < 0)
    return (ep, loss, g_norm, error)

def sample_z(size):
    return 2 * np.random.randint(0,2,size) - 1

def norm_scaled(x,D):
    return np.sqrt(np.sum(x * D * x))

def initialize_D(X,y,w, precond="hutchinson", N=100):
    if precond == "hessian":
        return np.diagonal(hessian(X,y,w)) + 1e-10
    elif precond == "hutchinson":
        # grad(w+z) - grad(w) approximates H(w)z
        D = 0.
        for _ in range(N):
            z = sample_z()
            D += z * (grad(X,y,w+z) - grad(X,y,w)) / N
        return D
    else:
        return 1.


def OASIS(X, y, T=10000, BS=1, gamma=1.0, beta=0.99, lam=0.0, alpha=1e-5):
    data = []
    theta = 1e10
    w = np.zeros(X.shape[1])
    D = initialize_D(X,y,w) + lam

    # first step
    i = np.random.choice(X.shape[0], BS)
    g = grad(X,y,w,i,lam=lam)
    w_prev = w[:]
    w = w - gamma * D**-1 * g

    for it in range(T):
        i = np.random.choice(X.shape[0], BS)
        # Calculate gradients
        g_prev = grad(X,y,w_prev,i,lam=lam)
        g = grad(X,y,w,i,lam=lam)
        # estimate hessian diagonal
        z = sample_z(w.shape)
        #D_est = z * hvp(X,y,w,z,i) + lam
        D_est = z * (grad(X,y,w+z,i,lam=lam) - g) + lam
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
        data.append(collect_data(X,y,w,lam))
    return w, np.array(data)


def SGD(X, y, T=10000, BS=1, gamma=0.0002, beta=0.999, lam=0.0, alpha=1e-5, precond="hutchinson"):
    data = []
    w = np.zeros(X.shape[1])
    D = initialize_D(X,y,w) + lam
    for it in range(T):
        # Calculate gradients
        i = np.random.choice(X.shape[0], BS)
        g = grad(X,y,w,i,lam=lam)

        # Diagonal preconditioning
        if precond == "hessian":
            # full hessian
            D = np.diagonal(hessian(X,y,w)) + lam + 1e-10
        elif precond == "hutchinson":
            # estimate hessian diagonal
            z = sample_z(w.shape)
            j = i
            #D_est = z * hvp(X,y,w,z,j) + lam  # why is this bad
            D_est = z * (grad(X,y,w+z,j,lam=lam) - g) + lam
            D = np.abs(beta * D + (1-beta) * D_est)
            D[D < alpha] = alpha

        # Update rule
        w = w - gamma * D**-1 * g

        # Update data
        data.append(collect_data(X,y,w,lam))

    return w, np.array(data)


def SARAH(X, y, T=10, BS=1, gamma=0.2, beta=0.999, lam=0.0, alpha=1e-5, precond="hutchinson"):
    data = []
    wn = np.zeros(X.shape[1])
    D = initialize_D(X,y,w) + lam
    for ep in range(T):
        v = grad(X,y,wn,lam=lam)
        nv0 = np.linalg.norm(v)
        wp = wn[:]
        for it in range(10**10):
            # Calculate gradients
            i = np.random.choice(X.shape[0], BS)
            gn = grad(X,y,wn,i,lam=lam)
            gp = grad(X,y,wp,i,lam=lam)
            v += gn - gp
            wp = wn[:]

            # Diagonal preconditioning
            if precond == "hessian":
                # full hessian
                D = np.diagonal(hessian(X,y,wn)) + lam + 1e-10
            elif precond == "hutchinson":
                # estimate hessian diagonal
                z = sample_z(wn.shape)
                j = i
                #D_est = z * hvp(X,y,wn,z,j) + lam  # why is this bad
                D_est = z * (grad(X,y,wn+z,j,lam=lam) - gn) + lam
                D = np.abs(beta * D + (1-beta) * D_est)
                D[D < alpha] = alpha

            # Update rule
            wn = wn - gamma * D**-1 * v

            # Update data
            data.append(collect_data(X,y,wn,lam))

            # Inner loop stopping criterion
            nv = np.linalg.norm(v)
            if nv < 0.1*nv0 or it > X.shape[0]:
                break

    return wn, np.array(data)


def SVRG(X, y, T=10, BS=1, gamma=0.2, beta=0.999, lam=0.0, alpha=1e-5, precond=None):
    data = []
    w_out = np.zeros(X.shape[1])
    D = initialize_D(X,y,w) + lam
    for ep in range(T):
        g_full = grad(X,y,w_out,lam=lam)
        gnorm0 = np.linalg.norm(g_full)
        w_in = w_out[:]
        for it in range(10**10):
            # Calculate gradients
            i = np.random.choice(X.shape[0], BS)
            g_in = grad(X,y,w_in,i,lam=lam)
            g_out = grad(X,y,w_out,i,lam=lam)
            v = g_in - g_out + g_full

            # Diagonal preconditioning
            if precond == "hessian":
                # full hessian
                D = np.diagonal(hessian(X,y,w_in)) + lam + 1e-10
            elif precond == "hutchinson":
                # estimate hessian diagonal
                z = sample_z(w_in.shape)
                j = i
                #D_est = z * hvp(X,y,wn,z,j) + lam  # why is this bad
                D_est = z * (grad(X,y,w_in+z,j,lam=lam) - g_in) + lam
                D = np.abs(beta * D + (1-beta) * D_est)
                D[D < alpha] = alpha

            # Update rule
            w_in = w_in - gamma * D**-1 * v

            # Update data
            data.append(collect_data(X,y,w_in,lam))

            # Inner loop stopping criterion
            gnorm = np.linalg.norm(g_full)
            if gnorm < 0.1 * gnorm0 or it > X.shape[0]:
                w_out = w_in[:]
                break

    return w_in, np.array(data)


def L_SVRG(X, y, T=10000, BS=1, gamma=0.2, beta=0.999, lam=0.0, alpha=1e-5, precond=None, p=0.99):
    data = []
    w_out = np.zeros(X.shape[1])
    w_in = w_out[:]
    g_full = grad(X,y,w_out,lam=lam)
    D = 1.
    for ep in range(T):
        # Calculate gradients
        i = np.random.choice(X.shape[0], BS)
        g_in = grad(X,y,w_in,i,lam=lam)
        g_out = grad(X,y,w_out,i,lam=lam)
        v = g_in - g_out + g_full

        # Diagonal preconditioning
        if precond == "hessian":
            # full hessian
            D = np.diagonal(hessian(X,y,w_in)) + lam + 1e-10
        elif precond == "hutchinson":
            # estimate hessian diagonal
            z = sample_z(w_in.shape)
            j = i #np.random.choice(X.shape[0], BS)
            #D_est = z * hvp(X,y,wn,z,j) + lam  # why is this bad
            D_est = z * (grad(X,y,w_in+z,j,lam=lam) - g_in) + lam
            D = np.abs(beta * D + (1-beta) * D_est)
            D[D < alpha] = alpha

        # Inner loop stopping criterion is now for updating w_out
        # and comes before the update rule
        if np.random.rand(1)[0] > p:
            w_out = w_in[:]
            g_full = grad(X,y,w_out,lam=lam)

        # Update rule
        w_in = w_in - gamma * D**-1 * v

        # Update data
        data.append(collect_data(X,y,w_in,lam))

    return w_in, np.array(data)

