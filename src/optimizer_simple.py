
from loss import F, grad, hessian, hvp
import numpy as np

def collect_data(X,y,w,lam=0.0):
    loss = F(X,y,w)
    g_norm = np.linalg.norm(grad(X,y,w))**2
    error = np.mean(X.dot(w)*y < 0)
    return (loss, g_norm, error)

def sample_z(size):
    return 2 * np.random.randint(0,2,size) - 1

def norm_scaled(x,D):
    return np.sqrt(np.sum(x * D * x))


def SGD(X, y, gamma=0.02, BS=1, T=10000, lam=0.0):
    data = []
    w = np.zeros(X.shape[1])
    for it in range(T):
        i = np.random.choice(X.shape[0], BS)
        g = grad(X,y,w,i,lam=lam)
        w = w - gamma * g
        data.append(collect_data(X,y,w,lam))
    return w, np.array(data)


def SARAH(X, y, gamma=0.2, BS=1, epochs=10, lam=0.0):
    data = []
    wn = np.zeros(X.shape[1])
    for ep in range(epochs):
        v = grad(X,y,wn,lam=lam)
        nv0 = np.linalg.norm(v)
        wp = wn[:]
        for it in range(10**10):
            i = np.random.choice(X.shape[0], BS)
            gn = grad(X,y,wn,i,lam=lam)
            gp = grad(X,y,wp,i,lam=lam)
            v += gn - gp
            wp = wn[:]
            wn = wn - gamma * v
            data.append(collect_data(X,y,wn,lam))
            nv = np.linalg.norm(v)
            if nv < 0.1*nv0 or it > X.shape[0]:
                break
                
    return wn, np.array(data)


def SVRG(X, y, gamma=0.2, BS=1, epochs=10, lam=0.0):
    data = []
    w_out = np.zeros(X.shape[1])
    for ep in range(epochs):
        g_full = grad(X,y,w_out,lam=lam)
        gnorm0 = np.linalg.norm(g_full)
        w_in = w_out[:]
        for it in range(10**10):
            i = np.random.choice(X.shape[0], BS)
            g_in = grad(X,y,w_in,i,lam=lam)
            g_out = grad(X,y,w_out,i,lam=lam)
            w_in = w_in - gamma * (g_in - g_out + g_full)
            
            data.append(collect_data(X,y,w_in,lam))
            gnorm = np.linalg.norm(g_full)
            if gnorm < 0.1 * gnorm0 or it > X.shape[0]:
                w_out = w_in[:]
                break
                
    return w_in, np.array(data)


def SGD_Hessian(X, y, gamma=0.0002, BS=1, T=10000, lam=0.0):
    data = []
    w = np.zeros(X.shape[1])
    for it in range(T):
        i = np.random.choice(X.shape[0], BS)
        g = grad(X,y,w,i,lam=lam)
        # full hessian
        D = np.diagonal(hessian(X,y,w)) + lam + 1e-10
        w = w - gamma * D**-1 * g
        ##
        data.append(collect_data(X,y,w,lam))
    return w, np.array(data)


def SARAH_Hessian(X, y, gamma=0.002, BS=1, epochs=10, lam=0.0):
    data = []
    wn = np.zeros(X.shape[1])
    for ep in range(epochs):
        v = grad(X,y,wn,lam=lam)
        nv0 = np.linalg.norm(v)
        wp = wn[:]
        for it in range(10**10):
            i = np.random.choice(X.shape[0], BS)
            gn = grad(X,y,wn,i,lam=lam)
            gp = grad(X,y,wp,i,lam=lam)
            v += gn - gp
            wp = wn[:]
            # full hessian
            D = np.diagonal(hessian(X,y,wn)) + lam + 1e-10
            wn = wn - gamma * D**-1 * v

            data.append(collect_data(X,y,wn,lam))
            nv = np.linalg.norm(v)
            if nv < 0.1*nv0 or it > X.shape[0]:
                break
                
    return wn, np.array(data)


def OASIS(X, y, T=10000, BS=1, gamma=1.0, beta=0.99, lam=0.0, alpha=1e-5):
    data = []
    w = np.zeros(X.shape[1])
    D = np.ones_like(w)
    theta = 1e10
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


def SARAH_OASIS(X, y, gamma=0.2, beta=0.999, alpha=1e-5, BS=1, epochs=10, lam=0.0):
    data = []
    wn = np.zeros(X.shape[1])
    D = np.ones_like(wn)
    for ep in range(epochs):
        v = grad(X,y,wn,lam=lam)
        nv0 = np.linalg.norm(v)
        wp = wn[:]
        for it in range(10**10):
            i = np.random.choice(X.shape[0], BS)
            gn = grad(X,y,wn,i,lam=lam)
            gp = grad(X,y,wp,i,lam=lam)
            v += gn - gp
            wp = wn[:]
            # estimate hessian diagonal
            z = sample_z(wn.shape)
            #D_est = z * hvp(X,y,wn,z,i) + lam  # why is this bad
            D_est = z * (grad(X,y,wn+z,i,lam=lam) - gn) + lam
            D = np.abs(beta * D + (1-beta) * D_est)
            D[D < alpha] = alpha
            # update rule
            wn = wn - gamma * D**-1 * v
            data.append(collect_data(X,y,wn,lam))
            nv = np.linalg.norm(v)
            if nv < 0.1*nv0 or it > X.shape[0]:
                break

    return wn, np.array(data)


