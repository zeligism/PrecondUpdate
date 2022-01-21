
from loss import F, grad, hessian, hvp
import numpy as np

def SGD(X, y, gamma=0.02, BS=1, T=10000):
    data = []
    w = np.zeros(X.shape[1])
    for it in range(T):
        i = np.random.choice(X.shape[0], BS)
        w = w - gamma * grad(X,y,w,i)
        data.append((F(X,y,w),
                    np.linalg.norm(grad(X,y,w))**2,
                    np.mean(X.dot(w)*y < 0)))
    return w, np.array(data)


def SARAH(X, y, gamma=0.2, BS=1, epochs=10):
    data = []
    wn = np.zeros(X.shape[1])
    for ep in range(epochs):
        v = grad(X,y,wn)
        nv0 = np.linalg.norm(v)
        wp = wn[:]
        for it in range(10**10):
            i = np.random.choice(X.shape[0], BS)
            gn = grad(X,y,wn,i)
            gp = grad(X,y,wp,i)
            v += gn - gp
            wp = wn[:]
            wn = wn - gamma * v
            data.append((F(X,y,wn),
                        np.linalg.norm(grad(X,y,wn))**2,
                        np.mean(X.dot(wn)*y < 0)))
            nv = np.linalg.norm(v)
            if nv < 0.1*nv0 or it > X.shape[0]:
                break
                
    return wn, np.array(data)


def SGD_Hessian(X, y, gamma=0.02, BS=1, T=10000, lam=0.0, full_hessian=False):
    data = []
    w = np.zeros(X.shape[1])
    for it in range(T):
        i = np.random.choice(X.shape[0], BS)
        # hessian
        g = grad(X,y,w,i) + lam * np.linalg.norm(w)
        iH = None if full_hessian else i
        H = hessian(X,y,w,iH) + lam * np.eye(w.shape[0]) # full Hessian
        w = w - gamma * np.diagonal(H+1e-10)**-1 * g
        ##
        data.append((F(X,y,w) + 0.5*lam*np.linalg.norm(w)**2,
                    np.linalg.norm(grad(X,y,w))**2,
                    np.mean(X.dot(w)*y < 0)))
    return w, np.array(data)


def SARAH_Hessian(X, y, gamma=0.2, BS=1, epochs=10, lam=0.0, full_hessian=False):
    data = []
    wn = np.zeros(X.shape[1])
    for ep in range(epochs):
        v = grad(X,y,wn)
        nv0 = np.linalg.norm(v)
        wp = wn[:]
        for it in range(10**10):
            i = np.random.choice(X.shape[0], BS)
            gn = grad(X,y,wn,i) + lam * np.linalg.norm(wn)
            gp = grad(X,y,wp,i) + lam * np.linalg.norm(wp)
            v += gn - gp
            wp = wn[:]
            # hessian
            iH = None if full_hessian else i
            H = hessian(X,y,wn,iH) + lam * np.eye(wn.shape[0])
            wn = wn - gamma * np.diagonal(H+1e-10)**-1 * v
            data.append((F(X,y,wn) + 0.5*lam*np.linalg.norm(wn)**2,
                        np.linalg.norm(grad(X,y,wn))**2,
                        np.mean(X.dot(wn)*y < 0)))
            nv = np.linalg.norm(v)
            if nv < 0.1*nv0 or it > X.shape[0]:
                break
                
    return wn, np.array(data)



def sample_z(size):
    return 2 * np.random.randint(0,2,size) - 1

def norm_scaled(x,D):
    return np.sqrt(np.sum(x * D * x))

def OASIS(X, y, gamma=1.0, beta=0.99, alpha=1e-5, BS=1, T=10000,):
    data = []
    w = np.zeros(X.shape[1])
    D = np.ones_like(w)
    theta = 1e10
    #D = hessian(X,y,w)
    # first step
    i = np.random.choice(X.shape[0], BS)
    g = grad(X,y,w,i)
    w_prev = w[:]
    w = w - gamma * D**-1 * g
    for it in range(T):
        i = np.random.choice(X.shape[0], BS)
        # Calculate gradients
        g_prev = grad(X,y,w_prev,i)
        g = grad(X,y,w,i)
        # estimate hessian diagonal
        z = sample_z(w.shape)
        #D_est = z * hvp(X,y,w,z,i)
        D_est = z * (grad(X,y,w+z,i) - g)
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
        data.append((F(X,y,w),
                    np.linalg.norm(grad(X,y,w))**2,
                    np.mean(X.dot(w)*y < 0),))
    return w, np.array(data)


def SARAH_AdaHessian(X, y, gamma=0.2, beta=0.999, alpha=1e-5, BS=1, epochs=10, lam=0.0):
    data = []
    wn = np.zeros(X.shape[1])
    D = np.ones_like(wn)
    for ep in range(epochs):
        v = grad(X,y,wn)
        nv0 = np.linalg.norm(v)
        wp = wn[:]
        for it in range(10**10):
            i = np.random.choice(X.shape[0], BS)
            gn = grad(X,y,wn,i) + lam * np.linalg.norm(wn)
            gp = grad(X,y,wp,i) + lam * np.linalg.norm(wp)
            v += gn - gp
            wp = wn[:]
            # estimate hessian diagonal
            z = sample_z(wn.shape)
            #D_est = z * hvp(X,y,wn,z,i) # why is this bad
            D_est = z * (grad(X,y,wn+z,i) - gn) + lam
            D = np.abs(beta * D + (1-beta) * D_est)
            D[D < alpha] = alpha
            # update rule
            wn = wn - gamma * D**-1 * v
            data.append((F(X,y,wn) + 0.5*lam*np.linalg.norm(wn)**2,
                        np.linalg.norm(grad(X,y,wn))**2,
                        np.mean(X.dot(wn)*y < 0)))
            nv = np.linalg.norm(v)
            if nv < 0.1*nv0 or it > X.shape[0]:
                break

    return wn, np.array(data)

