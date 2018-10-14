import numpy as np

eps = 1e-8

def pos(Y):
    return np.sum(np.round(Y)).astype(np.float32)

def neg(Y):
    return np.sum(np.logical_not(np.round(Y))).astype(np.float32)

def PR(Y):
    return pos(Y) / (pos(Y) + neg(Y))

def NR(Y):
    return neg(Y) / (pos(Y) + neg(Y))

def TP(Y, Ypred):
    return np.sum(np.multiply(Y, np.round(Ypred))).astype(np.float32)

def FP(Y, Ypred):
    return np.sum(np.multiply(np.logical_not(Y), np.round(Ypred))).astype(np.float32)

def TN(Y, Ypred):
    return np.sum(np.multiply(np.logical_not(Y), np.logical_not(np.round(Ypred)))).astype(np.float32)

def FN(Y, Ypred):
    return np.sum(np.multiply(Y, np.logical_not(np.round(Ypred)))).astype(np.float32)

def FP_soft(Y, Ypred):
    return np.sum(np.multiply(np.logical_not(Y), Ypred)).astype(np.float32)

def FN_soft(Y, Ypred):
    return np.sum(np.multiply(Y, 1 - Ypred)).astype(np.float32)

def TPR(Y, Ypred):
    return TP(Y, Ypred) / pos(Y)

def FPR(Y, Ypred):
    return FP(Y, Ypred) / neg(Y)

def TNR(Y, Ypred):
    return TN(Y, Ypred) / neg(Y)

def FNR(Y, Ypred):
    return FN(Y, Ypred) / pos(Y)

def FPR_soft(Y, Ypred):
    return FP_soft(Y, Ypred) / neg(Y)

def FNR_soft(Y, Ypred):
    return FN_soft(Y, Ypred) / pos(Y)

def errRate(Y, Ypred):
    return (FP(Y, Ypred) + FN(Y, Ypred)) / float(Y.shape[0])

def accuracy(Y, Ypred):
    return 1 - errRate(Y, Ypred)

def subgroup(fn, mask, Y, Ypred=None, A=None):
    m = np.greater(mask, 0.5).flatten()
    Yf = Y.flatten()
    if Ypred is None and A is None:
        return fn(Yf[m])
    elif not Ypred is None and A is None: #two-argument functions
        Ypredf = Ypred.flatten()
        return fn(Yf[m], Ypredf[m])
    else: #three-argument functions
        Ypredf = Ypred.flatten()
        Af = A.flatten()
        return fn(Yf[m], Ypredf[m], Af[m])

def DI_FP(Y, Ypred, A):
    fpr1 = subgroup(FPR, A, Y, Ypred)
    fpr0 = subgroup(FPR, 1 - A, Y, Ypred)
    return abs(fpr1 - fpr0)

def DI_FN(Y, Ypred, A):
    fnr1 = subgroup(FNR, A, Y, Ypred)
    fnr0 = subgroup(FNR, 1 - A, Y, Ypred)
    return abs(fnr1 - fnr0)

def DI_FP_soft(Y, Ypred, A):
    fpr1 = subgroup(FPR_soft, A, Y, Ypred)
    fpr0 = subgroup(FPR_soft, 1 - A, Y, Ypred)
    return abs(fpr1 - fpr0)

def DI_FN_soft(Y, Ypred, A):
    fnr1 = subgroup(FNR_soft, A, Y, Ypred)
    fnr0 = subgroup(FNR_soft, 1 - A, Y, Ypred)
    return abs(fnr1 - fnr0)

def DI(Y, Ypred, A):
    return (DI_FN(Y, Ypred, A) + DI_FP(Y, Ypred, A)) * 0.5

def DI_soft(Y, Ypred, A):
    return (DI_FN_soft(Y, Ypred, A) + DI_FP_soft(Y, Ypred, A)) * 0.5

def DP(Ypred, A): 
    return abs(subgroup(PR, A, Ypred) - subgroup(PR, 1 - A, Ypred))


def switch(x0, x1, s):
    return np.multiply(x0, 1. - s) + np.multiply(x1, s)

'''IDK specific metrics'''
def idkRate(idk):
    return PR(idk)

def rejErrRate(Y, Ypred, idk):
    return subgroup(errRate, 1. - idk, Y, Ypred)

