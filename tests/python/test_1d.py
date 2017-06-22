# Copied from http://www.rueckstiess.net/research/snippets/show/9bd4b418
from numpy import *
from matplotlib import pyplot as plt
from lwpr import LWPR


def testfunc(x):
    return 10*sin(7.8*log(1+x)) / (1 + 0.1*x**2)


Ntr = 500
Xtr = 10 * random.random((Ntr, 1))
Ytr = 5 + testfunc(Xtr) + 0.1 * random.normal(0, 1, (Ntr, 1)) * Xtr

# initialize the LWPR model
model = LWPR(1, 1)
model.init_D = 20 * eye(1)
model.update_D = True
model.init_alpha = 40 * eye(1)
model.meta = False
model.penalty = 1e-4
model.diag_only = True

# train the model
for k in range(20):
    ind = random.permutation(Ntr)
    mse = 0

    for i in range(Ntr):
        yp = model.update(Xtr[ind[i]], Ytr[ind[i]])
        mse = mse + (Ytr[ind[i], :] - yp)**2

    nMSE = mse/Ntr/var(Ytr)
    print "#Data: %5i  #RFs: %3i  nMSE=%5.3f" % (model.n_data, model.num_rfs, nMSE)


# test the model with unseen data
Ntest = 500
Xtest = linspace(0, 10, Ntest)

Ytest = zeros((Ntest, 1))
Conf = zeros((Ntest, 1))

for k in range(500):
    Ytest[k, :], Conf[k, :] = model.predict_conf(array([Xtest[k]]))

plt.plot(Xtr, Ytr, 'r.')

plt.plot(Xtest, Ytest, 'b-')
plt.plot(Xtest, Ytest+Conf, 'c-', linewidth=2)
plt.plot(Xtest, Ytest-Conf, 'c-', linewidth=2)

plt.show()
