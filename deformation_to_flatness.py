import numpy as np
d = np.loadtxt("dz.txt", skiprows=1)   # columns: node, x, y, z, dz
x, y, w = d[:,1], d[:,2], d[:,4]
m = np.hypot(x, y) <= 0.0508           # r <= 50.8 mm = 4 inch region
A = np.c_[x[m], y[m], np.ones(m.sum())]
c, *_ = np.linalg.lstsq(A, w[m], rcond=None)
r = w[m] - A @ c
print("flatness peak-to-valley (um):", (r.max() - r.min()) * 1e6)
print(len(x), len(set(zip(x.round(6), y.round(6)))))