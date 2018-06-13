import numpy as np

c_key_dim = 3

def create_rot():
	nd_rnd = (np.random.rand(c_key_dim, c_key_dim) -0.5) * 2.
	q,r = np.linalg.qr(nd_rnd)
	return r

a = np.random.rand(5, 3)
q_or_r = create_rot()
b = np.matmul(a, q_or_r)
pass

