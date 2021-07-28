import numpy.random as npr
import numpy as np
import scipy.linalg as la
from scipy.stats import ortho_group, special_ortho_group

def random_instance(n, k, m):
  U = ortho_group.rvs(n)

  blocks_1 = []
  blocks_1.append(np.diag(np.sign(npr.randn(n - 2 * k))))
  if k >= 1:
    blocks_1.append(
      np.kron(
        np.eye(k),
        np.array([[0, 1],[1, 0]])
      )
    )
  A1 = U.T @ la.block_diag(*blocks_1) @ U

  blocks_2 = []
  blocks_2.append(np.diag(npr.randn(n - 2 * k)))
  for i in range(k):
    a = npr.randn()
    b = npr.randn()
    blocks_2.append([[b, a], [a, -b]])
  A2 = U.T @ la.block_diag(*blocks_2) @ U

  # zero
  b1 = np.zeros(n)
  b2 = np.zeros(n)

  # N(0,1)
  L = npr.randn(m,n)

  return A1, A2, b1, b2, L



