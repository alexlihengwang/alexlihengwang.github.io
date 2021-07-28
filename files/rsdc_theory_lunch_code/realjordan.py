import numpy as np
import numpy.linalg as la

def real_jordan_pair(A,B):
  n = A.shape[0]

  vals, vecs = la.eig(la.inv(A) @ B)
  complexQ = [np.abs(np.imag(val)) >= np.finfo(float).eps for val in vals]
  realpart = np.real(vals)
  idx = np.lexsort((realpart, complexQ))
  vals = vals[idx]
  vecs = vecs[:,idx]

  i = 0
  while i < n:
    if np.abs(np.imag(vals[i])) >= np.finfo(float).eps:
      vecs[:, i], vecs[:, i + 1] = (vecs[:, i] + vecs[:, i + 1]) / 2, 1j * (vecs[:,i] - vecs[:, i + 1]) / 2
      beta = vecs[:, i].conj().T @ A @ vecs[:, i]
      alpha = vecs[:, i].conj().T @ A @ vecs[:, i + 1]
      if beta != 0:
        y = beta / np.sqrt(2 * (alpha ** 2 + beta ** 2) * (alpha + np.sqrt(alpha ** 2 + beta ** 2)))
        x = (y / beta) * (-alpha - np.sqrt(alpha ** 2 + beta ** 2))
      elif beta == 0 and alpha < 0:
        y = 0
        x = np.sqrt(alpha)
      else:
        x = 0
        y = np.sqrt(-alpha)

      vecs[:, i], vecs[:, i + 1] = x * vecs[:, i] + y * vecs[:, i + 1], -y * vecs[:, i] + x * vecs[:, i + 1]
      i += 1
    else:
      normalization = np.sqrt(np.abs(vecs[:, i].conj().T @ A @ vecs[:, i]))
      vecs[:, i] /= normalization
    i += 1

  P = np.real(vecs)

  return P, n - sum(complexQ)