import numpy as np
import scipy.linalg as la
from realjordan import real_jordan_pair

def padding(A, B, sigmas = None):
  k = int(A.shape[0] / 2)
  if sigmas is None:
    sigmas = np.linspace(-1,1, num = 2 * k + 1)

  a = [B[2 * i, 2 * i] for i in range(k)]
  b = [B[2 * i, 2 * i + 1] for i in range(k)]

  F_matrix = np.zeros((2 * k + 1, 2 * k + 1))
  for i in range(k):
    for j in range(2 * k + 1):
      F_matrix[j, 2 * i] = np.prod([(b[l] - sigmas[j]) ** 2 + a[l] ** 2 for l in range(k) if l != i])
      F_matrix[j, 2 * i + 1] = sigmas[j] * F_matrix[j, 2 * i]
  sigma_h = np.zeros(2 * k + 1)
  for j in range(2 * k + 1):
    F_matrix[j, -1] = F_matrix[j, 0] * ((b[0] - sigmas[j]) ** 2 + a[0] ** 2)
    sigma_h[j] = sigmas[j] * F_matrix[j, -1]

  weights = la.solve(F_matrix, sigma_h)
  tau = weights[-1]
  t = np.zeros(2 * k)
  for i in range(k):
    if np.abs(weights[2 * i + 1]) <= np.sqrt(np.finfo(float).eps):
      if (weights[2 * i] / a[i] >= 0):
        t[2 * i + 1] = np.sqrt(weights[2 * i] / a[i])
        t[2 * i] = 0
      else:
        t[2 * i] = np.sqrt(-weights[2 * i] / a[i])
        t[2 * i + 1] = 0
    else:
      quan = (weights[2 * i] + weights[2 * i + 1] * b[i]) / a[i]
      t[2 * i + 1] = np.sqrt((quan +  np.sqrt(quan ** 2 + weights[2 * i + 1] ** 2)) / 2)
      t[2 * i] = weights[2 * i + 1] / (2 * t[2 * i + 1])

  return t, tau

def augment(A,B):
  n = A.shape[0]
  P, r = real_jordan_pair(A,B)

  PAP = P.T @ A @ P
  PBP = P.T @ B @ P
  PAP_real = PAP[:r, :r]
  PAP_complex = PAP[r:, r:]
  PBP_real = PBP[:r, :r]
  PBP_complex = PBP[r:, r:]

  if r < n:
    t, tau = padding(PAP_complex, PBP_complex)

    aug_PAP = la.block_diag(PAP,np.eye(1))

    aug_PBP = la.block_diag(PBP_real,
      np.block([[PBP_complex, t.reshape((n - r, 1))],
      [t.reshape((1, n - r)), tau]]))

    aug_P = la.block_diag(P, np.eye(1))
    aug_P_inv = la.block_diag(la.inv(P), np.eye(1))

    aug_A = aug_P_inv.T @ aug_PAP @ aug_P_inv
    aug_B = aug_P_inv.T @ aug_PBP @ aug_P_inv

    _, Q = la.eig(la.inv(aug_A) @ aug_B)
    Q = np.real(Q)
    for i in range(n + 1):
      Q[:, i] /= la.norm(Q[:, i])
      # Q[:, i] /= np.sqrt(np.abs(Q[:, i].T @ aug_A @ Q[:, i]))

    return aug_A, aug_B, Q, True
  else:
    _, Q = la.eig(la.inv(A) @ B)
    Q = np.real(Q)
    for i in range(n):
      Q[:, i] /= la.norm(Q[:, i])
      # Q[:, i] /= np.sqrt(np.abs(Q[:, i].T @ A @ Q[:, i]))

    return A, B, Q, False

if __name__ == '__main__':
  from random_model import random_GOE
  import json

  n = 3
  A = random_GOE(n)
  B = random_GOE(n)

  augA, augB, Q, _ = augment(A, B)
  augA /= la.norm(augA, ord=2)
  augB /= la.norm(augB, ord=2)
  Q /= np.sqrt(max(la.norm(Q.T @ augA @ Q, ord=2), la.norm(Q.T @ augB @ Q, ord='nuc')))

  data = {
    'A': augA.tolist(),
    'B': augB.tolist(),
    'Q': Q.tolist() 
  }
  json_data = json.dumps(data)
  print(json_data)





