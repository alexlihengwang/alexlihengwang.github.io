import numpy as np
import scipy.linalg as la
from realjordan import real_jordan_pair

def padding(A, B, sigmas = None):
  k = int(A.shape[0] / 2)
  if sigmas is None:
    sigmas = np.linspace(-1,1, num = k + 1)

  lambdas = [B[2 * i + 1, 2 * i] + 1j * B[2 * i + 1, 2 * i + 1] for i in range(k)]

  F_matrix = np.zeros((k + 1, k + 1), dtype=np.csingle)
  for i in range(k):
    for j in range(k + 1):
      F_matrix[j, i] = np.prod([lambdas[l] - sigmas[j] for l in range(k) if l != i])
  sigma_h = np.zeros(k + 1, dtype=np.csingle)
  for j in range(k + 1):
    F_matrix[j, -1] = np.prod([lambdas[l] - sigmas[j] for l in range(k)])
    sigma_h[j] = sigmas[j] * F_matrix[j, -1]

  weights = la.solve(F_matrix, sigma_h)
  gamma = weights[:k]
  c = weights[-1]

  t = np.zeros((2 * k, 2))
  for i in range(k):
    sqrt_neg_gamma = np.sqrt(-weights[i])
    t[2 * i, 0] = -1 * sqrt_neg_gamma.imag
    t[2 * i + 1, 1] = sqrt_neg_gamma.imag
    t[2 * i, 1] = sqrt_neg_gamma.real
    t[2 * i + 1, 0] = sqrt_neg_gamma.real
  tau = np.array([
    [-1 * c.imag, c.real],
    [c.real, c.imag]])

  aug_A = la.block_diag(A, np.array([[0,1],[1,0]]))
  aug_B = np.block([[B, t],[t.T, tau]])

  # the following line uses the fact that la.inv(aug_A) == aug_A
  vals, vecs = la.eig(aug_A @ aug_B)
  vals = np.real(vals)
  idx = np.argsort(vals)
  # vals = vals[idx]
  vecs = vecs[:, idx]
  for i in range(k + 1):
    vecs[:, 2 * i], vecs[:, 2 * i + 1] = np.real(vecs[:, 2 * i]), np.imag(vecs[:, 2 * i + 1])
  vecs = np.real(vecs)

  # two-dimensional modifications
  blocked_aug_A = vecs.T @ aug_A @ vecs
  for i in range(k + 1):
    block = blocked_aug_A[2 * i: 2 * i + 2, 2 * i: 2 * i + 2]
    _, block_vecs = la.eigh(block)
    vecs[:, 2 * i: 2 * i + 2] = vecs[:, 2 * i: 2 * i + 2] @ block_vecs

  return t, tau, vecs

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
    t, tau, P2 = padding(PAP_complex, PBP_complex)

    aug_PAP = la.block_diag(PAP, np.array([[0,1],[1,0]]))

    aug_PBP = np.block([
      [PBP_real, np.zeros((r, n - r)), np.zeros((r, 2))],
      [np.zeros((n - r, r)), PBP_complex, t],
      [np.zeros((2,r)), t.T, tau]])

    aug_P = la.block_diag(P, np.eye(2))
    aug_P_inv = la.block_diag(la.inv(P), np.eye(2))

    aug_A = aug_P_inv.T @ aug_PAP @ aug_P_inv
    aug_B = aug_P_inv.T @ aug_PBP @ aug_P_inv

    Q = aug_P @ la.block_diag(np.eye(r), P2)
    for i in range(n + 2):
      Q[:, i] /= la.norm(Q[:, i])
      # Q[:, i] /= np.sqrt(np.abs(Q[:,i].T @ aug_A @ Q[:, i]))

    return aug_A, aug_B, Q, True
  else:
    _, Q = la.eig(la.inv(A) @ B)
    Q = np.real(Q)
    for i in range(n):
      Q[:, i] /= la.norm(Q[:, i])
      # Q[:, i] /= np.sqrt(np.abs(Q[:,i].T @ A @ Q[:, i]))

    return A, B, Q, False




