import gurobipy as gp
import numpy as np
import numpy.linalg as nla
import scipy.linalg as la
import one_rsdc
import two_rsdc
import random_model
import random
import sys

def solve_qp(A1, A2, b1,b2,L,H=None, timeout=300):
  """ Use gurobi to solve:
  min x^t A1 x + b1^t x
  s.t x^t A2 x + b2^t x <= 0
      L x <= 1
      H x == 0
  """
  n = A1.shape[0]

  lbounds = np.zeros(n)
  ubounds = np.zeros(n)

  for i in range(n):
    try:
      m = gp.Model('bound')
      m.params.OutputFlag = 0
      x = m.addMVar(n, lb=-float('inf'), ub=float('inf'))
      m.addConstr(L @ x <= 1)
      if H is not None:
        m.addConstr(H @ x == 0)

      m.setObjective(x[i])
      m.optimize()
      lbounds[i] = m.objVal
      
      m.setObjective(-1 * x[i])
      m.optimize()
      ubounds[i] = -1 * m.objVal

    except gp.GurobiError as e:
      print('Error code ' + str(e.errno) + ': ' + str(e))
      print("Polyhedron unbounded")
      return None

    except AttributeError:
      print("Polyhedron unbounded")
      return None

  try:
    m = gp.Model('polyqp')
    m.params.OutputFlag = 0
    m.params.NonConvex = 2
    m.params.TimeLimit = timeout

    x = m.addMVar(n,lb=lbounds, ub=ubounds)

    m.setObjective(x @ A1 @ x + 2 * b1 @ x)
    m.addConstr(x @ A2 @ x + 2 * b2 @ x <= 0)
    m.addConstr(L @ x <= 1)
    if H is not None:
      m.addConstr(H @ x == 0)

    m.optimize()
    return m

  except gp.GurobiError as e:
    print('Error code ' + str(e.errno) + ': ' + str(e))
    return None

  except AttributeError:
    print('Encountered an attribute error')
    return None

def as_is_QCQP(A1,A2,b1,b2,L,timeout=300):
  return solve_qp(A1,A2,b1,b2,L,timeout=timeout)

def n_rsdc_QCQP(A1,A2,b1,b2,L,timeout=300):
  m, n = L.shape
  _, P1 = la.eigh(A1)
  _, P2 = la.eigh(P1.T @ A2 @ P1)
  tilde_A1 = la.block_diag(P1.T @ A1 @ P1, np.zeros((n,n)))
  tilde_A2 = la.block_diag(np.zeros((n,n)), P2.T @ P1.T @ A2 @ P1 @ P2)
  tilde_A1, tilde_A2 = np.diag(np.diag(tilde_A1)), np.diag(np.diag(tilde_A2))
  tilde_b1 = np.zeros(2 * n)
  tilde_b2 = np.zeros(2 * n)
  tilde_b1[:n] = P1.T @ b1
  tilde_b2[:n] = P1.T @ b2
  tilde_L = np.block([L @ P1,np.zeros((m,n))])
  H = np.block([np.eye(n), -1 * P2])
  return solve_qp(tilde_A1, tilde_A2, tilde_b1,tilde_b2,tilde_L,H,timeout=timeout)

def one_rsdc_QCQP(A1,A2,b1,b2,L,timeout=300):
  m, n = L.shape
  aug_A1, aug_A2, P, flag = one_rsdc.augment(A1, A2)
  if flag:
    tilde_A1 = P.T @ aug_A1 @ P
    tilde_A2 = P.T @ aug_A2 @ P
    tilde_A1, tilde_A2 = np.diag(np.diag(tilde_A1)), np.diag(np.diag(tilde_A2))
    aug_b1 = np.zeros(n+1)
    aug_b2 = np.zeros(n+1)
    aug_b1[:n] = b1
    aug_b2[:n] = b2
    tilde_b1 = P.T @ aug_b1
    tilde_b2 = P.T @ aug_b2
    tilde_L = np.block([L, np.zeros((m,1))]) @ P
    H = P[n:n+1,:]
  else:
    tilde_A1 = P.T @ aug_A1 @ P
    tilde_A2 = P.T @ aug_A2 @ P
    tilde_A1, tilde_A2 = np.diag(np.diag(tilde_A1)), np.diag(np.diag(tilde_A2))
    tilde_b1 = P.T @ b1
    tilde_b2 = P.T @ b2
    tilde_L = L @ P
    H = None

  return solve_qp(tilde_A1, tilde_A2, tilde_b1,tilde_b2,tilde_L,H,timeout=timeout), P

def two_rsdc_QCQP(A1,A2,b1,b2,L,timeout=300):
  m, n = L.shape
  aug_A1, aug_A2, P, flag = two_rsdc.augment(A1, A2)
  if flag:
    tilde_A1 = P.T @ aug_A1 @ P
    tilde_A2 = P.T @ aug_A2 @ P
    tilde_A1, tilde_A2 = np.diag(np.diag(tilde_A1)), np.diag(np.diag(tilde_A2))
    aug_b1 = np.zeros(n+2)
    aug_b2 = np.zeros(n+2)
    aug_b1[:n] = b1
    aug_b2[:n] = b2
    tilde_b1 = P.T @ aug_b1
    tilde_b2 = P.T @ aug_b2
    tilde_L = np.block([L, np.zeros((m,2))]) @ P
    H = P[n:n+2,:]
  else:
    tilde_A1 = P.T @ aug_A1 @ P
    tilde_A2 = P.T @ aug_A2 @ P
    tilde_A1, tilde_A2 = np.diag(np.diag(tilde_A1)), np.diag(np.diag(tilde_A2))
    tilde_b1 = P.T @ b1
    tilde_b2 = P.T @ b2
    tilde_L = L @ P
    H = None
    
  return solve_qp(tilde_A1, tilde_A2, tilde_b1,tilde_b2,tilde_L,H,timeout=timeout), P

if __name__ == '__main__':
  n = 15
  m = 100
  num_instances = 10
  timeout = 600

  for k in [0, 1, 2, 3]:
    print('')
    print('k =', k, flush=True)

    headers = [
      "As-is time", "1-RSDC time", "2-RSDC time", "n-RSDC time",
      "1-RSDC cond", "2-RSDC cond",
      "As-is val", "1-RSDC val", "2-RSDC val", "n-RSDC val"]
    headers = [head.rjust(15, " ") for head in headers]
    headers.insert(6,'  |  ')
    headers.insert(4,'  |  ')
    print(''.join(headers), flush=True)

    for _ in range(num_instances):    
      data = [None] * 10

      A1, A2, b1, b2, L = random_model.random_instance(n, k, m)
      
      star_idx = None
      best_time = float('inf')

      # print('running as_is_qcqp', file=sys.stderr)
      model = as_is_QCQP(A1,A2,b1,b2,L,timeout=timeout)
      data[0] = model.RunTime
      data[6] = model.ObjVal
      if model.status == 2:
        if model.RunTime <= best_time:
          best_time = model.RunTime
          star_idx = 0


      # print('running one_rsdc_QCQP', file=sys.stderr)
      model, P = one_rsdc_QCQP(A1,A2,b1,b2,L,timeout=timeout)
      data[1] = model.RunTime
      data[4] = nla.cond(P)
      data[7] = model.ObjVal
      if model.status == 2:
        if model.RunTime <= best_time:
          best_time = model.RunTime
          star_idx = 1

      # print('running two_rsdc_QCQP', file=sys.stderr)
      model, P = two_rsdc_QCQP(A1,A2,b1,b2,L,timeout=timeout)
      data[2] = model.RunTime
      data[5] = nla.cond(P)
      data[8] = model.ObjVal
      if model.status == 2:
        if model.RunTime <= best_time:
          best_time = model.RunTime
          star_idx = 2

      # print('running n_rsdc_QCQP', file=sys.stderr)
      model = n_rsdc_QCQP(A1,A2,b1,b2,L,timeout=timeout)
      data[3] = model.RunTime
      data[9] = model.ObjVal
      if model.status == 2:
        if model.RunTime <= best_time:
          best_time = model.RunTime
          star_idx=3

      for j in range(10):
        if j == star_idx:
          data[j] = "    * %9.3e" % data[j]
        else:
          data[j] = "%15.3e" % data[j]

      data.insert(6,'  |  ')
      data.insert(4,'  |  ')
      print(''.join(data), flush=True)


























