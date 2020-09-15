from scipy.optimize import minimize
import numpy as np

def f(a,b,c,d,ai,bi,ci,di):
  return a**2 + b**2 + c**2 + d**2 + ai**2 + bi**2 + ci**2 + di**2

def g1(a,b,c,d,ai,bi,ci,di):
  return ai + di

def g2(a,b,c,d,ai,bi,ci,di):
  return 2 * (a - d) * (ai - di) + (b * ci + bi * c) + (a - d)

def g3(a,b,c,d,ai,bi,ci,di):
  return (a - d)**2 - (ai - di)**2 + 4 * (b * c - bi * ci) + 4 (di - ai) - 4

if __name__ == '__main__':
  b = [[-10,10]] * 8
  cons = [
    {
      'type': 'eq',
      'fun': g1,
    },
    {
      'type': 'eq',
      'fun': g2,
    },
    {
      'type': 'ineq',
      'fun': g3,
    }
  ]
  minimize(f, np.zeros(8), bounds=b, constraints=cons)