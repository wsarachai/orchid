import numpy as np

def example_geometric_margin(w, b, x, y):
  norm = np.linalg.norm(w)
  a = np.dot(w/norm, x) + b/norm
  result = y * a
  return result

def geometric_margin(w, b, X, y):
  a = [example_geometric_margin(w, b, x, y[i]) for i, x in enumerate(X)]
  return np.min(a)


def constraint(w, b, x, y):
  return y * (np.dot(w, x) + b)


def hard_constraint_is_satisfied(w, b, x, y):
  return constraint(w, b, x, y) >= 1


def soft_constraint_is_satisfied(w, b, x, y, zeta):
  return constraint(w, b, x, y) >= 1 - zeta


# Transform a two-dimensional vector x into a three-dimensional vector.
def transform(x):
  return [x[0]**2, np.sqrt(2)*x[0]*x[1], x[1]**2]


#def polynomial_kernel(a, b):
#  return a[0]**2 * b[0]**2 + 2*a[0]*b[0]*a[1]*b[1] + a[1]**2 * b[1]**2


def polynomial_kernel(a, b, degree, constant=0):
  result = sum([a[i] * b[i] for i in range(len(a))]) + constant
  return pow(result, degree)


def kernel(x1, x2):
  return np.dot(x1, x2.T)


def objective_function_to_minimize(X, y, a, kernel):
  m, n = np.shape(X)
  return 1 / 2 * np.sum([a[i] * a[j] * y[i] * y[j] * kernel(X[i, :], X[j, :])
                         for j in range(m)
                         for i in range(m)]) \
         - np.sum([a[i] for i in range(m)])

x1 = [3,6]
x2 = [10,10]
x1_3d = transform(x1)
x2_3d = transform(x2)

print(np.dot(x1_3d,x2_3d))
#print(polynomial_kernel(x1, x2))
print(polynomial_kernel(x1, x2, degree=2))