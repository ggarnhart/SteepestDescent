from numpy import linalg as LA  # let's make the linear algebra stuff a lil easier
import numpy
from math import log


class Quadratic(object):
    def f(self, x):
        return .5 * (x[0]**2 + 10*x[1]**2)

    def g(self, x):
        return numpy.array([x[0], 10*x[1]]).T


class Logistic_Regression(object):
    # the negative log-likelihood for logisitic regression:
    def __init__(self, X_train, y_train):
        # this constructor stores the training data and also prepends a column of ones.
        # Each row of X_train corresponds to a single training case
        self.num_cases = X_train.shape[0]
        e = numpy.ones((self.num_cases, 1))
        self.X_train = numpy.append(e, X_train, axis=1)
        self.y_train = y_train

    def f(self, w):
        # Return the negative log-likelihood. Remember that we wish to maximize the log-likelihood, so we minimize its negative
        z = self.X_train.dot(w)
        p = 1 / (1 + numpy.exp(-z))
        log_likelihood = 0
        for i in range(0, self.num_cases):
            if y_train[i] == 1:
                log_likelihood -= log(p[i])
            else:
                log_likelihood -= log(1-p[i])
        return log_likelihood

    # ah yes this does nothing rn so it makes sense that it does not work.
    def g(self, w):

        for i in range(0, self.num_cases):
            g = X_train[0]
            print("!",  g[i:])
        vec = X_train.T
        current_itr = vec
        for i in range(0, self.num_cases):

            if y_train[i] == 1:
                numerator = numpy.exp(self.X_train.dot(w))
                denominator = 1 + numpy.exp(self.X_train.dot(w))
                frac = numerator / denominator
                add_to_cur = frac * vec
                current_itr = numpy.add(current_itr, add_to_cur)

            else:
                numerator = - numpy.exp(self.X_train.dot(w))
                denominator = 1 + numpy.exp(self.X_train.dot(w))
                frac = numerator / denominator
                add_to_cur = frac * vec
                current_itr = numpy.add(current_itr, add_to_cur)

        return current_itr

# Steepest descent ofr minimaization using a linesearch.


def gd0(obj, x0, use_sgd=False, max_it=100, abs_grad_tol=1.0e-04, rel_grad_tol=1.0e-04, abs_stepsize_tol=1.0e-06, rel_stepsize_tol=1.0e-06, armijo_factor=1.0e-04):
    # x0 is the starting point for the minimzation, as a numpy column vector
    x_c = x0
    f_c = obj.f(x_c)
    g_c = obj.g(x_c)

    for i in range(0, max_it):
        # Try the Cauchy step.
        alpha = 1
        x_t = x_c - alpha*g_c
        f_t = obj.f(x_t)

        # perform the linsearch if needed
        # I think i'm gonna add the stopping conditions here as well
        while(obj.f(x_t) < obj.f(x_c) - armijo_factor * alpha * LA.norm([g_c], 2) and consent(g_c, abs_grad_tol, rel_grad_tol, f_c, x_t, x_c, abs_stepsize_tol, rel_stepsize_tol)):
            alpha /= 2
            x_t = x_c - alpha*g_c
            f_t = obj.f(x_t)
        # Accept the new iterate
        x_c = x_t
        f_c = f_t
        g_c = obj.g(x_c)
    return x_c


def consent(g_c, abs_grad_tol, rel_grad_tol, f_c, x_t, x_c, abs_stepsize_tol, rel_stepsize_tol):
    # the stopping conditions
    # small gradient in an absolute sense
    criteria_one = (LA.norm(g_c) < abs_grad_tol)

    # Gradient small relative to the magnitude of f.
    criteria_two = (LA.norm(g_c) < rel_grad_tol * max(abs(f_c), 1))

    # small step in an absolute sense
    criteria_three = (LA.norm(x_t - x_c) < abs_stepsize_tol)

    # samll step relative to magnitude of x_c
    criteria_four = (LA.norm(x_t - x_c) <
                     (rel_stepsize_tol * max(LA.norm(x_c), 1)))
    return (criteria_one and criteria_two and criteria_three and criteria_four)


# if(__name__ == 'main'):
print("here we go")

obj = Quadratic()
x0 = numpy.array([2, 3]).T

# Check the gradient x0.
f0 = obj.f(x0)
h = 1.0e-05
dx = 2 * (numpy.random.random_sample(x0.shape) - 0.5)
f1 = obj.f(x0+h*dx)
fd = (f1 - 0)/h
g_c = obj.g(x0)
an = numpy.dot(g_c, dx)
print('The following should be close.')
print('finite difference: ', fd)
print('analytical:   ', an)

# Minimize the objective
x = gd0(obj, x0, max_it=100)
print('solution: ', x)

# Logisitic Regression
X_train = numpy.array([(+1, +1), (+1, -1), (-1, +1), (-1, -1)])
y_train = numpy.array([1, 1, 0, 0])
w0 = numpy.array([1, 1, 1]).T  # the weights
# print('f(w0):', numpy.exp(-obj.f(w0)))

obj = Logistic_Regression(X_train, y_train)
w = gd0(obj, w0, max_it=2000)
print('solution: ', w)
# print('f(w): ', numpy.exp(-obj.f(w)))
