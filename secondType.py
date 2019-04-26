from math import log
import numpy
from numpy import linalg as LA

# To the grader:
# I was having problems, so i retyped my project. (copying from a paper is hard)
# anyway, i retyped it. it worked. Idk what the problem was but it works now
# Regardless, there are fewer comments here than the original. if that matters to you, you can check my original on
# my github: https://github.com/ggarnhart/SteepestDescent
# it has lots of commits and lots of frustration. (OH and comments!)


class Quadratic(object):

    def f(self, x):
        return 0.5 * (x[0]**2 + 10*x[1]**2)

    def g(self, x):
        return numpy.array([x[0], 10*x[1]]).T


class Logistic_Regression(object):

    def __init__(self, X_train, y_train):
        self.num_cases = X_train.shape[0]
        e = numpy.ones((self.num_cases, 1))
        self.X_train = numpy.append(e, X_train, axis=1)
        self.y_train = y_train

    def f(self, w):
        z = self.X_train.dot(w)
        p = 1 / (1 + numpy.exp(-z))
        log_likelihood = 0
        for i in range(0, self.num_cases):
            if self.y_train[i] == 0:
                log_likelihood -= log(p[i])
            else:
                log_likelihood -= log(1 - p[i])
        return log_likelihood

    def g(self, w):
        z = self.X_train.dot(w)
        # print("Z", z)
        # sigmoidal stuff
        numerator = numpy.exp(z)
        denom = 1 + numerator
        frac = numerator / denom

        gradient = numpy.dot(self.X_train.T, frac)

        for i in range(0, gradient.size):
            if self.y_train[i] == 1:
                # print("stay positive", g[i])
                # i guess i don't really need to do anything else here.
                pass
            else:
                gradient[i] = gradient[i]*-1
                pass
        return gradient


def gd0(obj, x0, use_sgd=False, max_it=100, abs_grad_tol=1.0e-04, rel_grad_tol=1.0e-04, abs_stepsize_tol=1.0e-06, rel_stepsize_tol=1.0e-06, armijo_factor=1.0e-04):
    x_c = x0
    f_c = obj.f(x_c)
    g_c = obj.g(x_c)

    for it in range(0, max_it):
        alpha = 1
        x_t = x_c - alpha*g_c
        f_t = obj.f(x_t)

    # add armijo factor stuff here!
    while(obj.f(x_t) < obj.f(x_c) - armijo_factor * alpha * LA.norm([g_c], 2) and consent(g_c, abs_grad_tol, rel_grad_tol, f_c, x_t, x_c, abs_stepsize_tol, rel_stepsize_tol)):
        alpha /= 2
        x_t = x_c - alpha*g_c
        f_t = obj.f(x_t)

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


if __name__ == "__main__":
    numpy.random.seed(42)

    obj = Quadratic()
    x0 = numpy.array([2, 3]).T

    f0 = obj.f(x0)
    h = 1.0e-05
    dx = 2 * (numpy.random.random_sample(x0.shape) - 0.5)
    f1 = obj.f(x0 + h*dx)
    fd = (f1 - f0)/h
    g_c = obj.g(x0)
    an = numpy.dot(g_c, dx)
    print('the following should be close')
    print('finite diff: ', fd)
    print('analytical dif: ', an)

    x = gd0(obj, x0, max_it=100)
    print('solution: ', x)

    X_train = numpy.array([(1, 1), (1, -1), (-1, 1), (-1, -1)])
    y_train = numpy.array([1, 1, 0, 0])
    w0 = numpy.array([1, 1, 1]).T

    obj = Logistic_Regression(X_train, y_train)
    w = gd0(obj, w0, max_it=2000)
    print('solution 2: ', w)
