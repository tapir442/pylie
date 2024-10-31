#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 18 13:45:11 2022

@author: tapir
"""
# from https://ask.sagemath.org/question/7929/computing-variational-derivatives/
from IPython.core.debugger import set_trace
from delierium.helpers import is_function
import functools
from operator import mul

from sympy.core.backend import *

def is_op_du(expr_op, u):
    is_derivative = isinstance(
        expr_op.operator(),
        sage.symbolic.operators.FDerivativeOperator
    )

    if is_derivative:
        # Returns True if the differentiated function is `u`.
        return expr_op.operator().function() == u.operator()

    else:
        return False


def iter_du_orders(expr, u):
    for sub_expr in expr.operands():
        if sub_expr == []:
            # hit end of tree
            continue

        elif is_op_du(sub_expr.operator(), u):
            # yield order of differentiation
            yield len(sub_expr.operator().parameter_set())

        else:
            # iterate into sub expression
            for order in iter_du_orders(sub_expr, u):
                yield order


def func_diff(L, u_in):
    # `u` must be a callable symbolic expression
    # in one variable.
    set_trace()
    if len(u_in.variables()) == 1:
        x = u_in.variables()[0]
        u = u_in.function(x)
    else:
        raise TypeError

    # This variable name must not collide
    # with an existing one.
    # I use "tapir" in hopes that
    # nobody else does this...
    t = SR.var('t')
    result = SR(0)
    # `orders` is the set of all
    # orders of differentiation of `u`
    orders = set(iter_du_orders(L, u)).union((0,))

    for c in orders:
        du = u(x).diff(x, c)
        sign = Integer(-1)**c

        # Temporarily replace all `c`th derivatives of `u` with `t`;
        # differentiate; then substitute back.
        dL_du = L.subs({du: t}).diff(t).subs({t: du})
        # Append intermediate term to `result`
        result += sign * dL_du.diff(x, c)

    return result

#ar('x')
#= function('u')(x)
#
#g=SR.var('g')
#L = sqrt((1 + u.diff(x)**2)/(2*g*x))
#L
#set_trace()
#print(func_diff(L, u))
# Baumann pp 67
#1/2*sqrt(1/2)*(2*diff(u(x), x)*diff(u(x), x, x)/(g*x) - (diff(u(x), x)^2 + 1)/(g*x^2))*diff(u(x), x)/(g*x*((diff(u(x), x)^2 + 1)/(g*x))^(3/2)) - sqrt(1/2)*diff(u(x), x, x)/(g*x*sqrt((diff(u(x), x)^2 + 1)/(g*x))) + sqrt(1/2)*diff(u(x), x)/(g*x^2*sqrt((diff(u(x), x)^2 + 1)/(g*x)))
#
#s = var('s')
#q = function('q')(s)
#l = function("l")

#L = l(s, q(s), q.diff(s))
#print(func_diff(l(s, q(s), q.diff(s)), u))


t = var('t')

def EulerD(density, depend, independ):
    r'''
    >>> t = var("t")
    >>> u= function('u')
    >>> v= function('v')
    >>> L=u(t)*v(t) + diff(u(t), t)**2 + diff(v(t), t)**2 - u(t)**2 - v(t)**2
    >>> EulerD(L, (u,v), t)
    [-2*u(t) + v(t) - 2*diff(u(t), t, t), u(t) - 2*v(t) - 2*diff(v(t), t, t)]
    >>> L=u(t)*v(t) + diff(u(t), t)**2 + diff(v(t), t)**2 + 2*diff(u(t), t) * diff(v(t), t)
    >>> EulerD(L, (u,v), t)
    [v(t) - 2*diff(u(t), t, t) - 2*diff(v(t), t, t), u(t) - 2*diff(u(t), t, t) - 2*diff(v(t), t, t)]
    '''
    wtable = [function("w_%s" % i) for i in range(len(depend))]
    y = function('y')
    w = function('w')
    e = var('e')
    result = []
    for j in range(len(depend)):
        loc_result = 0
        def f0(*args):
            return y(independ) + e * w(independ)
        def dep(*args):
            return depend[j](independ)
        fh = density.substitute_function(depend[j], f0)
        fh = fh.substitute_function(y, dep)
        fh = fh.substitute_function(w, wtable[j])
        fh = fh.diff(e)
        fh = fh.subs({e:0}).expand()
        if fh.operator().__name__ == 'mul_vararg':
            operands = [fh]
        else:
            operands = fh.operands()
        for operand in operands:
            d     = None
            coeff = []
            for _ops in operand.operands():
                if is_op_du(_ops, wtable[j](independ)):
                    d = _ops.operands().count(independ)
                elif is_function(_ops) and _ops.operator() == wtable[j]:
                    pass
                else:
                    coeff.append(_ops)
            coeff = functools.reduce(mul, coeff, 1)
            if d is not None:
                coeff = ((-1)**d)*diff(coeff, independ, d)
            loc_result += coeff
        result.append(loc_result)
    return result


def FrechetD (support, dependVar, independVar, testfunction):
    """
    >>> x,t = var ("x t")
    >>> v   = function ("v")
    >>> u   = function ("u")
    >>> w1  = function ("w1")
    >>> w2  = function ("w2")
    >>> eqsys = [diff(v(x,t), x) - u(x,t), diff(v(x,t), t) - diff(u(x,t), x)/(u(x,t)**2)]
    >>> m = Matrix(FrechetD (eqsys, [u,v], [x,t], [w1,w2]))
    >>> m[0][0]
    -w1(x, t)
    >>> m[0][1]
    diff(w2(x, t), x)
    >>> m[1][0]
    2*w1(x, t)*diff(u(x, t), x)/u(x, t)^3 - diff(w1(x, t), x)/u(x, t)^2
    >>> m[1][1]
    diff(w2(x, t), t)
    """
    frechet = []
    eps = var ("eps")
    for j in range (len(support)):
        deriv = []
        for i in range (len(support)):
            def r0(*args):
                return dependVar[i](*independVar)+ testfunction[i](*independVar) * eps
            #def _r0(*args):
            #    # this version has issues as it always uses w2 ?!? investigate further
            #    # when time and motivation. Online version on asksage works perfectly
            #    return dependVar[i](*independVar)+ testfunction[i](*independVar) * eps
            #r0 = function('r0', eval_func=_r0)
            s  =  support[j].substitute_function(dependVar[i], r0)
            deriv.append (diff(s, eps).subs ({eps: 0}))
        frechet.append (deriv)
    return frechet




def AdjointFrechetD(support, dependVar, independVar, testfunction):
    frechet = FrechetD(support, dependVar, independVar, testfunction)



if __name__ == "__main__":
    import doctest
    doctest.testmod()
