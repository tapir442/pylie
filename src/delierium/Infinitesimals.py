#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  7 18:49:33 2022

@author: tapir
"""

import types
from collections import namedtuple
from itertools import product

from anytree import PreOrderIter
from sympy.core.backend import symbols, Function, diff

from .DerivativeOperators import FrechetD
from .JanetBasis import Janet_Basis
from .helpers import ExpressionTree


def prolongationFunction(f: list, x: list, order) -> list:
    '''
    >>> x, y, z = var("x y z")
    >>> f = function("f")(x, y, z)
    >>> set(prolongationFunction([f], [x, y, z], 2)) == set(
    ... [diff(f, z, z), diff(f, y), diff(f, x),
    ... diff(f, z), f, diff(f, x, z),
    ... diff(f, x, y), diff(f, x, x),
    ... diff(f, y, y), diff(f, y, z)])
    True
    '''
    result = f
    aux = result[:]

    def outer(fun, l1, l2):
        return list(map(lambda v: fun(v[0], v[1]), product(l1, l2)))
    for i in range(order):
        result += (aux := outer(diff, aux, x)[:])
    return sorted(list(set(result)))


def infini(eq):
    pass


def prolongation(eq, dependent, independent):
    """

    Doctest stolen from Baumann pp.92/93
    >>> x = var('x')
    >>> u = function('u')
    >>> u_x = u(x)
    >>> f = function("f")
    >>> f_x = f(x, u(x), diff(u(x),x))
    >>> ppp = prolongation([f_x], [u], [x])
    >>> print(ppp[0].expand())
    -D[2](f)(x, u(x), diff(u(x), x))*diff(u(x), x)^2*D[1](xi_1)(x, u(x)) + D[2](f)(x, u(x), diff(u(x), x))*D[1](phi_1)(x, u(x))*diff(u(x), x) - D[2](f)(x, u(x), diff(u(x), x))*diff(u(x), x)*D[0](xi_1)(x, u(x)) + xi_1(x, u(x))*D[0](f)(x, u(x), diff(u(x), x)) + phi_1(x, u(x))*D[1](f)(x, u(x), diff(u(x), x)) + D[2](f)(x, u(x), diff(u(x), x))*D[0](phi_1)(x, u(x))
    >>> # this one here is from Baumann, p.93
    >>> f_x = f(x, u(x), diff(u(x),x),  diff(u(x), x ,x))
    >>> # Baumann's example p. 94
    >>> x = var('x')
    >>> y = function('y')
    >>> print(prolongation([diff(y(x),x,2)], [y], [x])[0].expand())
    -D[1, 1](xi_1)(x, y(x))*diff(y(x), x)^3 + D[1, 1](phi_1)(x, y(x))*diff(y(x), x)^2 - 2*D[0, 1](xi_1)(x, y(x))*diff(y(x), x)^2 - 3*D[1](xi_1)(x, y(x))*diff(y(x), x)*diff(y(x), x, x) + 2*D[0, 1](phi_1)(x, y(x))*diff(y(x), x) - D[0, 0](xi_1)(x, y(x))*diff(y(x), x) + D[1](phi_1)(x, y(x))*diff(y(x), x, x) - 2*D[0](xi_1)(x, y(x))*diff(y(x), x, x) + D[0, 0](phi_1)(x, y(x))
    """
    Depend = [d(*independent) for d in dependent]
    vars = independent + Depend
    xi = [function("xi_%s" % (j+1), latex_name = r"\xi_{i+1}") for j in range(len(independent))]
    eta = []
    for i in range(len(dependent)):
        phi = function(f"phi_{i+1}", latex_name = fr"\phi_{i+1}")
        eta.append(phi(*vars) -
                   sum(xi[j](*vars) *
                       Depend[i].diff(independent[j])
                       for j in range(len(independent))))
    test = list(map(lambda _: function("t_%s" % _),  range(len(Depend))))
    prolong = FrechetD(eq, dependent, independent, testfunction=test)
    prol = []
    for p in prolong:
        _p = []
        for l in p:
            _p.extend([l.substitute_function(test[i], _e.function()) for _e in eta])
        prol.append(sum(_ for _ in _p))
    prolong = prol[:]
    prol = []
    for j in range(len(prolong)):
        for i in range(len(independent)):
            prol.append(
                (prolong[j] +
                 xi[i](*vars) * sum(_.diff(independent[i]) for _ in eq).expand())
            )
    return prol


def prolongationODE(equations,
                    dependent,
                    independent,
                    infinitesimals=None):
    """
    >>> # Baumann, ex 1, pp.136
    >>> x    = var("x")
    >>> u    = function('u')
    >>> F    = function("F")
    >>> ode3 = diff(u(x), x) - F(u(x),x)
    >>> X=function('X')
    >>> Y=function('Y')
    >>> prolongationODE(ode3,u,x, infinitesimals=(X,Y))
    [X(u(x), x)*D[0](F)(u(x), x)*diff(u(x), x) - D[0](X)(u(x), x)*diff(u(x), x)^2 - (D[0](F)(u(x), x)*diff(u(x), x) + D[1](F)(u(x), x) - diff(u(x), x, x))*X(u(x), x) - Y(u(x), x)*D[0](F)(u(x), x) - D[1](X)(u(x), x)*diff(u(x), x) + D[0](Y)(u(x), x)*diff(u(x), x) - X(u(x), x)*diff(u(x), x, x) + D[1](Y)(u(x), x)]
    >>> # Baumann, ex 2, p.137
    >>> g = function("g")
    >>> f = function("f")
    >>> ode4 = diff(u(x),x)-g(u(x))*f(x)
    >>> p = prolongationODE(ode4,u,x)[0].expand()
    >>> sol = solve(ode4, diff(u(x), x))
    >>> p = p.subs({sol[0].lhs() : sol[0].rhs()})
    >>> print(p.expand())
    -f(x)^2*g(u(x))^2*D[0](xi)(u(x), x) - g(u(x))*xi(u(x), x)*diff(f(x), x) - f(x)*phi(u(x), x)*D[0](g)(u(x)) + f(x)*g(u(x))*D[0](phi)(u(x), x) - f(x)*g(u(x))*D[1](xi)(u(x), x) + D[1](phi)(u(x), x)
    """
    vars     = [dependent(independent), independent]
    if infinitesimals is None:
        infinitesimals = (function("xi", latex_name=r"\xi"), function("phi", latex_name=r"\phi"))
    xi, phi  = infinitesimals
    eta      = phi(*vars) - xi(*vars) * diff(dependent(independent), independent)
    test     = function('test')
    prolong  = FrechetD([equations], [dependent], [independent], testfunction=[test])
    prol     = []
    for p in prolong:
        _p = (l.substitute_function(test, eta.function()).expand() for l in p)
        prol.append(sum(_ for _ in _p))
    return list(map (lambda _ : _ + xi(*vars) * equations.diff(independent), prol))

term = namedtuple("term", ["power", "coeff"])

def overdeterminedSystemODE (ode,
                       dependent,
                       independent,
                       infinitesimals=None
                       , *args, **kw):
    """Computes the overdetermined system which is computed from the prolongation
    of an ODE of order > 1

    Parameters
    ----------
    ode: a sagemath expression as the left side of '<expr> == 0'. No need to
        add " == 0'!!
    dependent: the name of the dependent variable, i.e. the unknown function
    independent:
        the name of the independent variable
    infinitesimals: ordered pair of sagemath variables, to be used as the names
        for the infinitesimals, to avoid potential name clashes with  variables in your
        application. If not specified, 'xi' and 'phi' are used as the defaults

    Returns
    -------
    list
        a list of expressions, each expression to be interpreted as left side of an
        'expr' == 0. For further manipulation ane has to add ' == 0'.


    >>> # Arrigo Example 2.20
    >>> x   = var('x')
    >>> y   = function('y')
    >>> ode = diff(y(x), x, 3) + y(x) * diff(y(x), x, 2)
    >>> X=function('X')
    >>> Y=function('Y')
    >>> inf = overdeterminedSystemODE(ode, y, x, infinitesimals=(X,Y))
    >>> for _ in inf:
    ...     print(_)
    -3*D[0](X)(y(x), x)
    -6*D[0, 0](X)(y(x), x)
    y(x)*D[0](X)(y(x), x) - 9*D[0, 1](X)(y(x), x) + 3*D[0, 0](Y)(y(x), x)
    y(x)*D[1](X)(y(x), x) + Y(y(x), x) - 3*D[1, 1](X)(y(x), x) + 3*D[0, 1](Y)(y(x), x)
    -D[0, 0, 0](X)(y(x), x)
    -y(x)*D[0, 0](X)(y(x), x) - 3*D[0, 0, 1](X)(y(x), x) + D[0, 0, 0](Y)(y(x), x)
    -2*y(x)*D[0, 1](X)(y(x), x) + y(x)*D[0, 0](Y)(y(x), x) - 3*D[0, 1, 1](X)(y(x), x) + 3*D[0, 0, 1](Y)(y(x), x)
    -y(x)*D[1, 1](X)(y(x), x) + 2*y(x)*D[0, 1](Y)(y(x), x) - D[1, 1, 1](X)(y(x), x) + 3*D[0, 1, 1](Y)(y(x), x)
    y(x)*D[1, 1](Y)(y(x), x) + D[1, 1, 1](Y)(y(x), x)
    """
    if infinitesimals is None:
        infinitesimals = (function("xi", latex_name=r"\xi"), function("phi", latex_name=r"\phi"))
    prolongation = prolongationODE(ode, dependent, independent, infinitesimals=infinitesimals)[0].expand()
    tree = ExpressionTree(prolongation)
    mine = [_ for _ in tree.diffs if _.operator().function() in [dependent]]
    order= max([len(_.operator().parameter_set()) for _ in mine])
    if order == 1:
        print("Order 1 ODEs have no meaningful infinitesimals")
        return []
    s1 = solve(ode==0, diff(dependent(independent),independent, order))
    ode1 = prolongation.subs({s1[0].lhs() : s1[0].rhs()}).simplify()
    tree = ExpressionTree(ode1)
    l = (_ [0] for _ in ode1.coefficients(diff(dependent(independent), independent, order)))
    equations = []
    e         = next(l)
    all_this_stuff = set()
    for node in PreOrderIter(tree.root):
        # powercollector: an array which stores powers of derivatives
        # the index is the(reversed) order, the value is the power
        # of the derivative.
        # Example: we have an ODE of order three. The prolongation and
        # substitution step produces 'ode1' which is now of reduced order
        # two. So we can have differentials of order one and two, so we need an
        # array of lenght two which is initialized with zeroes. A term like
        #    diff(y, x)^5 * diff(y, x, x)^2
        # will create the entry
        #    [2,5]
        # The higher order (=2) has power 2, so the first entry (=highest order)
        # will be set to 2, the lowest order(=1) has power 5, so the index 1 contains
        # the power 5. This way we now have on ordered list which than can be
        # looped over from highest_order^highest_power to lowest_order^lowest_power
        # to factor out the derivatives to get the determining equations
        powercollector = [0]*(order-1)
        v = node.value
        if v.operator() in [sage.symbolic.operators.add_vararg, None]:
             continue
        if isinstance(v.operator(), FDerivativeOperator):
            # standalone diff operator
            if v.operands()[0] != independent:
                # differential coming from prolongation, ignore
                continue
            powercollector[order - len(v.operator().parameter_set())-1] = 1
            all_this_stuff.add(term(tuple(powercollector), v))
            continue
        if v.operator() is sage.symbolic.operators.mul_vararg:
            # the factors containing derivatives can be combined multiplicatively
            # We will analize them factor by factor, put powers and orders into
            # 'power_collector', and multiply these factors together into 'local_term',
            # ans store both together into 'all_this_stuff'
            local_term = 1
            for w in v.operands():
                if isinstance(w.operator(), FDerivativeOperator):
                    if w.operands()[0] != independent:
                        # differential coming from prolongation, ignore
                        continue
                    local_term *= w
                    powercollector[order - len(w.operator().parameter_set())-1] = 1
                if isinstance(w.operator(), types.BuiltinFunctionType):
                    if w.operator().__qualname__ != 'pow':
                        continue
                    if isinstance(w.operands()[0].operator(), FDerivativeOperator):
                        if w.operands()[0].operands()[0] != independent:
                            # differential coming from prolongation, ignore
                            continue
                        local_term *= w
                        powercollector[order - len(w.operands()[0].operator().parameter_set())-1] = w.operands()[-1]

            if powercollector != [0]*(order-1):
                all_this_stuff.add(term(tuple(powercollector), local_term))
    for _ in reversed(sorted(all_this_stuff)):
        new = e.coefficient(_.coeff)
        if new != 0:
            equations.append(new)
        e = (e - new * _.coeff).expand()
    if e != 0:
        equations.append(e)
    return equations

def Janet_Basis_from_ODE(ode, dependent, independent, order = "Mgrevlex", *args, **kw):
    overdetermined_system = overdeterminedSystemODE(ode, dependent, independent)
    #ToDo: 2 way:
    #    * either as Janet_Basis
    #    * or try to solve the undetermined system
    Y = var('Y')
    intermediate_system = []
    for e in overdetermined_system:
        # ToDo: make the next three lines into a function for helpers(code duplication
        #       with overdeterminedSystemODE. Idea: return a dict with {function: order}#
        tree = ExpressionTree(e)
        mine = [_ for _ in tree.diffs if _.operator().function() in [dependent]]
        order= max([len(_.operator().parameter_set()) for _ in mine]) if mine else 0
        e = e.subs({dependent(independent): Y})
        for j in range(1, order+1):
            d = diff(dependent(independent), independent, j)
            e = e.subs({d: 0})
        intermediate_system.append(e)
    # ToDo: get rid of hardcoded phi and xi

    print("Overdeterminedsystemode")
    for _ in intermediate_system:
        _.show()


    janet = Janet_Basis(intermediate_system, [phi, xi], [Y, independent])
    pols = list(map(lambda _ : _.expression().subs({Y : dependent(independent)}), janet.S))
    return pols


if __name__ == "__main__":
    import doctest
    doctest.testmod()
