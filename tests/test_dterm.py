import delierium.matrix_order as M
import delierium.JanetBasis as JB
import functools

from sympy import *


from delierium.helpers import (is_derivative, is_function)
from operator import mul


def _generate_terms(expr):
    return Add.make_args(expr)


def analyze_expression(ctx, term):
    operands = Mul.make_args(term)
    coeffs = []
    d = []
    for operand in operands:
        if is_function(operand):
            if ctx.is_ctxfunc(operand):
                d.append(operand)
            else:
                coeffs.append(operand)
        elif is_derivative(operand):
            if ctx.is_ctxfunc(operand.operator().function()):
                d.append(operand)
            else:
                coeffs.append(operand)
        else:
            coeffs.append(operand)

    coeffs = functools.reduce(mul, coeffs, 1)
    return coeffs, d[0]


Dterm = JB._Dterm


def test_simple():
    R = QQ["x", "y", "z"]  # pylint: disable=protected-access
    x, y, z = symbols("x, y, z")
    f = Function('f')
    g = Function('g')
    h = Function('h')
    t = Function('t')

    ctx = M.Context([f, g, h], [x, y, z])

    expr = diff(Rational('-1/4') * f(x, y, z), x, x, y) - t(x, y, z) * Rational(9) * diff(h(x, y, z), z, 3)
    terms = _generate_terms(expr)
    dterms = [Dterm(*(analyze_expression(ctx, _) +(ctx,))) for _ in terms]
    print(dterms)
