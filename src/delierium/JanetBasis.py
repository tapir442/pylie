"""
Janet Basis
"""



import functools
import os

from collections import OrderedDict, namedtuple
from collections.abc import Iterable
from dataclasses import dataclass
from itertools import groupby, islice
from operator import mul

from IPython.core.debugger import set_trace
from IPython.display import Math
from more_itertools import bucket, flatten, powerset

from delierium.exception import DelieriumNotALinearPDE
from delierium.helpers import (_adiff, adiff, eq, expr_eq, expr_is_zero, is_derivative,
                               is_function, pairs_exclude_diagonal, is_numeric)
from delierium.Involution import My_Multiplier
from delierium.matrix_order import Context, Mgrevlex, Mgrlex
from delierium.typedefs import *

from collections.abc import Callable, Iterator
from typing import ClassVar, Optional, Union

from line_profiler import profile


from sympy import *

from sympy.core.backend import *


try:
    __IPYTHON__
    _in_ipython_session = True
except NameError:
    _in_ipython_session = False


def compute_comparison_vector(dependent, func, ctxcheck):
    iv = [0] * len(dependent)
    if func in dependent:
        iv[dependent.index(func.name)] = 1
    elif ctxcheck(func):
        iv[dependent.index(func.name)] = 1
    else:
        pass
    return iv

@profile
def compute_order(derivative, independent, comp_order):
    """Computes the monomial tuple from the derivative part."""
    if is_derivative(derivative):
        return comp_order(derivative)
    # XXX: Check can that be within a system of linear PDEs ?
    return [0] * len(independent)


@dataclass()
class _Dterm:
    coeff: int
    derivative: int
    context: Context

    @profile
    def __post_init__(self):
        object.__setattr__(self, 'order', self._compute_order())
        if is_derivative(self.derivative):
            object.__setattr__(self, 'function', self.derivative.args[0].func)
        else:
            object.__setattr__(self, 'function', self.derivative.func)
        object.__setattr__(self, 'comparison_vector', self._compute_comparison_vector())

    @profile
    def expression(self):
        return self.coeff * self.derivative
    @profile
    def _compute_comparison_vector(self):
        """Concatenates order and comparison vector for input for ..."""
        iv = compute_comparison_vector(self.context.dependent, self.function, self.context.is_ctxfunc)
        return tuple(self.order + iv)

    def __str__(self):
        if self.coeff == 1:
            result = f"{self.derivative}"
        else:
            result = f"({self.coeff}) * { self.derivative}"
        return result.replace("Derivative", "D")

    def term(self):
        return self.coeff * self.derivative
    @profile
    def _compute_order(self):
        """computes the monomial tuple from the derivative part"""
        return compute_order(self.derivative, self.context.independent, self.context.order_of_derivative)

    @profile
    def is_zero(self):
        return expr_is_zero(self.coeff)

    def __sub__(self, other):
        if self.comparison_vector != other.comparison_vector:
            raise ValueError
        return self.__class__(coeff = self.coeff - other.coeff,
                              derivative = self.derivative,
                              context = self.context)

    def __add__(self, other):
        if self.comparison_vector != other.comparison_vector:
            raise ValueError
        return self.__class__(coeff = self.coeff + other.coeff,
                              derivative = self.derivative,
                              context = self.context)



    def is_coefficient(self):
        # XXX nonsense
        return self.derivative == 1


    @profile
    def __bool__(self):
        # ToDo, lets think about that again, may be too slow
        return not self.is_zero()

    @profile
    def __lt__(self, other):
        """
        >>> x,y,z = symbols("x y z")
        >>> g     = Function("g")(x,y,z)
        >>> h     = Function("h")(x,y,z)
        >>> ctx   = Context ((f,g,h),(x,y,z), Mlex)
        >>> dterm1 = _Dterm(derivative=diff(f, x, y), coeff=x**2, context=ctx)
        >>> dterm2 = _Dterm(derivative=diff(f, x, y, z), coeff=1 , context=ctx)
        >>> print(bool(dterm1 < dterm2))
        True
        """
        # XXX context.gt still a bad place
        return not self == other and \
            self.context.gt(other.comparison_vector, self.comparison_vector)

    @profile
    def __eq__(self, other):
        return self is other or \
            (self.comparison_vector == other.comparison_vector and \
             expr_eq(self.coeff, other.coeff))

    def show(self, rich=True):
        if not rich:
            return str(self)
        return self.latex()

    def latex(self):
        """Converts a _Dterm into Lie traditional form, latex style"""

        def _latex_derivative(deriv):
            if is_derivative(deriv):
                func = deriv.args[0].func
                ps = deriv.args[1:]
#                import pdb; pdb.set_trace()
                inter = []
                for v in ps:
                    inter.append(",".join((latex(v[0])*v[1])))
                sub = ",".join(inter)
                return f"{func}_{{{sub}}}"
            elif is_function(deriv):
                return latex(deriv.func)
            else:
                return latex(deriv)

        def _latex_coeff(coeff):
            if str(coeff) in ['1', '1.0']:
                return ""
            if str(coeff) in ['-1', '-1.0']:
                return "-"
            # ToDo: need to be more fine granular
            if hasattr(coeff, "expand"):
                c = latex(coeff.expand().simplify())
            else:
                c = latex(coeff)
            if hasattr(coeff, "operator") and \
                coeff.operator() != None and \
                ((hasattr(coeff.operator(), "__name__") and coeff.operator().__name__ == "add_vararg") or is_function(coeff.operator())):
                return rf"({c})"
            return c

        d = _latex_derivative(self.derivative)
        c = _latex_coeff(self.coeff)
        return f"{c} {d}"

    _latex_ = latex

    def add_coefficient(self, c):
        return _Dterm(coeff = self.coeff + c,
                      derivative = self.derivative,
                      context = self.context)

    @profile
    def diff(self, *variables):
        f = self.coeff
        g = self.derivative
        try:
            fprime = adiff(f, self.context, *variables)
        except AttributeError:
            fprime = 0
        result = []
        if not is_numeric(fprime) or (is_numeric(fprime) and fprime != 0):
            d1 = _Dterm(coeff=fprime,
                        derivative=g,
                        context=self.context)
            result = [d1]
        gprime = adiff(g, self.context, *variables)
        d2 = _Dterm(coeff=f,
                    derivative=gprime,
                    context=self.context)
        if d2:
            result.append(d2)
        return result

    @profile
    def __hash__(self):
        return hash(str(self.coeff) + str(self.derivative))

    _cache_key = __hash__

class LHDP:
    """Linear Homogenious Differential Polynomial."""
    @profile
    def __init__(self, e, context, dterms=[]):
        self.context = context
        self.p = []
        self.multipliers = []
        self.nonmultipliers = []
        self.hash = 0
        if dterms:
            self.p = dterms[:]
        else:
            self._init(e.expand())

        self.p.sort(reverse=True)
        self.normalize()

    @profile
    def _init(self, e):
        operands = e.make_args(e)
        import pdb; pdb.set_trace()
        r = [analyze_term(self.context, o) for o in operands]
        dterms = {}
        for _r in r:
            dterms.setdefault(_r[0], []).append((_r[1], _r[2]))
        self.p = []
        for v in dterms.values():
            # v is a list of tuples
            c = 0
            for tup in v:
                c += tup[1]
            self.p.append(_Dterm(derivative=v[0][0], coeff=c, context=self.context))

    def expression(self):
        return sum(_.expression() for _ in self.p)

    def _collect_terms(self, e):
        pass

    def show_derivatives(self):
        print(list(self.derivatives()))

    def Lterm(self):
        return self.p[0].term()

    def Lder(self):
        return self.p[0].derivative

    def Lfunc(self):
        return self.p[0].function

    def Lcoeff(self):
        return self.p[0].coeff

    def terms(self):
        for p in self.p:
            yield p.term()

    def derivatives(self):
        for p in self.p:
            yield p.derivative

    def Ldervec(self):
        # implement asap
        pass

    def coefficients(self):
        for p in self.p:
            yield p.coeff

    @profile
    def normalize(self):
        if self.p:
            if not is_numeric(self.Lcoeff()) or \
               (is_numeric(self.Lcoeff()) and self.Lcoeff() != 1):
                c = self.Lcoeff()
                self.p = [_Dterm(_.coeff / c, _.derivative, self.context) for _ in self.p if not _.is_zero()]
        # XXX: wrong place?
        if self.p:
            self.order = self.p[0].order
            self.function = self.p[0].function
            self.comparison_vector = self.p[0].comparison_vector

    def __bool__(self):
        return len(self.p) > 0

    @profile
#    @cache
    def __lt__(self, other):
        for _ in zip(self.p, other.p):
            if eq(_[0], _[1]):
                continue
            if _[0] < _[1]:
                return True
            return False
        return False

    @profile
    def __le__(self, other):
        return eq(self, other) or self < other

    @profile
    def __eq__(self, other):
        if self is other:
            return True
        if len(self.p) != len(other.p):
            return False
        return all(_[0] == _[1] for _ in zip(self.p, other.p))

    def show(self, rich=True, short=False):
        if not rich:
            return str(self)
        res = ""
        for _ in self.p:
            if short:
                res += " " + str(_.derivative)
            else:
                s = _.show()
                if not res:
                    res = s
                    continue
                if s.startswith("-1 "):
                    s = s.replace("-1 ", "-")
                if s.startswith("-"):
                    res += s
                else:
                    res += " + " + s
        res += f"{self.multipliers}, {self.nonmultipliers}"
        return res

    def latex(self):
        return "+".join(_.latex() for _ in self.p).replace("(-", "(").replace("+-", "-")

    _latex_ = latex

    @profile
    def diff(self, *args):
        new_dterms = {}
        for dterm in self.p:
            _dterms = dterm.diff(*args)
            for new_dterm in _dterms:
                if new_dterm.comparison_vector in new_dterms:
                    new_dterms[new_dterm.comparison_vector].coeff += new_dterm.coeff
                else:
                    new_dterms[new_dterm.comparison_vector] = new_dterm
        return self.__class__(e=0,
                              dterms=[_ for _ in new_dterms.values() if _.coeff != 0],
                              context=self.context)

    def __str__(self):
        m = [self.context.independent[_] for _ in self.multipliers]
        n = [self.context.independent[_] for _ in self.nonmultipliers]
        result = " + ".join([str(_) for _ in self.p])
        if m or n:
            result += f", {m}, {n}"
        return result

    def __repr__(self):
        return str(self)

    def __hash__(self):
        if self.hash == 0:
            self.hash = hash("".join([str(hash(_)) for _ in self.p]))
        return self.hash

    _cache_key = __hash__

def analyze_term(context, term):
    operands = split_into_operands(term)
    coeffs = []
    d = []
    for operand in operands:
        if is_function(operand):
            if context.is_ctxfunc(operand):
                d.append(operand)
            else:
                coeffs.append(operand)
        elif is_derivative(operand):
            if context.is_ctxfunc(operand.args[0].func()):
                d.append(operand)
            else:
                coeffs.append(operand)
        else:
            coeffs.append(operand)

    coeffs = functools.reduce(mul, coeffs, 1)
    return str(d[0]), d[0], coeffs


def split_into_operands(term):
    if is_derivative(term):
        operands = [term]
    elif is_function(term):
        operands = [term]
    else:
        try:
            operands = term.as_ordered_factors()
        except AttributeError:
            # symengine
            operands = term.args_as_sympy()
    return operands

# ToDo: Janet_Basis as class as this object has properties like rank, order ...


def Reorder(S, context, ascending=False):
    return list(sorted(S))


def reduceS(e: LHDP, S: list, context: Context) -> LHDP:
    reducing = True
    for _ in S:
        _.show(rich=True, short=True)
    gen = (_ for _ in S)
    while reducing:
        for dp in gen:
            enew = reduce(e, dp, context)
            # XXX check wheter we can replace "==" by 'is'
            if enew == e:
                reducing = False
            else:
                e = enew
                gen = (_ for _ in S if _)
                reducing = True
    return enew


#@functools.cache
@profile
def _order(der, context):
    # pretty sure we don't need it
    if der != 1:
        return context.order_of_derivative(der)
    return [0] * len(context.independent)


def _reduce_inner(e1, e2, context):
    for t in (_ for _ in e1.p if _.function == e2.function):
        c = t.coeff
        dif = [a - b for a, b in zip(t.order, e2.order)]
        changed = OrderedDict([(_.comparison_vector, _) for _ in e1.p])
        subs = []
        have_changed = False
        if all(map(lambda h: h == 0, dif)):
            have_changed = True
            # S2 from Algorithm 2.4
            for p2 in e2.p:
                pc = p2.coeff * c
                hit = changed.get(p2.comparison_vector, None)
                if hit:
                    if not expr_eq(hit.coeff, pc):
                        hit.coeff -= pc
                    else:
                        del changed[hit.comparison_vector]
                else:
                    dt = _Dterm(coeff=-pc, derivative=p2.derivative, context=e1.context)
                    if dt:
                        subs.append(dt)

        elif all(map(lambda h: h >= 0, dif)):
            have_changed = True
            variables_to_diff = get_diff_vars(context, dif)
            changed = OrderedDict([(_.comparison_vector, _) for _ in e1.p])
            subs = []
            for p2 in e2.p:
                dterms = p2.diff(*variables_to_diff)
                for dterm in dterms:
                    hit = changed.get(dterm.comparison_vector, None)
                    pc = dterm.coeff * c
                    if hit:
                        if not expr_eq(hit.coeff, pc):
                            hit.coeff -= pc
                        else:
                            del changed[hit.comparison_vector]
                    else:
                        subs.append(_Dterm(coeff=-dterm.coeff,
                                           derivative=dterm.derivative,
                                           context=dterm.context))
        else:
            pass
        if have_changed:
            # avoid creating a _Dterm in case of no change
            # destroys algorith, though it should create copy of e1 which
            # should'nt hurt, but it hurts
            dterms = [_ for _ in [*changed.values()] + subs if _]
            if dterms:
                return LHDP(e=0, context=e2.context, dterms=dterms)

    return e1

def get_diff_vars(context, dif):
    variables = []
    for i in range(len(context.independent)):
        if dif[i] != 0:
            variables.extend([context.independent[i]] * abs(dif[i]))
    return variables


def reduce(e1: LHDP, e2: LHDP, context: Context) -> LHDP:
    while not (new_e1 := _reduce_inner(e1, e2, context)) == e1:
        e1 = new_e1
    return new_e1


def Autoreduce(S, context):
    dps = list(S)
    i = 0
    _p, r = dps[:i + 1], dps[i + 1:]
    while r:
        newdps = []
        have_reduced = False
        for _r in r:
            rnew = reduceS(_r, _p, context)
            have_reduced = have_reduced or rnew != _r
            if rnew:
                newdps.append(rnew)
        dps = Reorder(_p + [_ for _ in newdps if _ not in _p], context, ascending=True)
        if not have_reduced:
            i += 1
        else:
            i = 0
        _p, r = dps[:i + 1], dps[i + 1:]
    return dps


def vec_degree(v, m):
    return m[v]

@profile
def vec_multipliers(m, M, Vars):
    """multipliers and nonmultipliers for differential vectors aka tuples

    m   : a tuple representing a differential vector
    M   : the complete set of differential vectors
    Vars: a tuple representing the order of indizes in m
          Examples:
              (0,1,2) means first index in m represents the highest variable
              (2,1,0) means last index in m represents the highest variable

    ......................................................
    The doctest example is from Schwarz, Example C.1, p. 384
    This example is in on variables x1,x2,x3, with x3 the highest rated variable.
    So we have to specify (2,1,0) to represent this

    >>> M = [(2,2,3), (3,0,3), (3,1,1), (0,1,1)]
    >>> r = vec_multipliers (M[0],M, (2,1,0))
    >>> print (M[0], r[0], r[1])
    (2, 2, 3) [2, 1, 0] []
    >>> r = vec_multipliers (M[1],M, (2,1,0))
    >>> print (M[1], r[0], r[1])
    (3, 0, 3) [2, 0] [1]
    >>> r = vec_multipliers (M[2],M, (2,1,0))
    >>> print (M[2], r[0], r[1])
    (3, 1, 1) [1, 0] [2]
    >>> r = vec_multipliers (M[3],M, (2,1,0))
    >>> print (M[3], r[0], r[1])
    (0, 1, 1) [1] [0, 2]
    >>> N=[[0,2], [2,0], [1,1]]
    >>> r =vec_multipliers(N[0], N,  (0,1))
    >>> print(r)
    ([1], [0])
    >>> r =vec_multipliers(N[1], N,  (0,1))
    >>> print(r)
    ([0, 1], [])
    >>> r =vec_multipliers(N[2], N,  (0,1))
    >>> print(r)
    ([1], [0])
    >>> r =vec_multipliers(N[0], N,  (1,0))
    >>> print(r)
    ([1, 0], [])
    >>> r =vec_multipliers(N[1], N,  (1,0))
    >>> print(r)
    ([0], [1])
    >>> r =vec_multipliers(N[2], N,  (1,0))
    >>> print(r)
    ([0], [1])
    >>> # next example form Gerdt/Blinkov: Janet-like monomial divisiom, Table1
    >>> # x1 -> Index 2
    >>> # x2 -> Index 1 (this is easy)
    >>> # x3 -> Index 0
    >>> U = [[0,0,5], [1,2,2],[2,0,2], [1,4,0],[2,1,0],[5,0,0]]
    >>> vec_multipliers(U[0], U, (2,1,0))
    ([2, 1, 0], [])
    >>> vec_multipliers(U[1], U, (2,1,0))
    ([1, 0], [2])
    >>> vec_multipliers(U[2], U, (2,1,0))
    ([0], [1, 2])
    >>> vec_multipliers(U[3], U, (2,1,0))
    ([1, 0], [2])
    >>> vec_multipliers(U[4], U, (2,1,0))
    ([0], [1, 2])
    >>> vec_multipliers(U[5], U, (2,1,0))
    ([0], [1, 2])
    """
    d = max((vec_degree(v, u) for u in M for v in Vars), default=0)
    mult = []
    if vec_degree(Vars[0], m) == d:
        mult.append(Vars[0])
    for j in range(1, len(Vars)):
        v = Vars[j]
        dd = list(map(lambda x: vec_degree(x, m), Vars[:j]))
        V = []
        for _u in M:
            if [vec_degree(_v, _u) for _v in Vars[:j]] == dd:
                V.append(_u)
        if vec_degree(v, m) == max((vec_degree(v, _u) for _u in V), default=0):
            mult.append(v)
    return mult, list(sorted(set(Vars) - set(mult)))

coll = namedtuple('coll', ['monom', 'dp', 'multipliers', 'nonmultipliers'])

@profile
def complete(S, context):
    result = set(S)
    if len(result) == 1:
        return list(result)
    vars = list(range(len(context.independent)))
    map_old_to_new = lambda v: context.independent[vars.index(len(vars)-1-v)]

    while 1:
        monomials = [(_,list(reversed(_.order))) for _ in result]
        ms = tuple([_[1] for _ in monomials])
        m0 = []

        # multiplier-collection is our M
        multiplier_collection = []
        for dp, monom in monomials:
            # S1
            _multipliers, _nonmultipliers = vec_multipliers(monom, ms, vars)
            multiplier_collection.append(
                coll(monom, dp,_multipliers, _nonmultipliers))
        for entry in multiplier_collection:
            if not entry.nonmultipliers:
                m0.append((entry.monom, None, entry.dp))
            else:
                # todo: do we need subsets or is a multiplication by only one
                # nonmultiplier one after the other enough ?
                for n in entry.nonmultipliers:
                    _m0 = list(entry.monom)
                    _m0[n] += 1
                    m0.append((_m0, n, entry.dp))

        to_remove = []
        for _m0 in m0:
            # S3: check whether in class of any of the monomials
            for entry in multiplier_collection:
                # XXX debug!
                if all(map(lambda x: _m0[0][x] >= entry.monom[x], entry.multipliers)) and \
                   all(map(lambda x: _m0[0][x] == entry.monom[x], entry.nonmultipliers)):
                    # this is in _m0's class
                    to_remove.append(_m0)
        for _to in to_remove:
            try:
                m0.remove(_to)
            except:
                pass
        if not m0:
            return list(result)
        for _m0 in m0:
            # differentiate by nonmultiplier
            dp = _m0[2].diff(map_old_to_new(_m0[1]))
            result.add(dp)
        result = set(Reorder(list(result), context, ascending=False))


def CompleteSystem(S, context):
    """
    Algorithm C1, p. 385

    >>> tvars=var("x y z")
    >>> w = function("w")(*tvars)
    >>> # these DPs are constructed from C1, pp 384
    >>> h1=diff(w, x,x,x, y,y,z,z)
    >>> h2=diff(w, x,x,x,     z,z,z)
    >>> h3=diff(w, x,     y,  z,z,z)
    >>> h4=diff(w, x,     y)
    >>> ctx=Context((w,),(x,y,z), Mgrlex)
    >>> dps=[LHDP(_, ctx) for _ in [h1,h2,h3,h4]]
    >>> cs = CompleteSystem(dps, ctx)
    >>> # things are sorted up
    >>> for _ in cs: print(_)
    diff(w(x, y, z), x, y)
    diff(w(x, y, z), x, y, z)
    diff(w(x, y, z), x, x, y)
    diff(w(x, y, z), x, y, z, z)
    diff(w(x, y, z), x, x, y, z)
    diff(w(x, y, z), x, x, x, y)
    diff(w(x, y, z), x, y, z, z, z)
    diff(w(x, y, z), x, x, y, z, z)
    diff(w(x, y, z), x, x, x, y, z)
    diff(w(x, y, z), x, x, x, y, y)
    diff(w(x, y, z), x, x, y, z, z, z)
    diff(w(x, y, z), x, x, x, z, z, z)
    diff(w(x, y, z), x, x, x, y, z, z)
    diff(w(x, y, z), x, x, x, y, y, z)
    diff(w(x, y, z), x, x, x, y, z, z, z)
    diff(w(x, y, z), x, x, x, y, y, z, z)
    >>> # example from Schwarz, pp 54
    >>> w = function("w")(x,y)
    >>> z = function("z")(x,y)
    >>> g1 = diff(z,y,y) + diff(z, y)/(2*y)
    >>> g5 = diff(z,x,x,x) + diff(w,y,y)*8*y**2 + diff(w,x,x)/y - diff(z,x,y)*4*y**2 - diff(z,x)*32*y-16*w
    >>> g6 = diff(z,x,x,y) - diff(z,y,y)*4*y**2 - diff(z,y)*8*y
    >>> ctx = Context((w,z),(x,y), Mgrlex)
    >>> dps=[LHDP(_, ctx) for _ in [g1,g5,g6]]
    >>> cs = CompleteSystem(dps, ctx)
    >>> for _ in cs: print(_)
    diff(z(x, y), y, y) + (1/2/y) * diff(z(x, y), y)
    diff(z(x, y), x, y, y) + (1/2/y) * diff(z(x, y), x, y)
    diff(z(x, y), x, x, y) + (-4*y^2) * diff(z(x, y), y, y) + (-8*y) * diff(z(x, y), y)
    diff(z(x, y), x, x, x) + (1/y) * diff(w(x, y), x, x) + (8*y^2) * diff(w(x, y), y, y) + (-4*y^2) * diff(z(x, y), x, y) + (-32*y) * diff(z(x, y), x) + (-16) * w(x, y)
    """
    s = bucket(S, key=lambda d: d.Lfunc())
    res = flatten([complete(s[k], context) for k in s])
    return Reorder(res, context, ascending=True)


def split_by_function(S, context):
    s = bucket(S, key=lambda d: d.Lfunc())
    murksi=[FindIntegrableConditions(s[k], context) for k in s]
    return flatten(murksi)

@profile
def FindIntegrableConditions(S, context):
    result = list(S)
    if len(result) == 1:
        return []

    vars = list(range(len(context.independent)))

    # reverse order as in context the highest independent is first,
    # but for multiplier computation it is last
    monomials = [(_, list(reversed(_.order))) for _ in result]

    ms = tuple([_[1] for _ in monomials])

    map_old_to_new = lambda i: context.independent[vars.index(len(vars)-1-i)]

    # multiplier-collection is our M
    multiplier_collection = []
    for dp, monom in monomials:
        # S1
        _multipliers, _nonmultipliers = vec_multipliers(monom, ms, vars)
        multiplier_collection.append(
            coll(monom, dp, [map_old_to_new(_) for _ in _multipliers], [map_old_to_new(_) for _ in _nonmultipliers]))


    result = []
    for ei, ej in pairs_exclude_diagonal(multiplier_collection):
        for n in ei.nonmultipliers:
            a1 = adiff(ei.dp.Lder(), context, n)
            for m in islice(powerset(ej.multipliers), 1, None):
                a2 = adiff(ej.dp.Lder(), context, *m)
                if a1 == a2:
                    # integrability condition
                    # don't need leading coefficients because in DPs
                    # it is always 1
                    d1 = ei.dp.diff(n)
                    d2 = ej.dp.diff(*m)
                    new_terms = []
                    terms_from_first = dict([(_.comparison_vector, _) for _ in d1.p])
                    # looks like overkill, but when coefficients get very
                    # groebnerish it pays to keep coefficients small, at least
                    # with maxim. Let's see how sympy behaves
                    for s in d2.p:
                        if s.comparison_vector in terms_from_first:
                            terms_from_first[s.comparison_vector].coeff -= s.coeff
                        else:
                            new_terms.append(
                                _Dterm(coeff = -s.coeff,
                                       derivative = s.derivative,
                                       context = context)
                            )
                    dterms = [_ for _ in new_terms + [*terms_from_first.values()] if _]
                    if dterms:
                        result.append(LHDP(e=0,
                                                               context=context,
                                                               dterms=dterms))
    return result


class Janet_Basis:

    def __init__(self, S, dependent, independent, sort_order=Mgrevlex):
        """
        Parameters:
            * List of homogenous PDE's
            * List of dependent variables, i.e. the functions to searched for
            * List of variables
            * sort order, default is grevlex

        >>> vars = var ("x y")
        >>> z = function("z")(*vars)
        >>> w = function("w")(*vars)
        >>> f1 = diff(w, y) + x*diff(z,y)/(2*y*(x**2+y)) - w/y
        >>> f2 = diff(z,x,y) + y*diff(w,y)/x + 2*y*diff(z, x)/x
        >>> f3 = diff(w, x,y) - 2*x*diff(z, x,2)/y - x*diff(w,x)/y**2
        >>> f4 = diff(w, x,y) + diff(z, x,y) + diff(w, y)/(2*y) - diff(w,x)/y + x* diff(z, y)/y - w/(2*y**2)
        >>> f5 = diff(w,y,y) + diff(z,x,y) - diff(w, y)/y + w/(y**2)
        >>> system_2_24 = [f1,f2,f3,f4,f5]
        >>> checkS=Janet_Basis(system_2_24, (w,z), vars)
        >>> checkS.show()
        diff(z(x, y), y)
        diff(z(x, y), x) + (1/2/y) * w(x, y)
        diff(w(x, y), y) + (-1/y) * w(x, y)
        diff(w(x, y), x)
        >>> vars = var ("x y")
        >>> z = function("z")(*vars)
        >>> w = function("w")(*vars)
        >>> g1 = diff(z, y,y) + diff(z,y)/(2*y)
        >>> g2 = diff(w,x,x) + 4*diff(w,y)*y**2 - 8*(y**2) * diff(z,x) - 8*w*y
        >>> g3 = diff(w,x,y) - diff(z,x,x)/2 - diff(w,x)/(2*y) - 6* (y**2) * diff(z,y)
        >>> g4 = diff(w,y,y) - 2*diff(z,x,y) - diff(w,y)/(2*y) + w/(2*y**2)
        >>> system_2_25 = [g2,g3,g4,g1]
        >>> checkS=Janet_Basis(system_2_25, (w,z), vars)
        >>> checkS.show()
        diff(z(x, y), y)
        diff(z(x, y), x) + (1/2/y) * w(x, y)
        diff(w(x, y), y) + (-1/y) * w(x, y)
        diff(w(x, y), x)
        >>> vars = var ("x y")
        >>> z = function("z")(*vars)
        >>> w = function("w")(*vars)
        >>> f1 = diff(w, y) + x*diff(z,y)/(2*y*(x**2+y)) - w/y
        >>> f2 = diff(z,x,y) + y*diff(w,y)/x + 2*y*diff(z, x)/x
        >>> f3 = diff(w, x,y) - 2*x*diff(z, x,2)/y - x*diff(w,x)/y**2
        >>> f4 = diff(w, x,y) + diff(z, x,y) + diff(w, y)/(2*y) - diff(w,x)/y + x* diff(z, y)/y - w/(2*y**2)
        >>> f5 = diff(w,y,y) + diff(z,x,y) - diff(w, y)/y + w/(y**2)
        >>> system_2_24 = [f1,f2,f3,f4,f5]
        >>> checkS=Janet_Basis(system_2_24, (w,z), vars, Mgrlex)
        >>> checkS.show()
        diff(z(x, y), y)
        diff(z(x, y), x) + (1/2/y) * w(x, y)
        diff(w(x, y), y) + (-1/y) * w(x, y)
        diff(w(x, y), x)
        >>> vars = var ("x y")
        >>> z = function("z")(*vars)
        >>> w = function("w")(*vars)
        >>> g1 = diff(z, y,y) + diff(z,y)/(2*y)
        >>> g2 = diff(w,x,x) + 4*diff(w,y)*y**2 - 8*(y**2) * diff(z,x) - 8*w*y
        >>> g3 = diff(w,x,y) - diff(z,x,x)/2 - diff(w,x)/(2*y) - 6* (y**2) * diff(z,y)
        >>> g4 = diff(w,y,y) - 2*diff(z,x,y) - diff(w,y)/(2*y) + w/(2*y**2)
        >>> system_2_25 = [g2,g3,g4,g1]
        >>> checkS=Janet_Basis(system_2_25, (w,z), vars, Mgrlex)
        >>> checkS.show()
        diff(z(x, y), y)
        diff(z(x, y), x) + (1/2/y) * w(x, y)
        diff(w(x, y), y) + (-1/y) * w(x, y)
        diff(w(x, y), x)
        """
        context = Context(dependent, independent, sort_order)
        if not isinstance(S, Iterable):
            # XXX bad criterion
            self.S = [S]
        else:
            self.S = S[:]
        old = []
        self.S = Reorder([LHDP(s, context, dterms=[]) for s in self.S], context, ascending=True)
        while 1:
            if old == self.S:
                # no change since last run
                return
            old = self.S[:]
            print("This is where we start")
            self.show(rich=True, short=False)
            import pdb; pdb.set_trace()
            self.S = Autoreduce(self.S, context)
            print("after autoreduce")
            self.show(rich=True, short=False)
            self.S = CompleteSystem(self.S, context)
            print("after complete system")
            self.show(rich=False, short=False)
            conditions = list(split_by_function(self.S, context))
            print("after conditions")
            print(conditions)
            reduced = [reduceS(_m, self.S, context) for _m in conditions]
            reduced = [_ for _ in reduced if _]
            print("after reduced", reduced)
            if not reduced:
                self.S = Reorder(self.S, context)
                return
            self.S += [_ for _ in reduced if not (_ in self.S or eq(_.expression(), 0))]
            self.S = Reorder(self.S, context, ascending=True)

    def show(self, rich=True, short=False):
        """Print the Janet basis with leading derivative first."""
        for _ in self.S:
            if rich:
                if _in_ipython_session:
                    display(Math(_.show()))
                else:
                    print([p.derivative for p in _.p])
            else:
                if not short:
                    print(_)
                else:
                    print([p.derivative for p in _.p])

    def rank(self):
        """Return the rank of the computed Janet basis."""
        return 0

    def order(self):
        """Return the order of the computed Janet basis which is the same as
        the rank
        """
        return self.rank()

    def type(self):
        '''Computes the type of the Janet Basis, i.e. the leading derivatives
        '''
        self._type = [_.Lder() for _ in self.S]


if __name__ == "__main__":
    import doctest
    doctest.testmod()
# -

# https://amirhashemi.iut.ac.ir/sites/amirhashemi.iut.ac.ir/files//file_basepage/invbasis.txt#overlay-context=contents

########### Pommaret Division #############
#def LeftPommaret(u,U,Vars):
#    local N,Ind,i
#    N=NULL
#    Ind=indets(u):
#    for i from 1 to nops(Vars) while not (Vars[i] in Ind):
#        N = N,Vars[i]
#    N = N,Vars[i]
#    return N

#def RightPommaret(u,U,Vars):
#    local N,Ind,i
#    N:=NULL
#    Ind:=indets(u)
#    for i from  nops(Vars) by -1 to 1 while not (Vars[i] in Ind):
#        N:=N,Vars[i]
#    N:=N,Vars[i]
#    return N
