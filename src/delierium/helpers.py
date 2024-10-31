"""Convenience functions"""

import itertools
import random
import re
from functools import cache

import more_itertools
from IPython.core.debugger import set_trace  # type: ignore

from typing import Iterable, Tuple, Any, Generator, TypeAlias

from sympy import *
from sympy.core.relational import Equality
from sympy.core.numbers import Integer, Rational, Zero, One
from sympy import ordered, sympify

from sympy.core.backend import *

from line_profiler import profile

@profile
def eq(d1, d2):
    if d1.__class__ != d2.__class__:
        return False
    return d1 == d2


@profile
def is_numeric(e):
    return type(e) in (Integer, Rational, int, float, complex, Zero, One) \
        and not type(e) == bool

@profile
def expr_eq(e1, e2):
    # '==' is structural equality
    e1 = sympify(e1)
    e2 = sympify(e2)
    res = e1 == e2
    return res

    return (sympify(e1) - sympify(e2)).simplify() == 0
    if e1 != e2:
        return False
    # a sum and a product can't be equal
    if type(e1) != type(e2) and (type(e1) in [Add, Mul] or type(e2) in [Add, Mul]):
        return False

    if is_numeric(e1) and is_numeric(e2):
        return e1 == e2
    if not is_numeric(e1) and is_numeric(e2):
        return False
    if not is_numeric(e2) and is_numeric(e1):
        return False
    if e1.free_symbols != e2.free_symbols:
        return False
    for i in e1.free_symbols:
        r = random.randint(10, 50)
        e1 = e1.subs(i, r)
        e2 = e2.subs(i, r)
        res = e1 == e2
        if type(res) == bool:
            return res
    raise NotImplementedError(f"this kind of comparison is not implemented yet {e1=}, {e2=}")

@profile
def expr_is_zero(e):
    return e == 0
    try:
        if e.is_numeric():
            return e == 0
    except AttributeError:
        return e == 0
    return False

    try:
        vars=e.variables()
    except AttributeError:
        return bool(e == 0)
    rlist = [random.randint(100, 1_000) for i in range(len(vars))]
    try:
        ev = e.subs(dict(zip(vars, rlist)))
    except AttributeError:
        ev = e
    # XXX make test twice
    return bool(ev == 0)



def pairs_exclude_diagonal(it):
    for x, y in itertools.product(it, repeat=2):
        if x != y:
            yield (x, y)


def tangent_vector(f):
    # https://doc.sagemath.org/html/en/reference/manifolds/sage/manifolds/differentiable/tangent_vector.html?highlight=partial%20differential
    # XXX:  There is TangentVector in Sage but a little bit more complicated.
    # Does it pay to use that one ?
    r"""
    Do a tangent vector

    DEFINITION:

    missing

    INPUT:

    - ``f`` - symbolic expression of type 'function'

    OUTPUT:

    the tangent vector

    .. NOTE::

    none so far

    ..

    EXAMPLES: compute the tangent vector of

    ::
    sage: from delierium.helpers import tangent_vector
    sage: x,y,z = var ("x y z")
    sage: tangent_vector (x**2 - 3*y**4 - z*x*y + z - x)
    [-y*z + 2*x - 1, -12*y^3 - x*z, -x*y + 1]
    sage: tangent_vector (x**2 + 2*y**3 - 3*z**4)
    [2*x, 6*y^2, -12*z^3]
    sage: tangent_vector (x**2)
    [2*x]
    """
    t = var("t")
    newvars = [var("x%s" % i) for i in f.variables()]
    for o, n in zip(f.variables(), newvars):
        f = f.subs({o: o + t * n})
    d = diff(f, t).limit(t=0)
    return [d.coefficient(_) for _ in newvars]


def is_derivative(e):
    """checks whether an expression 'e' is a pure derivative

    >>> from delierium.helpers import is_derivative
    >>> x = symbols('x')
    >>> f = Function('f')(x)
    >>> is_derivative (f)
    False
    >>> is_derivative (diff(f,x))
    True
    >>> is_derivative (diff(f,x)*x)
    False
    """
    return e.is_Derivative


def is_function(e) -> bool:
    """checks whether an expression 'e' is a pure function without any
    derivative as a factor
    """
    return e.is_Function


def compactify(*vars):
    pairs = list(more_itertools.pairwise(vars))
    if not pairs:
        return [vars[0]]
    result = []
    for pair in pairs:
        if isinstance(pair[0], Integer):
            continue
        if isinstance(pair[1], Integer):
            result.extend([pair[0]] * pair[1])
        else:
            result.append(pair[0])
    return result


@profile
def _adiff(f, *vars):
    return f.diff(*vars)

@profile
def adiff(f, context, *vars):
    return _adiff(f, *tuple(vars))
    return  f.diff(*vars)

    use_func_diff = any(type(v) == Function for v in vars)
    for op in f.operands():
        if "NewSymbolicFunction" in op.operator().__class__.__name__:
            use_func_diff = True
            break
    if use_func_diff:
        for v in vars:
            if "NewSymbolicFunction" in v.__class__.__name__:
                f = func_diff(f, v(context._independent[1]))
            else:
                xx = SR.var("xx")
                gg = f.subs(
                    {context._dependent[0](context._independent[1]): xx})
                gg = diff(gg, v)
                f = gg.subs(
                    {xx: context._dependent[0](context._independent[1])})
    else:
        f = f.diff(*vars)
    return f


def is_op_du(expr_op, u):
    is_derivative = isinstance(expr_op,
                               sage.symbolic.operators.FDerivativeOperator)
    if is_derivative:
        # Returns True if the differentiated function is `u`.
        return expr_op.function() == u.operator()
    else:
        return False


def iter_du_orders(expr, u):
    for sub_expr in expr.operands():
        if sub_expr == []:
            # hit end of tree
            continue
        if is_op_du(sub_expr.operator(), u):
            # yield order of differentiation
            yield len(sub_expr.operator().parameter_set())
        else:
            # iterate into sub expression
            for order in iter_du_orders(sub_expr, u):
                yield order


def func_diff(L, u_in):
    """`u` must be a callable symbolic expression"""
    #    https://ask.sagemath.org/question/7929/computing-variational-derivatives/
    x = u_in.variables()[0]
    u = u_in.function(x)

    # This variable name must not collide
    # with an existing one.
    # who will call a variable "tapir"
    # nobody else does this...
    t = SR.var("tapir")
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


class ExpressionGraph:
    """simple internal helper class
    analyzes the expression as a tree and stores the latex expression
    for each subexpression
    stolen from https://ask.sagemath.org/question/58145/tree-representing-an-expression/
    and adapted accordingly, quick 'n dirty
    """

    def __init__(self, expr):
        self.G = Graph()
        self.i = 0
        self.expr = expr
        self.root = None
        self.latex_names = {}
        self.funcs_found = set()
        self.graph_expr(self.expr)

    def plot(self, *args, **kwds):
        # print ("root is {0}".format(self.root))
        return self.G.plot(*args, layout="tree", tree_root=self.root, **kwds)

    def graph_expr(self, expr):
        # print("."*80)
        # print (expr, expr.__class__)
        # set_trace()
        self.latex_names[str(expr)] = expr._latex_()
        try:
            operator = expr.operator()
        except AttributeError:  # e.g. if expr is an integer
            operator = None
        # print(f"{operator=} {operator.__class__=}")
        if operator is None:
            name = "[{0}] {1}".format(self.i, expr)
            # print(f"{self.i=}")
            # print(f"(leaf) {expr=} {expr.__class__=}")
            self.latex_names[str(expr)] = expr._latex_()
            self.i += 1
            self.G.add_vertex(name)
            return name
        else:
            try:
                name = "[{0}] {1}".format(self.i, operator.__name__)
                # print(f"named {self.i=} {name=}")
            except AttributeError:
                if "FDerivativeOperator" in operator.__class__.__name__:
                    self.latex_names[str(
                        operator.function())] = operator.function()._latex_()
                    name = "FDerivativeOperator"
                    # print(f"unnamed {self.i=} {name=}")
                elif "NewSymbolicFunction" in operator.__class__.__name__:
                    name = "[{0}] {1}".format(self.i, str(operator))
                    # print(f"unnamed {self.i=} {name=}")
                    self.funcs_found.add(expr)
                    # print("AAA")
                else:
                    name = "[{0}] {1}".format(self.i, str(operator))
                # print(f"unnamed {self.i=} {name=}")
            try:
                self.latex_names[str(operator)] = operator._latex_()
            except AttributeError:
                self.latex_names[str(expr)] = expr._latex_()
            if self.i == 0:
                self.root = name
                # print("  ** root is '{0}' **".format(self.root))
            self.i += 1
            new_nodes = []
            # print(f"{expr.operands()=}")
            for opnd in expr.operands():
                new_nodes += [self.graph_expr(opnd)]
            self.G.add_vertex(name)
            self.G.add_edges([(name, node) for node in new_nodes])
            return name


def latexer(e):
    """Converts any SageMath expression into a TraditionalForm.
    Linear differential polynomials have their on latex style, but we don't
    have them always i hand, so this may still be useful
    """
    re_diff1 = re.compile(
        r".*(?P<D>D\[)(?P<vars>.+)\]\((?P<f1>[^\)]+)\)\((?P<args>\S*\), [^)]\)).*"
    )
    nakedf = re.compile(r"^(?P<fname>\w+)\(.*$")
    pat = r".*(diff\((?P<funcname>\w+)(?P<vars>\([a-zA-Z ,]+\)), (?P<diffs>[a-zA-Z ,]+)\))"
    r = re.compile(r"%s" % pat)
    teststring = str(e.expand())
    graph = ExpressionGraph(e)
    funcs_found=graph.funcs_found
    latexdict = graph.latex_names
    funcabbrevs = set()
    while match := r.match(teststring):
        # check 'diff'
        res = "%s_{%s}" % (
            match.groupdict()["funcname"],
            ",".join(match.groupdict()["diffs"].split(",")),
        )
        funcabbrevs.add((
            match.groupdict()["funcname"] + ",".join(match.groupdict()["vars"]),
            match.groupdict()["funcname"],
        ))
        teststring = teststring.replace(match.groups(0)[0], res)
    while match := re_diff1.match(teststring):
        # check 'D[...]'
        # set_trace()
        params = match.groupdict()["args"].split(",")
        params = [_.strip() for _ in params]
        # XXX not sure this will work properly. What if params is ['y(x)']
        # and not ['y(x)','x)'] ? in that case we will fail ...
        params[-1] = params[-1].replace(")", "")
        fu = params[0]
        vv = [int(_) for _ in match.groupdict()["vars"].split(",")]

        f1 = match.groupdict()["f1"]
        to_replace = "".join((
            "D[",
            match.groupdict()["vars"],
            "]",
            "(",
            f1,
            ")(",
            match.groupdict()["args"],
        ))
        vars = [_.replace(")", "") for _ in params][1:]
        downvar = ""
        for _v in vv:
            if m := nakedf.match(params[_v]):
                downvar += ",".join(m.groupdict()["fname"])
            else:
                downvar += params[_v]
        teststring = teststring.replace(to_replace, r" %s_{%s}" % (f1, downvar))
        teststring = teststring.replace(f1, latexdict.get(f1, f1))
        args = match.groupdict()["args"][:-1]  # remove trailing ")"
        args = args.split(",")
        funcabbrevs.add((args[0], nakedf.match(args[0]).groupdict()["fname"]))

    for f in funcs_found:
        # matches phi(y(x), x) with:
        # outer = phi
        # inner = y
        # innervars = x
        # outervars = x
        # set_trace()
        nested_function = re.compile(
            r"^(?P<outer>\w+)\((?P<inner>\w+)\((?P<innervars>[\w+ ,]+)\), (?P<outervars>[\w ,]+)\)$"
        )
        if match := nested_function.match(str(f)):
            res = "%s(%s(%s), %s)" % (
                match.groupdict()["outer"],
                match.groupdict()["inner"],
                match.groupdict()["innervars"],
                match.groupdict()["outervars"],
            )
            teststring = teststring.replace(res, match.groupdict()["outer"])

    for f in funcs_found:
        simple_function = re.compile(r"^(?P<outer>\w+)\((?P<args>[\w ,]+)\)$")
        # matches y(x, z) with:
        # outer = y
        # args = x, z
        # matches  y(x) with:
        # outer = y
        # args = x
        if match := simple_function.match(str(f)):
            res = "%s(%s)" % (match.groupdict()["outer"],
                              match.groupdict()["args"])
            teststring = teststring.replace(res, match.groupdict()["outer"])
    for fu in funcabbrevs:
        teststring = teststring.replace(fu[0], fu[1])
    return teststring.replace("*", " ")


class ExpressionTree:
    """simple internal helper class
    analyzes the expression as a tree and stores the latex expression
    for each subexpression
    stolen from https://ask.sagemath.org/question/58145/tree-representing-an-expression/
    and adapted accordingly, quick 'n dirty
    """

    def __init__(self, expr):
        self.root = None
        self.latex_names = {}
        self.gschisti = set()
        self._expand(expr, self.root)
        self.diffs = set([
            node.value
            for node in PreOrderIter(self.root)
            if node.value.operator().__class__ == FDerivativeOperator
        ])
        self.funcs = set([
            node.value
            for node in PreOrderIter(self.root)
            if node.value.operator().__class__.__name__ == "NewSymbolicFunction"
        ])
        self.powers = set([
            node.value
            for node in PreOrderIter(self.root)
            if str(node.value.operator()) == "<built-in function pow>"
        ])
        self.latex = set([
            (node.value, node.latex) for node in PreOrderIter(self.root)
        ])

    def _expand(self, e, parent):
        try:
            opr = e.operator()
        except AttributeError:  # e.g. if expr is an integer
            opr = None
        l = ""
        if opr:
            if "FDerivativeOperator" in opr.__class__.__name__:
                l = "%s_{%s}" % (
                    opr.function()._latex_(),
                    ",".join([str(_) for _ in e.operands()]),
                )
            elif "NewSymbolicFunction" in opr.__class__.__name__:
                l = opr._latex_()
            else:
                try:
                    l = e._latex_()
                except AttributeError:
                    l = ""
        try:
            self.latex_names[str(opr)] = opr._latex_()
        except AttributeError:
            self.latex_names[str(e)] = e._latex_()

        n = Node(str(e), value=e, operator=opr, parent=parent, latex=l)
        self.root = n if self.root is None else self.root
        if opr is not None:
            try:
                ops = e.operands()
            except AttributeError:  # can that really happen?
                ops = []
            for o in ops:
                if "FDerivativeOperator" in o.operator().__class__.__name__:
                    self.gschisti.add(o)
                if hasattr(o.operator(),
                           "__name__") and o.operator().__name__ == "pow":
                    for _o in o.operands():
                        if "FDerivativeOperator" in _o.operator(
                        ).__class__.__name__:
                            self.gschisti.add(o)
                            self.gschisti.add(e)
                self._expand(o, n)


class BSTNode:
    def __init__(self, val=None):
        self.left = None
        self.right = None
        self.val = val


    def insert(self, val):
        if val:
            existing_node = self.get_val_with_key(val.comparison_vector)
            if existing_node:
                existing_node.coeff += val.coeff
                return
        if not self.val:
            self.val = val
            return
        if val.context.lt(val.comparison_vector, self.val.comparison_vector):
            if self.left:
                self.left.insert(val)
                return
            self.left = BSTNode(val)
            return
        if self.right:
            self.right.insert(val)
            return
        self.right = BSTNode(val)
    def get_min(self):
        current = self
        while current.left is not None:
            current = current.left
        return current.val

    def get_max(self):
        current = self
        while current.right is not None:
            current = current.right
        return current.val

    def delete(self, val):
        if self == None:
            return self
        if self.val.context.gt(val.comparison_vector, self.val.comparison_vector):
            self.right = self.right.delete(val)
            return self
        if self.val.context.lt(val.comparison_vector, self.val.comparison_vector):
            self.left = self.left.delete(val)
            return self
        if self.right == None:
            return self.left
        if self.left == None:
            return self.right
        min_larger_node = self.right
        while min_larger_node.left:
            min_larger_node = min_larger_node.left
        self.val = min_larger_node.val
        self.right = self.right.delete(min_larger_node.val)
        return self

    def exists(self, val):
        if val == self.val:
            return True

        if val.context.lt(val.comparison_vector, self.val.comparison_vector):
            if self.left == None:
                return False
            return self.left.exists(val)

        if self.right == None:
            return False
        return self.right.exists(val)

    def get_val_with_key(self, key):
        if self.val is None:
            return None
        if key == self.val.comparison_vector:
            return self.val
        if self.val.context.gt(key, self.val.comparison_vector):
            if self.right == None:
                return None
            return self.right.get_val_with_key(key)

        if self.left == None:
            return None
        return self.left.get_val_with_key(key)

    def inorder(self, vals):
        if self.left is not None:
            self.left.inorder(vals)
        if self.val is not None:
            vals.append(self.val)
        if self.right is not None:
            self.right.inorder(vals)
        return vals

    def reverseorder(self, vals):
        if self.right is not None:
            self.right.reverseorder(vals)
        if self.val is not None:
            vals.append(self.val)
        if self.left is not None:
            self.left.reverseorder(vals)
        return vals

    min_to_max =  inorder
    max_to_min = reverseorder

class BSTNode_DP(BSTNode):
    def insert(self, val):
        if not self.val:
            self.val = val
            return
        if val.context.lt(val.comparison_vector, self.val.comparison_vector):
            if self.left:
                self.left.insert(val)
                return
            self.left = BSTNode(val)
            return
        if self.right:
            self.right.insert(val)
            return
        self.right = BSTNode(val)


# ToDo (from AllTypes.de
#    cfdgfdgfd
# CommutatorTable
# DterminingSystem
# Free Resolution
# Gcd
# Groebner Basis
# In?
# Intersection
# JanetBasis
# Lcm
# LöwDecomposioipn
# Primary Decomposition
# Product
# Qutiont
# Radical
# Random
# Saturation
# Sum
# Symmetric Power
# # Syzygys
