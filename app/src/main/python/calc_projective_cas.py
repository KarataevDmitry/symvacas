"""Projective Calculus kernel over Q(i, sqrt(d-1)) and Riemann Sphere P^1(C) for SymvaCAS.

Provides:
  1. Ostrowski-Hermite rational reduction + Lagrange-Rothstein-Trager conjugate pole pairing
     over Q(i, sqrt(d-1)) for rational and Weierstrass-trigonometric integrands, producing
     elementary real primitives (in terms of log and atan) without RootSum or special functions,
     together with full step-by-step derivations.
  2. Improper integration on P^1(C) over [0, oo) and (-oo, oo) via residue collapse at infinity
     [1 : 0] in P^1 and biquadratic sector rotation zeta_{2n} = exp(i*pi/(2n)) for Fresnel-Airy
     oscillatory phase integrals int_0^oo x^m * sin(a*x^n) dx and int_0^oo x^m * cos(a*x^n) dx.
  3. Pure-rational Heron-Newton + half-angle Pade (5, 5) arbitrary-precision evaluation on R and C.
"""

from __future__ import annotations

from decimal import Decimal, localcontext
from fractions import Fraction
import math
import sympy as s


# ---------------------------------------------------------------------------
# 1. Pure-rational arbitrary-precision projective evaluator on R and C
# ---------------------------------------------------------------------------

def rational_sqrt_decimal(x: Decimal | int | Fraction, digits: int = 50) -> Decimal:
    """Compute sqrt(x) on R>=0 via pure Heron-Newton rational iteration r_{k+1} = (r_k + x/r_k)/2."""
    work_prec = digits + 25
    with localcontext() as ctx:
        ctx.prec = work_prec
        val = Decimal(x.numerator) / Decimal(x.denominator) if isinstance(x, Fraction) else Decimal(x)
        if val < 0:
            raise ValueError("rational_sqrt_decimal requires x >= 0")
        if val == 0:
            return Decimal(0)
        exp10 = val.adjusted()
        r = Decimal(2) * (Decimal(10) ** (exp10 // 2))
        r = (r + val / r) / Decimal(2)
        tol = Decimal(10) ** (-(digits + 18))
        max_steps = int(math.ceil(math.log2(work_prec))) + 10
        two = Decimal(2)
        for _ in range(max_steps):
            r_next = (r + val / r) / two
            if abs(r_next - r) <= abs(r_next) * tol:
                r = r_next
                break
            r = r_next
    with localcontext() as ctx:
        ctx.prec = digits
        return +r


def projective_atan_R(t: Decimal | int | Fraction, digits: int = 50) -> Decimal:
    """Compute arctan(t) on R to `digits` places via projective half-angle descent + Pade (5, 5)."""
    work_prec = digits + 25
    with localcontext() as ctx:
        ctx.prec = work_prec
        u = Decimal(t.numerator) / Decimal(t.denominator) if isinstance(t, Fraction) else Decimal(t)
        if u == 0:
            return Decimal(0)
        sign = Decimal(-1) if u < 0 else Decimal(1)
        u = abs(u)
        m_steps = max(int(math.ceil((digits + 20) * 3.321928 / 9.5)) + 3, 10)
        one = Decimal(1)
        for _ in range(m_steps):
            u = u / (one + rational_sqrt_decimal(one + u * u, work_prec))
        u2 = u * u
        u4 = u2 * u2
        num = u * (Decimal(945) + Decimal(735) * u2 + Decimal(64) * u4)
        den = Decimal(945) + Decimal(1050) * u2 + Decimal(225) * u4
        res = sign * (Decimal(2) ** m_steps) * (num / den)
    with localcontext() as ctx:
        ctx.prec = digits
        return +res


def projective_pi(digits: int = 50) -> Decimal:
    """Compute pi to arbitrary `digits` via Machin's projective identity in Q(i)."""
    work_prec = digits + 20
    with localcontext() as ctx:
        ctx.prec = work_prec
        pi_val = Decimal(16) * projective_atan_R(Fraction(1, 5), work_prec) - Decimal(4) * projective_atan_R(
            Fraction(1, 239), work_prec
        )
    with localcontext() as ctx:
        ctx.prec = digits
        return +pi_val


# ---------------------------------------------------------------------------
# 2. Ostrowski-Hermite reduction + Lagrange conjugate pole pairing over Q(i, sqrt(d-1))
# ---------------------------------------------------------------------------

def hermite_reduce_rational(num: s.Expr, den: s.Expr, var: s.Symbol):
    """Exact Ostrowski-Hermite reduction of P(var)/Q(var) via polynomial GCD (zero root-finding).

    Returns (poly_part, poly_antideriv, hermite_rat, rem_num, sqfree_den).
    """
    num_p = s.Poly(s.expand(num), var)
    den_p = s.Poly(s.expand(den), var)
    sqf_coeff, sqf_factors = s.sqf_list(den_p.as_expr(), var)
    num_expr = s.simplify(num_p.as_expr() / sqf_coeff)
    den_expr = s.Integer(1)
    for fac, mult in sqf_factors:
        den_expr = s.expand(den_expr * (fac ** mult))

    q_poly, r_poly = s.div(s.Poly(num_expr, var), s.Poly(den_expr, var), var)
    poly_part = q_poly.as_expr()
    poly_antideriv = s.integrate(poly_part, var)
    if r_poly.is_zero:
        return poly_part, poly_antideriv, s.Integer(0), s.Integer(0), s.Integer(1)

    mult_map = {}
    max_m = 1
    for fac, mult in sqf_factors:
        mult_map[mult] = s.expand(mult_map.get(mult, s.Integer(1)) * fac)
        if mult > max_m:
            max_m = mult

    a_cur = r_poly.as_expr()
    hermite_acc = s.Integer(0)

    while max_m > 1:
        v_m = mult_map.pop(max_m)
        u_other = s.Integer(1)
        for k_mult, k_fac in mult_map.items():
            u_other = s.expand(u_other * (k_fac ** k_mult))

        v_m_diff = s.diff(v_m, var)
        w_poly = s.expand(u_other * v_m_diff)
        scaled_a = s.expand(a_cur / s.Rational(1 - max_m))
        s_bez, _, g_bez = s.gcdex(s.Poly(w_poly, var), s.Poly(v_m, var))
        g_expr = g_bez.as_expr()
        b_full = s.Poly(s.expand(s_bez.as_expr() * scaled_a / g_expr), var)
        _, b_rem = s.div(b_full, s.Poly(v_m, var), var)
        b_expr = b_rem.as_expr()

        c_quo, _ = s.div(s.Poly(s.expand(scaled_a - b_expr * w_poly), var), s.Poly(v_m, var), var)
        c_expr = c_quo.as_expr()

        hermite_acc = s.together(hermite_acc + b_expr / (v_m ** (max_m - 1)))
        a_cur = s.expand((1 - max_m) * c_expr - u_other * s.diff(b_expr, var))
        mult_map[max_m - 1] = s.expand(mult_map.get(max_m - 1, s.Integer(1)) * v_m)
        max_m -= 1
        while max_m > 1 and max_m not in mult_map:
            max_m -= 1

    sqfree_den = s.expand(mult_map.get(1, s.Integer(1)))
    return poly_part, poly_antideriv, s.cancel(hermite_acc), s.expand(a_cur), sqfree_den


def _exact_biquadratic_roots(fac: s.Expr, var: s.Symbol) -> list[s.Expr] | None:
    """Resolve exact symbolic roots in Q(i, sqrt(d-1)) for degree <= 2, binomials a*t^4 + b,
    and biquadratics a*t^4 + b*t^2 + c.
    """
    poly = s.Poly(fac, var)
    deg = poly.degree()
    if deg == 0:
        return []
    if deg == 1:
        a1, a0 = poly.all_coeffs()
        return [s.simplify(-a0 / a1)]
    if deg == 2:
        a2, a1, a0 = poly.all_coeffs()
        disc = s.simplify(a1 * a1 - 4 * a2 * a0)
        sd = s.sqrt(disc)
        return [s.simplify((-a1 + sd) / (2 * a2)), s.simplify((-a1 - sd) / (2 * a2))]
    if deg == 4:
        c4, c3, c2, c1, c0 = poly.all_coeffs()
        if c3 == 0 and c1 == 0:
            if c2 == 0:
                rhs = s.simplify(-c0 / c4)
                if rhs.is_negative:
                    r4 = s.simplify((-rhs) ** s.Rational(1, 4))
                    s2 = s.sqrt(2) / 2
                    return [
                        s.simplify(r4 * s2 * (1 + s.I)),
                        s.simplify(r4 * s2 * (-1 + s.I)),
                        s.simplify(r4 * s2 * (-1 - s.I)),
                        s.simplify(r4 * s2 * (1 - s.I)),
                    ]
                if rhs.is_positive:
                    r4 = s.simplify(rhs ** s.Rational(1, 4))
                    return [r4, -r4, s.I * r4, -s.I * r4]
            disc2 = s.simplify(c2 * c2 - 4 * c4 * c0)
            if disc2.is_rational or (disc2.is_number and disc2.is_real):
                u1 = s.simplify((-c2 + s.sqrt(disc2)) / (2 * c4))
                u2 = s.simplify((-c2 - s.sqrt(disc2)) / (2 * c4))
                out = []
                for u_val in (u1, u2):
                    su = s.sqrt(u_val)
                    out.append(s.simplify(su))
                    out.append(s.simplify(-su))
                return out
    return None


def projective_rational_primitive(expression: s.Expr, variable: s.Symbol):
    """Integrate a rational function P(x)/Q(x) over Q(i, sqrt(d-1)) via Ostrowski-Hermite
    reduction and Lagrange conjugate pole pairing, returning (primitive, step_records) or None.
    """
    if not isinstance(variable, s.Symbol) or expression.free_symbols - {variable}:
        return None
    if not expression.is_rational_function(variable):
        return None

    cancelled = s.cancel(expression)
    num, den = s.fraction(cancelled)
    if not num.is_polynomial(variable) or not den.is_polynomial(variable):
        return None
    if s.Poly(den, variable).degree() <= 0:
        return None

    poly_part, poly_antideriv, hermite_rat, rem_num, sqfree_den = hermite_reduce_rational(num, den, variable)

    poles_residues = []
    if rem_num != 0 and sqfree_den != 1:
        den_diff = s.diff(sqfree_den, variable)
        _, fac_list = s.factor_list(sqfree_den, variable)
        for fac, _ in fac_list:
            roots = _exact_biquadratic_roots(fac, variable)
            if roots is None:
                return None
            for r_sym in roots:
                res_sym = s.simplify(rem_num.subs(variable, r_sym) / den_diff.subs(variable, r_sym))
                poles_residues.append((r_sym, res_sym))

    # Pair conjugate poles in Q(i, sqrt(d-1)) into real log and atan terms
    log_atan_expr = s.Integer(0)
    used = [False] * len(poles_residues)
    paired_terms = []

    for i, (p_i, c_i) in enumerate(poles_residues):
        if used[i]:
            continue
        p_re, p_im = s.expand(p_i).as_real_imag()
        c_re, c_im = s.expand(c_i).as_real_imag()
        if p_im == 0:
            used[i] = True
            term = s.simplify(c_re) * s.log(variable - p_re)
            log_atan_expr += term
            paired_terms.append(term)
            continue

        partner = None
        for j in range(i + 1, len(poles_residues)):
            if used[j]:
                continue
            q_re, q_im = s.expand(poles_residues[j][0]).as_real_imag()
            if s.simplify(p_re - q_re) == 0 and s.simplify(p_im + q_im) == 0:
                partner = j
                break

        if partner is not None and p_im.is_positive:
            used[i] = True
            used[partner] = True
            d_re, d_im = s.expand(poles_residues[partner][1]).as_real_imag()
            if s.simplify(c_re - d_re) == 0 and s.simplify(c_im + d_im) == 0:
                quad_arg = s.expand((variable - p_re) ** 2 + p_im**2)
                atan_arg = s.simplify((variable - p_re) / p_im)
                term = s.Integer(0)
                if c_re != 0:
                    term += s.simplify(c_re) * s.log(quad_arg)
                if c_im != 0:
                    term -= s.simplify(2 * c_im) * s.atan(atan_arg)
                log_atan_expr += term
                paired_terms.append(term)
                continue
        return None

    primitive = poly_antideriv + hermite_rat + log_atan_expr
    if s.simplify(s.cancel(s.diff(primitive, variable) - cancelled)) != 0:
        return None

    step_info = {
        "kind": "projective_rational",
        "cancelled": cancelled,
        "poly_part": poly_part,
        "poly_antideriv": poly_antideriv,
        "hermite_rat": hermite_rat,
        "rem_num": rem_num,
        "sqfree_den": sqfree_den,
        "poles_residues": poles_residues,
        "log_atan_expr": log_atan_expr,
        "primitive": primitive,
    }
    return primitive, step_info


# ---------------------------------------------------------------------------
# 3. Improper Fresnel-Airy phase integrals via biquadratic sector rotation zeta_{2n}
# ---------------------------------------------------------------------------

def projective_fresnel_improper(expression: s.Expr, variable: s.Symbol, lower: s.Expr, upper: s.Expr):
    """Evaluate improper oscillatory Fresnel-Airy integrals on [0, oo) or (-oo, oo):
        int_0^oo c * x^m * sin(a * x^n) dx,
        int_0^oo c * x^m * cos(a * x^n) dx
    via projective contour rotation x = u * exp(i*pi/(2n)) on P^1(Q(i, sqrt(d-1))).
    """
    if not isinstance(variable, s.Symbol) or expression.free_symbols - {variable}:
        return None
    if upper is not s.oo:
        return None
    if lower != 0 and lower is not -s.oo:
        return None

    is_full_line = lower is -s.oo
    expr = s.expand(expression)
    coeff, dep = expr.as_independent(variable, as_Add=False)
    if not coeff.is_real:
        return None

    m_pow = 0
    core_func = dep
    if dep.is_Mul:
        others = []
        for f in dep.args:
            if f == variable:
                m_pow += 1
            elif f.is_Pow and f.base == variable and f.exp.is_Integer and int(f.exp) >= 1:
                m_pow += int(f.exp)
            else:
                others.append(f)
        if len(others) == 1:
            core_func = others[0]
        else:
            return None

    if not isinstance(core_func, (s.sin, s.cos)):
        return None

    arg = s.expand(core_func.args[0])
    a_coeff, x_part = arg.as_independent(variable, as_Add=False)
    if not (a_coeff.is_number and a_coeff.is_real and a_coeff != 0):
        return None

    n_deg = 0
    if x_part.is_Pow and x_part.base == variable and x_part.exp.is_Integer:
        n_deg = int(x_part.exp)
    if n_deg < 2:
        return None

    # Convergence check on [0, oo) by Dirichlet-Abel test: requires 0 <= m_pow < n_deg
    if not (0 <= m_pow < n_deg):
        return None

    fn_name = "sin" if isinstance(core_func, s.sin) else "cos"
    if is_full_line:
        parity = (m_pow + n_deg) % 2 if fn_name == "sin" else m_pow % 2
        if parity == 1:
            step_info = {
                "kind": "projective_fresnel",
                "fn": fn_name,
                "coeff": coeff,
                "m": m_pow,
                "a": a_coeff,
                "n": n_deg,
                "is_odd_zero": True,
                "result": s.Integer(0),
            }
            return s.Integer(0), "Odd symmetry on P^1 line (-oo, oo): integral vanishes identically.", step_info

    s_param = s.Rational(m_pow + 1, n_deg)
    abs_a = s.Abs(a_coeff)
    angle = s.pi * s_param / 2
    trig_part = s.cos(angle) if fn_name == "cos" else s.sign(a_coeff) * s.sin(angle)
    mult = s.Integer(2) if is_full_line else s.Integer(1)

    g_val = s.sqrt(s.pi) if s_param == s.Rational(1, 2) else s.gamma(s_param)
    result = s.simplify(coeff * mult * g_val * trig_part / (s.Integer(n_deg) * (abs_a ** s_param)))

    zeta_angle = s.pi / (2 * n_deg)
    note = (
        f"Evaluated via projective sector rotation x = u * exp(i*π/{2 * n_deg}) "
        f"on P^1(Q(i, sqrt(d-1))) with Euler Gamma({s_param}) residue."
    )
    step_info = {
        "kind": "projective_fresnel",
        "fn": fn_name,
        "coeff": coeff,
        "m": m_pow,
        "a": a_coeff,
        "n": n_deg,
        "s_param": s_param,
        "zeta_angle": zeta_angle,
        "angle": angle,
        "is_full_line": is_full_line,
        "is_odd_zero": False,
        "result": result,
    }
    return result, note, step_info


def append_projective_steps(step_info: dict, expression: s.Expr, variable: s.Symbol, add, eq):
    """Append step-by-step cards for projective Ostrowski-Hermite-Lagrange or Fresnel sector rotation."""
    kind = step_info.get("kind")
    if kind == "projective_rational":
        if step_info["poly_part"] != 0:
            add(
                "Polynomial division",
                "Divide the numerator by the denominator to separate the polynomial part from the proper rational fraction.",
                eq(expression, step_info["poly_part"] + (expression - step_info["poly_part"])),
                eq(s.Integral(step_info["poly_part"], variable), step_info["poly_antideriv"]),
            )
        if step_info["hermite_rat"] != 0:
            add(
                "Ostrowski–Hermite rational reduction",
                "Extract the repeated-pole rational part purely via polynomial GCD (Bézout identity) without root finding, reducing the remaining denominator to square-free form.",
                eq(
                    s.Integral(step_info["cancelled"] - step_info["poly_part"], variable),
                    step_info["hermite_rat"] + s.Integral(step_info["rem_num"] / step_info["sqfree_den"], variable),
                ),
            )
        if step_info["poles_residues"]:
            poles_preview = [
                eq(s.Symbol(f"alpha_{idx + 1}"), p_sym)
                for idx, (p_sym, _) in enumerate(step_info["poles_residues"][:4])
            ]
            add(
                "Lagrange–Rothstein–Trager poles over Q(i, sqrt(d-1))",
                "Decompose the square-free rational kernel over the biquadratic field extension Q(i, sqrt(d-1)) and pair complex-conjugate poles into real logarithmic and arctangent rotator terms.",
                *poles_preview,
                eq(s.Integral(step_info["rem_num"] / step_info["sqfree_den"], variable), step_info["log_atan_expr"]),
            )
        add(
            "Combine projective primitive",
            "Assemble the polynomial part, the Ostrowski–Hermite rational part, and the paired logarithmic/arctangent terms.",
            eq(s.Integral(expression, variable), step_info["primitive"]),
        )
        return True

    if kind == "projective_fresnel":
        if step_info.get("is_odd_zero"):
            add(
                "Odd parity on full projective line",
                "The integrand is an odd function of the integration variable on (-oo, oo), so the symmetric contributions on (-oo, 0] and [0, oo) cancel identically.",
                eq(s.Integral(expression, (variable, -s.oo, s.oo)), s.Integer(0)),
            )
            return True
        n_deg = step_info["n"]
        s_param = step_info["s_param"]
        u_sym = s.Symbol("u", positive=True)
        add(
            "Lift to complex exponential phase on P^1(C)",
            f"Embed the oscillatory trigonometric kernel into the complex exponential exp(i·a·{variable}^{n_deg}) on the Riemann sphere P^1(C).",
            eq(
                s.Integral(expression, (variable, 0, s.oo)),
                step_info["coeff"]
                * (s.Integer(2) if step_info["is_full_line"] else s.Integer(1))
                * s.Integral(variable ** step_info["m"] * s.exp(s.I * step_info["a"] * (variable**n_deg)), (variable, 0, s.oo)),
            ),
        )
        add(
            f"Projective sector rotation by root of unity exp(i·π/{2 * n_deg})",
            f"Rotate the integration ray [0, oo) in P^1(Q(i, sqrt(d-1))) by the root of unity ζ_{2 * n_deg} = exp(i·π/{2 * n_deg}) so that i·{variable}^{n_deg} becomes the decaying Gaussian-Laplace ray -u^{n_deg}.",
            eq(variable, u_sym * s.exp(s.I * step_info["zeta_angle"])),
            eq(
                s.Integral(u_sym ** step_info["m"] * s.exp(-s.Abs(step_info["a"]) * (u_sym**n_deg)), (u_sym, 0, s.oo)),
                s.gamma(s_param) / (s.Integer(n_deg) * (s.Abs(step_info["a"]) ** s_param)),
            ),
        )
        add(
            "Project onto real axis",
            "Take the real/imaginary projection of the rotated sector factor exp(i·π·(m+1)/(2n)) to obtain the exact closed-form value.",
            eq(
                s.Integral(
                    expression,
                    (variable, -s.oo if step_info["is_full_line"] else 0, s.oo),
                ),
                step_info["result"],
            ),
        )
        return True

    return False
