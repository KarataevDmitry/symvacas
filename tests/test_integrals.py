"""Derivative and domain checks for the rational trig substitution fallback."""
import pathlib
import sys
import unittest

sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1]/"app/src/main/python"))
import sympy as s
from calc_integrals import rational_trig_primitive
from calc_shared import numeric_integral

class PiecewiseNumericIntegralTests(unittest.TestCase):
    def test_quadrature_splits_root_cusps_and_disjoint_domains(self):
        x = s.Symbol("x", real=True)
        root = s.Piecewise((x**s.Rational(1, 3), x >= 0), (-(-x)**s.Rational(1, 3), True))
        self.assertAlmostEqual(48.75, float(numeric_integral(root, x, -8, 27, 30)), delta=1e-12)
        base = 5-x**2
        curve = s.Piecewise((base**s.Rational(1, 3), base >= 0), (-(-base)**s.Rational(1, 3), True))
        result = numeric_integral(curve, x, -3, 3, 50)
        self.assertTrue(result.is_real and result.is_finite)
        self.assertAlmostEqual(4.65378418486832451445, float(result), delta=1e-12)

    def test_piecewise_preserves_cancellation_small_values_and_complex_parts(self):
        x = s.Symbol("x", real=True)
        tiny = s.Rational(1, 10**100)
        expression = s.Piecewise((x, x < 0), (x+tiny, True))
        self.assertEqual(s.Float(tiny, 30), numeric_integral(expression, x, -1, 1, 30))
        complex_curve = s.Piecewise((s.I*x, x < 0), (2*s.I*x, True))
        self.assertEqual(0, numeric_integral(complex_curve, x, -1, 1, 30)-s.I/2)
        constant = s.Piecewise((s.Float(1), x < 0), (s.Integer(1), True))
        self.assertEqual(0, numeric_integral(constant, x, -1, 1, 30)-2)


class RationalTrigIntegralTests(unittest.TestCase):
    def test_primitives_differentiate_to_original_integrands(self):
        x=s.Symbol("x")
        expressions=[s.sqrt(s.tan(x)),s.sqrt(s.tan(2*x+1)),s.sqrt(s.tan(-2*x+1)),
                     s.sqrt(s.cot(x)),1/s.sqrt(s.tan(x)),s.tan(x)**s.Rational(1,3)]
        for expression in expressions:
            with self.subTest(expression=expression):
                primitive,conditions,note=rational_trig_primitive(expression,x)
                self.assertFalse(primitive.has(s.Integral))
                self.assertEqual(0,s.trigsimp(s.simplify(s.diff(primitive,x)-expression)))
                self.assertIn(s.Eq(s.im(x),0),conditions)
                self.assertIn("continuous real interval",note)
        t=s.Symbol("t",positive=True)
        formula=(s.atan(s.sqrt(2)*t-1)+s.atan(s.sqrt(2)*t+1))/s.sqrt(2)+s.log((t**2-s.sqrt(2)*t+1)/(t**2+s.sqrt(2)*t+1))/(2*s.sqrt(2))
        self.assertEqual(0,s.cancel(s.diff(formula,t)-2*t**2/(1+t**4)))


class ProjectiveCalculusTests(unittest.TestCase):
    def test_projective_hermite_lagrange_and_fresnel_sector_rotation(self):
        import json
        import calc_engine
        from calc_projective_cas import projective_fresnel_improper, projective_pi, projective_rational_primitive

        x = s.Symbol("x", real=True)
        # 1. Repeated-pole biquadratic rational integral via Ostrowski-Hermite + Lagrange over Q(i, sqrt(2))
        prim, step_info = projective_rational_primitive(x**2 / (1 + x**4) ** 2, x)
        self.assertFalse(prim.has(s.Integral, s.RootSum))
        self.assertEqual(0, s.simplify(s.cancel(s.diff(prim, x) - x**2 / (1 + x**4) ** 2)))
        self.assertNotEqual(0, step_info["hermite_rat"])

        # 2. Improper Fresnel-Airy integrals via sector rotation zeta_{2n}
        res_f2, note_f2, info_f2 = projective_fresnel_improper(s.sin(x**2), x, s.Integer(0), s.oo)
        self.assertEqual(0, s.simplify(res_f2 - s.sqrt(2 * s.pi) / 4))
        self.assertIn("sector rotation", note_f2)

        res_f3, _, _ = projective_fresnel_improper(s.cos(x**3), x, s.Integer(0), s.oo)
        self.assertEqual(0, s.simplify(res_f3 - s.gamma(s.Rational(1, 3)) / (2 * s.sqrt(3))))

        # 3. Full engine dispatch with step-by-step explanations and arbitrary precision (400 digits)
        tree = {
            "kind": "call",
            "value": "integrate",
            "args": [
                {"kind": "call", "value": "sin", "args": [{"kind": "binary", "value": "^", "args": [{"kind": "symbol", "value": "x"}, {"kind": "number", "value": "2"}]}]},
                {"kind": "symbol", "value": "x"},
                {"kind": "number", "value": "0"},
                {"kind": "symbol", "value": "oo"},
            ],
        }
        out = json.loads(
            calc_engine.dispatch(
                json.dumps(
                    {
                        "tree": tree,
                        "precision": 400,
                        "displayDigits": 400,
                        "removeComputationLimit": True,
                        "solutionSteps": True,
                    }
                )
            )
        )
        self.assertTrue(out["ok"], out)
        self.assertGreaterEqual(len(out["decimal"]), 395)
        step_titles = [st["title"] for st in out["solutionSteps"]["steps"]]
        self.assertTrue(any("Projective sector rotation" in t for t in step_titles), step_titles)
        self.assertTrue(str(projective_pi(60)).startswith("3.1415926535897932384626433832795028841971693993751058209"))


if __name__=="__main__":
    unittest.main()
