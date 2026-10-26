"""Math expressions are automatic; advanced Python follows the existing custom-code setting."""
import ast
import os

from ..protocol import EngineError, UNSUPPORTED


def validate_command(command):
    if os.environ.get('SDFX_ALLOW_CODE') == '1':
        return
    try:
        tree = ast.parse(command.strip(), mode='eval')
    except SyntaxError as exc:
        raise EngineError(UNSUPPORTED, 'Composable LoRA advanced commands require Settings > Allow custom code') from exc
    nodes = (ast.Expression, ast.BinOp, ast.UnaryOp, ast.BoolOp, ast.Compare, ast.IfExp,
             ast.Name, ast.Load, ast.Constant, ast.Call, ast.Add, ast.Sub, ast.Mult,
             ast.Div, ast.FloorDiv, ast.Mod, ast.Pow, ast.USub, ast.UAdd, ast.Not,
             ast.And, ast.Or, ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE)
    # Only named math helpers can be called on the automatic path. Native object
    # access, statements and user functions belong to the explicit Python setting.
    helpers = {'abs', 'ceil', 'floor', 'trunc', 'fmod', 'gcd', 'lcm', 'perm', 'comb', 'gamma',
               'sqrt', 'cbrt', 'exp', 'pow', 'log', 'ln', 'log2', 'log10', 'clamp',
               'asin', 'acos', 'atan', 'sin', 'cos', 'tan', 'sinr', 'cosr', 'tanr',
               'asinr', 'acosr', 'atanr', 'sinh', 'cosh', 'tanh', 'asinh', 'acosh',
               'atanh', 'abssin', 'abscos', 'random', 'warmup', 'cooldown'}
    for node in ast.walk(tree):
        if not isinstance(node, nodes) or (isinstance(node, ast.Name) and node.id.startswith('_')):
            raise EngineError(UNSUPPORTED, 'Composable LoRA native object/advanced commands require Settings > Allow custom code')
        if isinstance(node, ast.Call) and (not isinstance(node.func, ast.Name) or node.func.id not in helpers):
            raise EngineError(UNSUPPORTED, 'Composable LoRA advanced calls require Settings > Allow custom code')
