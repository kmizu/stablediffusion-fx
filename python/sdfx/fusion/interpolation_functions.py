# Adapted from ljleb/prompt-fusion-extension, MIT, commit 1d86996ac4c3d32efaacb8a115600aa6fde6dab6.
import copy
import math
from sdfx.fusion import interpolation_tensor, geometries


def compute_linear(control_points, params: interpolation_tensor.InterpolationParams):
    if len(control_points) <= 2:
        return geometries.slerp_geometry(control_points, params)
    else:
        target_curve = min(int(params.t * (len(control_points) - 1)), len(control_points) - 1)
        cp0 = control_points[target_curve]
        cp1 = control_points[target_curve + 1] if target_curve + 1 < len(control_points) else control_points[-1]
        new_params = interpolation_tensor.InterpolationParams(math.fmod(params.t * (len(control_points) - 1), 1.), *params[1:])
        return geometries.slerp_geometry([cp0, cp1], new_params)


def compute_bezier(control_points, params: interpolation_tensor.InterpolationParams):
    def compute_casteljau(ps, size):
        for i in reversed(range(1, size)):
            for j in range(i):
                ps[j] = geometries.slerp_geometry([ps[j], ps[j+1]], params)

        return ps[0]

    if len(control_points) == 1:
        return control_points[0]
    elif len(control_points) == 2:
        return geometries.slerp_geometry(control_points, params)
    copied_control_points = copy.deepcopy(control_points)
    return compute_casteljau(copied_control_points, len(copied_control_points))


def compute_catmull(control_points, params: interpolation_tensor.InterpolationParams):
    if len(control_points) <= 2:
        return compute_linear(control_points, params)
    else:
        target_curve = min(int(params.t * (len(control_points) - 1)), len(control_points) - 1)
        g0 = control_points[target_curve - 1] if target_curve > 0 else control_points[0] * 2 - control_points[1]
        cp0 = control_points[target_curve]
        cp1 = control_points[target_curve + 1] if target_curve + 1 < len(control_points) else control_points[-1]
        g1 = control_points[target_curve + 2] if target_curve + 2 < len(control_points) else cp1 * 2 - cp0
        ip0 = cp0 + (cp1 - g0)/6
        ip1 = cp1 + (cp0 - g1)/6

        new_params = interpolation_tensor.InterpolationParams(math.fmod(params.t * (len(control_points) - 1), 1.), *params[1:])
        return compute_bezier([cp0, ip0, ip1, cp1], new_params)
