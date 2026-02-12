# parsing_utils.py


def parse_box(s: str):
    """Parse box: xmin,xmax,ymin,ymax (fractions of L, 0-1)."""
    vals = [float(x) for x in s.split(",")]
    if len(vals) != 4:
        raise ValueError(f"Box requires 4 values: xmin,xmax,ymin,ymax (got: {s})")
    return tuple(vals)

def parse_boundary_segment(s: str):
    """Parse boundary segment: side,tmin,tmax"""
    parts = s.split(",")
    if len(parts) != 3:
        raise ValueError(f"Boundary segment requires: side,tmin,tmax (got: {s})")
    side = parts[0].strip()
    tmin = float(parts[1])
    tmax = float(parts[2])
    if side not in ["x0", "xL", "y0", "yL"]:
        raise ValueError(f"side must be one of x0/xL/y0/yL (got: {side})")
    return (side, tmin, tmax)

def parse_robin_segment(s: str):
    """Parse Robin boundary segment: side,tmin,tmax,a"""
    parts = s.split(",")
    if len(parts) != 4:
        raise ValueError(f"Robin segment requires: side,tmin,tmax,a (got: {s})")
    side = parts[0].strip()
    tmin = float(parts[1])
    tmax = float(parts[2])
    a = float(parts[3])
    if side not in ["x0", "xL", "y0", "yL"]:
        raise ValueError(f"side must be one of x0/xL/y0/yL (got: {side})")
    return (side, tmin, tmax, a)

def parse_dirichlet_bc(s: str):
    """Parse Dirichlet BC segment: side,tmin,tmax,value"""
    parts = s.split(",")
    if len(parts) != 4:
        raise ValueError(f"Dirichlet BC requires: side,tmin,tmax,value (got: {s})")
    side = parts[0].strip()
    tmin = float(parts[1])
    tmax = float(parts[2])
    value = float(parts[3])
    if side not in ["x0", "xL", "y0", "yL"]:
        raise ValueError(f"side must be one of x0/xL/y0/yL (got: {side})")
    return (side, tmin, tmax, value)

def parse_dirichlet_disturbance(s: str):
    """Parse Dirichlet disturbance segment: side,tmin,tmax,func_type,param"""
    parts = s.split(",")
    if len(parts) != 5:
        raise ValueError(f"Dirichlet disturbance requires: side,tmin,tmax,func_type,param (got: {s})")
    side = parts[0].strip()
    tmin = float(parts[1])
    tmax = float(parts[2])
    func_type = parts[3].strip()
    param = float(parts[4])
    if side not in ["x0", "xL", "y0", "yL"]:
        raise ValueError(f"side must be one of x0/xL/y0/yL (got: {side})")
    if func_type not in ["tanh", "sin", "cos", "const"]:
        raise ValueError(f"func_type must be one of tanh/sin/cos/const (got: {func_type})")
    return (side, tmin, tmax, func_type, param)
