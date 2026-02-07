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
