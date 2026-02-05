# parsing_utils.py


def parse_box(s: str):
    """Parse box: xmin,xmax,ymin,ymax (in frazioni di L, 0-1)"""
    vals = [float(x) for x in s.split(",")]
    if len(vals) != 4:
        raise ValueError(f"Box richiede 4 valori: xmin,xmax,ymin,ymax (dato: {s})")
    return tuple(vals)


def parse_boundary_segment(s: str):
    """Parse boundary segment: side,tmin,tmax"""
    parts = s.split(",")
    if len(parts) != 3:
        raise ValueError(f"Boundary segment: side,tmin,tmax (dato: {s})")
    side = parts[0].strip()
    tmin = float(parts[1])
    tmax = float(parts[2])
    if side not in ["x0", "xL", "y0", "yL"]:
        raise ValueError(f"side deve essere x0/xL/y0/yL (dato: {side})")
    return (side, tmin, tmax)
