import argparse

def build_parser():
    parser = argparse.ArgumentParser(
        description="Time-Dependent Optimal Control Of Heat Equation",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    # Domain
    parser.add_argument("--L", type=float, default=0.1)
    parser.add_argument("--n", type=int, default=3)

    # Time
    parser.add_argument("--dt", type=float, default=20.0)
    parser.add_argument("--T-final", type=float, default=600.0)

    # Physical
    parser.add_argument("--k", type=float, default=0.15)
    parser.add_argument("--rho", type=float, default=1100.0)
    parser.add_argument("--c", type=float, default=1500.0)
    parser.add_argument("--T-ambient", type=float, default=25.0)
    parser.add_argument("--T-cure", type=float, default=160.0)

    # Control zones
    parser.add_argument("--control-boundary-dirichlet", action="append", default=[])
    parser.add_argument("--control-boundary-neumann", action="append", default=[])
    parser.add_argument("--control-distributed", action="append", default=[])
    parser.add_argument("--no-default-controls", action="store_true")

    # Target zones
    parser.add_argument("--target-zone", action="append", default=[])

    # Optimization
    parser.add_argument("--alpha-track", type=float, default=1.0)
    parser.add_argument("--alpha-u", type=float, default=1e-4)
    parser.add_argument("--gamma-u", type=float, default=1e-2)
    parser.add_argument("--beta", type=float, default=1e3)
    parser.add_argument("--sc-maxit", type=int, default=5)
    parser.add_argument("--sc-tol", type=float, default=1e-6)
    parser.add_argument("--u-init", type=float, default=180.0)
    parser.add_argument("--u-min", type=float, default=25.0)
    parser.add_argument("--u-max", type=float, default=250.0)
    parser.add_argument("--check-grad", action="store_true")
    parser.add_argument("--beta-u", type=float, default=0.0)
    parser.add_argument("--dirichlet-spatial-reg", type=str, default="L2", choices=["L2", "H1"])

    # State constraint selection
    parser.add_argument("--sc-type", type=str, default="lower", choices=["lower", "upper", "box"])
    parser.add_argument("--sc-lower", type=float, default=None)
    parser.add_argument("--sc-upper", type=float, default=None)
    parser.add_argument("--sc-start-time", type=float, default=None)
    parser.add_argument("--sc-end-time", type=float, default=None)
    parser.add_argument("--constraint-zone", action="append", default=[])

    # Gradient descent
    parser.add_argument("--lr", type=float, default=5.0)
    parser.add_argument("--inner-maxit", type=int, default=30)
    parser.add_argument("--grad-tol", type=float, default=1e-3)
    parser.add_argument("--fd-eps", type=float, default=1e-2)

    # Output/visualization
    parser.add_argument("--output-freq", type=int, default=1)
    parser.add_argument("--output-dir", type=str, default="resu")
    parser.add_argument("--no-vtk", action="store_true")

    return parser

