
Time-dependent optimal control for the heat equation with **FEniCSx**:
- Dirichlet boundary control
- Neumann boundary control (flux)
- Distributed control (source term)
- Adjoint-based gradient
- Path state constraints via Moreau–Yosida / SC loop

## Requirements
- Python 3.10+
- dolfinx (FEniCSx)
- mpi4py
- petsc4py
- numpy
- ruff (optional, for linting)

## Quick run

Example (Dirichlet control on y=L, default target/constraint box):
```bash
python3 main.py \
  --n 5 --dt 20 --T-final 600 --L 0.1 \
  --k 0.15 --rho 1100 --c 1500 \
  --T-ambient 25 --T-cure 160 \
  --alpha-track 1.0 --alpha-u 1e-4 --gamma-u 1e-2 \
  --beta 1e3 --sc-maxit 1 --inner-maxit 1 \
  --lr 5.0 --grad-tol 1e-3 \
  --output-dir resu --output-freq 1 --no-vtk
