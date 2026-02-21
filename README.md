# PyHeatControl

Time-dependent **PDE-constrained optimal control** for the heat equation using **FEniCSx / dolfinx**.

The library implements a fully discrete **discretize-then-optimize adjoint workflow** consistent with the accompanying paper.

## Main features

- implicit Euler time stepping
- discrete adjoint (backward solve)
- adjoint-based gradients
- heterogeneous actuators:
  - Dirichlet boundary temperature
  - Neumann boundary heat flux
  - distributed volumetric heat sources
- L² + H¹-in-time control regularization
- time-window **state constraints** enforced via Moreau–Yosida / augmented Lagrangian
- segregated optimization loop: forward → adjoint → gradient → projected update
- native **MPI parallel execution** (PETSc + dolfinx)

---

## Requirements

⚠️ **dolfinx must be installed from conda-forge (pip installation is not supported).**

Recommended setup:

```bash
conda create -n fenicsx -c conda-forge python=3.11 fenics-dolfinx mpi4py petsc4py
conda activate fenicsx
pip install -e .
```

---

## Quick start (serial)

```bash
pyheatcontrol --n 5 --dt 20 --T-final 600
```

---

## Running

### Installed CLI (recommended)

```bash
pyheatcontrol --help
```

Example:

```bash
pyheatcontrol --n 5 --dt 20 --T-final 600 --alpha-track 1.0 --alpha-u 1e-4 --gamma-u 1e-2
```

### Development mode (without install)

```bash
python scripts/main.py ...
```

---

## Parallel execution (MPI)

Run on multiple processes:

```bash
mpirun -n 4 pyheatcontrol --n 5 --dt 20 --T-final 600
```

**Known issue**: Results may differ slightly (~0.02%) between serial (np=1) and parallel (np>1) execution due to floating-point reduction order. This is a minor numerical precision issue and does not affect solution quality.

---

## Parameter guidance

### Regularization parameters

The cost functional is:
```
J = α_track * J_track + α_u * J_reg_L2 + γ_u * J_reg_H1_time + β_u * J_reg_H1_space
```

| Parameter | Typical range | Effect |
|-----------|--------------|--------|
| `alpha-track` | 1.0 | Weight of tracking term |
| `alpha-u` | 1e-5 to 1e-2 | L2 regularization - higher = smoother controls |
| `gamma-u` | 1e-3 to 1e-1 | H1 temporal regularization - higher = smoother in time |
| `beta-u` | 0 to 1e-8 | H1 spatial regularization - use with **dirichlet-spatial-reg H1**. ⚠️ Must be ~4 orders of magnitude smaller than `alpha-u` to avoid numerical overflow |

### State constraints

| Parameter | Typical value | Effect |
|-----------|--------------|--------|
| `sc-type` | lower/upper/box | Constraint type |
| `sc-lower` | 25.0 | Lower temperature bound (°C) |
| `sc-upper` | 200.0 | Upper temperature bound (°C) |
| `beta` | 1e3 | Penalty parameter for augmented Lagrangian |
| `sc-maxit` | 2-5 | Number of SC iterations |

### Optimization

| Parameter | Typical value | Effect |
|-----------|--------------|--------|
| `lr` | 1.0 to 10.0 | Learning rate / step size |
| `inner-maxit` | 4-30 | Maximum gradient descent iterations |
| `grad-tol` | 1e-3 | Gradient norm convergence tolerance |

---

## Project structure

```
src/pyheatcontrol/
  solver/
    forward.py
    adjoint.py
    gradient.py
  optimization.py
  cli.py

scripts/
  main.py
  test_suite.py
  profile_bottlenecks.py
```

---

## Testing

Run the test suite:

```bash
python scripts/test_suite.py
```

Run specific tests:

```bash
python scripts/test_suite.py --test dirichlet_control
python scripts/test_suite.py --test neumann_control
python scripts/test_suite.py --test distributed_control
```

Run profiling:

```bash
python scripts/profile_bottlenecks.py
```

---

## Performance notes

Typical runtimes (n=3, dt=50, T=200):

| Case | Runtime |
|------|---------|
| Dirichlet control | ~1s |
| Neumann control | ~3s |
| Distributed control | ~1s |
| Mesh n=4 | ~5s |
| Mesh n=5 | ~20s |

Optimizations applied:
- Relaxed KSP tolerances for internal solves (1e-8 instead of 1e-12)
- Reduced Armijo line search iterations (max 5 instead of 20)
- Reused work vectors in gradient computation

---

## Known limitations

1. **Parallel results**: Slight numerical differences (~0.02%) between np=1 and np>1 due to MPI reduction order. This is acceptable for practical use.

2. **beta_u parameter**: When using `dirichlet-spatial-reg H1`, `beta_u` must be ~1e-8 or smaller to avoid numerical overflow. This is because H1 spatial regularization scales differently than L2.

---

## Code quality

```bash
ruff check . --fix
ruff format .
```

---

## CLI options

```bash
pyheatcontrol --help
```
