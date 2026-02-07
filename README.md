# PyHeatControl

Time-dependent **PDE-constrained optimal control** for the heat equation using **FEniCSx / dolfinx**.

The library implements a fully discrete **discretize-then-optimize adjoint workflow** consistent with the accompanying paper.

Main features:

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

Serial execution works automatically without `mpirun`.

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
```

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
