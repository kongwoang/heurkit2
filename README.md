# HeurKit

A heuristic optimization framework with a **shared metaheuristic runtime** and **domain-specific kernels** for combinatorial optimization problems.

## Design Goal

HeurKit demonstrates a clean architectural pattern:

> **Generic search algorithms + domain-specific problem kernels = one framework, many problems.**

The core runtime knows nothing about cities, routes, or bins. It operates entirely through abstract interfaces (`Problem`, `Solution`, `Move`, `Evaluator`). Each problem domain implements its own *kernel* — a self-contained module that defines the solution representation, move operators, evaluators, and constructors for that specific problem type.

This means the same Simulated Annealing or Tabu Search implementation runs unchanged across TSP, CVRP, and Bin Packing.

## Key Architecture

```
┌─────────────────────────────────────────────────────┐
│                    User API                          │
│   solver.solve(problem) → SearchResult              │
├─────────────────────────────────────────────────────┤
│              Portfolio / AutoSolver                   │
│   Picks algorithms + presets per problem type        │
├─────────────────────────────────────────────────────┤
│            Generic Search Algorithms                 │
│   Hill Climbing · Tabu · SA · ILS · Greedy          │
│   (domain-agnostic — uses abstract interfaces)      │
├──────────┬──────────┬───────────────────────────────┤
│ TSP      │ CVRP     │ Bin Packing                   │
│ Kernel   │ Kernel   │ Kernel                        │
│          │          │                               │
│ ·Solution│ ·Solution│ ·Solution                     │
│ ·Moves   │ ·Moves   │ ·Moves                       │
│ ·Eval    │ ·Eval    │ ·Eval                         │
│ ·Constr  │ ·Constr  │ ·Constr                       │
├──────────┴──────────┴───────────────────────────────┤
│              Core Abstractions                       │
│   Problem · Solution · Move · Evaluator             │
│   StoppingCriteria · SearchResult · RNG             │
└─────────────────────────────────────────────────────┘
```

## Supported Problems (MVP)

| Problem | Type | Objective |
|---------|------|-----------|
| **TSP** | Symmetric Travelling Salesman | Minimize tour distance |
| **CVRP** | Capacitated Vehicle Routing | Minimize total route distance |
| **Bin Packing** | 1-D Bin Packing | Minimize number of bins |

## Folder Structure

```
heurkit/
├── core/                    # Domain-agnostic abstractions
│   ├── problem.py           # Abstract Problem
│   ├── solution.py          # Abstract Solution
│   ├── move.py              # Abstract Move
│   ├── evaluator.py         # Evaluator + Evaluation
│   ├── runtime.py           # SearchAlgorithm base + protocols
│   ├── result.py            # SearchResult dataclass
│   ├── stopping.py          # Reusable stopping criteria
│   ├── callbacks.py         # Optional search callbacks
│   └── random_state.py      # Deterministic RNG
│
├── algorithms/              # Generic search algorithms
│   ├── greedy.py            # Constructor wrapper
│   ├── hill_climb.py        # First-improvement hill climbing
│   ├── tabu.py              # Tabu search
│   ├── simulated_annealing.py
│   └── iterated_local_search.py
│
├── kernels/                 # Domain-specific problem kernels
│   ├── tsp/                 # TSP kernel
│   │   ├── problem.py       #   Problem definition
│   │   ├── solution.py      #   Tour permutation
│   │   ├── evaluator.py     #   Tour distance evaluator
│   │   ├── moves.py         #   Swap, 2-opt, insert
│   │   ├── constructors.py  #   Random, nearest-neighbour
│   │   └── neighbors.py     #   Neighbourhood generator
│   ├── cvrp/                # CVRP kernel
│   │   ├── problem.py
│   │   ├── solution.py      #   List of routes
│   │   ├── evaluator.py     #   Distance + capacity penalty
│   │   ├── moves.py         #   Relocate, swap, intra-2opt
│   │   ├── constructors.py  #   Greedy sequential, random
│   │   └── neighbors.py
│   └── binpacking/          # Bin Packing kernel
│       ├── problem.py
│       ├── solution.py      #   List of bins
│       ├── evaluator.py     #   Bin count + overflow penalty
│       ├── moves.py         #   Move item, swap items
│       ├── constructors.py  #   First Fit, First Fit Decreasing
│       └── neighbors.py
│
├── portfolio/               # Auto-solver layer
│   ├── auto.py              # AutoSolver
│   └── presets.py           # Default algorithm configs
│
└── utils/                   # Utilities
    ├── distance.py          # Distance matrix helpers
    ├── metrics.py           # Result formatting
    ├── plotting.py          # Convergence plots
    └── io.py                # Instance generators

examples/                    # Runnable demos
tests/                       # pytest test suite
```

## Installation

```bash
# Clone the repository
git clone <repo-url>
cd heurkit

# Install dependencies
pip install -r requirements.txt

# Optional: install in development mode
pip install -e .
```

**Requirements:** Python 3.11+, numpy. Optional: matplotlib (plotting), pytest (tests).

## Quickstart

### TSP

```python
from heurkit.kernels.tsp.problem import TSPProblem
from heurkit.algorithms.simulated_annealing import SimulatedAnnealing

# Create a random 30-city TSP
problem = TSPProblem.generate_random(n_cities=30, seed=42)

# Solve with Simulated Annealing
solver = SimulatedAnnealing(max_seconds=2.0, seed=42)
result = solver.solve(problem)

print(result.best_objective)  # Total tour distance
print(result)                 # Full summary
```

### CVRP

```python
from heurkit.kernels.cvrp.problem import CVRPProblem
from heurkit.algorithms.tabu import TabuSearch

problem = CVRPProblem.generate_random(n_customers=20, capacity=50.0, seed=42)

solver = TabuSearch(max_seconds=2.0, seed=42)
result = solver.solve(problem)

print(f"Distance: {result.best_objective:.2f}")
print(f"Feasible: {result.is_feasible}")
```

### Bin Packing

```python
from heurkit.kernels.binpacking.problem import BinPackingProblem
from heurkit.algorithms.hill_climb import HillClimbing

problem = BinPackingProblem.generate_random(n_items=40, capacity=100.0, seed=42)

solver = HillClimbing(max_seconds=2.0, seed=42)
result = solver.solve(problem)

print(f"Bins used: {result.best_objective:.0f}")
```

### Custom Instance

```python
from heurkit.kernels.tsp.problem import TSPProblem

coords = [(0, 0), (10, 0), (10, 10), (0, 10), (5, 5)]
problem = TSPProblem.from_coordinates(coords, instance_name="my-tsp")
```

## Available Algorithms

| Algorithm | Description | Key Parameters |
|-----------|-------------|---------------|
| `GreedyConstructor` | One-shot construction (baseline) | — |
| `HillClimbing` | First-improvement local search | `max_seconds`, `no_improvement_limit` |
| `TabuSearch` | Short-term memory search | `tabu_tenure`, `max_seconds` |
| `SimulatedAnnealing` | Probabilistic acceptance with cooling | `initial_temp`, `cooling_rate` |
| `IteratedLocalSearch` | Local search + perturbation | `perturbation_strength`, `local_search_iters` |

All algorithms share the same interface:

```python
result = algorithm.solve(problem)
```

Every algorithm accepts: `max_seconds`, `max_iterations`, `seed`.
Every algorithm returns a `SearchResult` with: `best_objective`, `best_solution`, `is_feasible`, `iterations`, `runtime_seconds`, `history`.

## AutoSolver / Presets

The `AutoSolver` picks sensible algorithm presets per problem type and returns the best result:

```python
from heurkit.portfolio.auto import AutoSolver

# Automatically detects problem type from the class
result = AutoSolver(time_limit=3.0, seed=42).solve(problem)

# Or specify explicitly
result = AutoSolver(problem_type="tsp", time_limit=3.0).solve(problem)
```

**Default presets:**

| Problem | Algorithms |
|---------|-----------|
| TSP | Hill Climbing + Simulated Annealing |
| CVRP | Hill Climbing + Tabu Search |
| Bin Packing | Hill Climbing + Iterated Local Search |

## Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run tests for a specific domain
pytest tests/test_tsp.py -v
pytest tests/test_cvrp.py -v
pytest tests/test_binpacking.py -v

# Run algorithm cross-domain tests
pytest tests/test_algorithms.py -v
```

## Running Demo / Benchmark Scripts

```bash
# Individual problem demos
python examples/tsp_demo.py
python examples/cvrp_demo.py
python examples/binpacking_demo.py

# Cross-domain comparison (all problems × all algorithms)
python examples/compare_algorithms.py
```

The benchmark script prints summary tables and saves convergence plots (requires matplotlib).

## Stopping Criteria

Algorithms support composable stopping criteria:

- **`max_iterations`** — stop after N iterations
- **`max_seconds`** — stop after wall-clock time limit
- **`no_improvement_iterations`** — stop if no improvement for N iterations

The search stops when **any** criterion fires.

## Current Limitations

This is an MVP designed to demonstrate the architecture. Known limitations:

- **Not production-optimized.** Move evaluation is full re-evaluation (no incremental/delta updates).
- **Small instances only.** Designed for instances with tens to low hundreds of entities.
- **Simple algorithm implementations.** Correct and readable, not state-of-the-art.
- **No parallel execution.** All algorithms run sequentially.
- **CVRP is simplified.** No time windows, pickup-delivery, or heterogeneous fleets.
- **No TSPLIB / standard format parsing.** Only random generators and manual construction.

## Future Extensions

The architecture is designed to be extended along these axes:

- **New kernels:** Job Shop Scheduling, Knapsack, Graph Coloring, etc.
- **New algorithms:** Genetic Algorithm, GRASP, Variable Neighbourhood Search, etc.
- **Incremental evaluation:** Delta-evaluation for faster move assessment.
- **Parallel portfolio:** Run multiple algorithms concurrently and take the best.
- **Instance I/O:** TSPLIB, CVRPLIB, OR-Library format parsers.
- **ML-based algorithm selection:** Use instance features to pick the best algorithm.
- **Richer visualization:** Interactive tour/route/packing plots.

## License

MIT
