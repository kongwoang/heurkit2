# HeurKit

A lightweight Python framework for heuristic optimization of combinatorial problems.

**One shared runtime. Multiple problem kernels. Consistent API.**

---

## Why HeurKit?

Most heuristic optimization code is written as **monolithic, single-problem scripts** — the algorithm logic is tangled with the domain representation, making it hard to reuse, benchmark, or extend.

HeurKit separates these concerns:

| Layer | Responsibility | Domain-aware? |
|-------|---------------|:---:|
| **Algorithms** | Search logic (SA, Tabu, VNS, …) | ❌ No |
| **Kernels** | Solution structure, moves, evaluation | ✅ Yes |
| **Core** | Interfaces, stopping criteria, results | ❌ No |

The same Simulated Annealing implementation runs **unchanged** on TSP, CVRP, and Bin Packing — because it only interacts with abstract `Solution`, `Move`, and `Evaluator` interfaces. The domain-specific logic lives entirely in swappable **kernels**.

This is not a production solver. It's a **clean framework prototype** for learning, presenting, and extending.

---

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                  User API                            │
│   AutoSolver · Benchmark Runner · Examples           │
├─────────────────────────────────────────────────────┤
│            Generic Search Algorithms                 │
│   Hill Climbing · Tabu · SA · ILS · VNS · Greedy    │
│   (domain-agnostic — uses abstract interfaces)      │
├──────────┬──────────┬───────────────────────────────┤
│  TSP     │  CVRP    │  Bin Packing                  │
│  Kernel  │  Kernel  │  Kernel                       │
│          │          │                               │
│  Solution│  Solution│  Solution                     │
│  Moves   │  Moves   │  Moves                       │
│  Eval    │  Eval    │  Eval                         │
│  Constr  │  Constr  │  Constr                       │
├──────────┴──────────┴───────────────────────────────┤
│              Core Abstractions                       │
│   Problem · Solution · Move · Evaluator             │
│   StoppingCriteria · SearchResult · Callbacks       │
└─────────────────────────────────────────────────────┘
```

---

## Supported Problems

| Problem | Description | Objective |
|---------|-------------|-----------|
| **TSP** | Symmetric Travelling Salesman | Minimise tour distance |
| **CVRP** | Capacitated Vehicle Routing | Minimise total route distance |
| **Bin Packing** | 1-D Bin Packing | Minimise number of bins |

## Supported Algorithms

| Algorithm | Type | Key Idea |
|-----------|------|----------|
| `GreedyConstructor` | Construction | One-shot baseline |
| `HillClimbing` | Local Search | First-improvement descent |
| `TabuSearch` | Local Search | Short-term memory avoids cycling |
| `SimulatedAnnealing` | Metaheuristic | Probabilistic acceptance with cooling |
| `IteratedLocalSearch` | Metaheuristic | Perturbation + local search |
| `VariableNeighborhoodSearch` | Metaheuristic | Systematic neighbourhood switching |

All algorithms share the same interface and accept: `time_limit`, `max_iterations`, `seed`, `callbacks`.

---

## Installation

```bash
git clone https://github.com/kongwoang/heurkit2.git
cd heurkit2
pip install -e .          # editable install
# or just:
pip install -r requirements.txt
```

**Requirements:** Python 3.11+, numpy. Optional: matplotlib (plots), pytest (tests).

---

## Quickstart

### Solve a TSP in 3 lines

```python
from heurkit.kernels.tsp.problem import TSPProblem
from heurkit.algorithms.simulated_annealing import SimulatedAnnealing

problem = TSPProblem.generate_random(n_cities=30, seed=42)
result = SimulatedAnnealing(time_limit=2.0, seed=42).solve(problem)
print(result)  # objective, feasibility, iterations, runtime
```

### Solve CVRP

```python
from heurkit.kernels.cvrp.problem import CVRPProblem
from heurkit.algorithms.tabu import TabuSearch

problem = CVRPProblem.generate_random(n_customers=20, capacity=50.0, seed=42)
result = TabuSearch(time_limit=2.0, seed=42).solve(problem)
print(f"Distance: {result.best_objective:.2f}, Feasible: {result.is_feasible}")
```

### Solve Bin Packing

```python
from heurkit.kernels.binpacking.problem import BinPackingProblem
from heurkit.algorithms.vns import VariableNeighborhoodSearch

problem = BinPackingProblem.generate_random(n_items=40, capacity=100.0, seed=42)
result = VariableNeighborhoodSearch(time_limit=2.0, seed=42).solve(problem)
print(f"Bins: {result.best_objective:.0f}")
```

### AutoSolver — automatic algorithm selection

```python
from heurkit.portfolio.auto import AutoSolver

# Works on any supported problem type — auto-detects from class
result = AutoSolver(time_limit=3.0, seed=42).solve(problem)
```

---

## Benchmarking

HeurKit includes a benchmark runner for reproducible algorithm comparisons:

```python
from heurkit.benchmark.runner import BenchmarkConfig, run_benchmark
from heurkit.kernels.tsp.problem import TSPProblem
from heurkit.algorithms import HillClimbing, SimulatedAnnealing, TabuSearch

config = BenchmarkConfig(
    name="tsp_comparison",
    problems=[TSPProblem.generate_random(n_cities=30, seed=i) for i in range(3)],
    algorithms=[
        HillClimbing(time_limit=2.0, seed=42),
        SimulatedAnnealing(time_limit=2.0, seed=42),
        TabuSearch(time_limit=2.0, seed=42),
    ],
    output_dir="output",
)

bench = run_benchmark(config)
print(bench.summary_table())
# Results auto-saved to output/tsp_comparison.csv and .json
```

### Run the built-in benchmark

```bash
python examples/compare_algorithms.py
```

---

## AutoSolver — Presets & Customisation

```python
from heurkit.portfolio.auto import AutoSolver

# List all available presets
print(AutoSolver.available_presets())
# {'tsp': ['HillClimbing', 'SimulatedAnnealing', ...], ...}

# Use default presets
result = AutoSolver(time_limit=5.0, seed=42).solve(problem)

# Override with custom picks
result = AutoSolver(
    time_limit=5.0, seed=42,
    picks=["TabuSearch", "VNS"],
).solve(problem)

# Get all trial results
result = AutoSolver(time_limit=5.0, seed=42, return_all=True).solve(problem)
print(result.metadata["all_results"])
```

---

## Callbacks & Observability

```python
from heurkit.algorithms.hill_climb import HillClimbing
from heurkit.core.callbacks import PrintCallback, LoggingCallback, HistoryCallback

# Print progress every 100 iterations
result = HillClimbing(time_limit=2.0).solve(problem, callbacks=[PrintCallback(100)])

# Use Python logging
result = HillClimbing(time_limit=2.0).solve(problem, callbacks=[LoggingCallback(500)])

# Record improvement events programmatically
cb = HistoryCallback()
result = HillClimbing(time_limit=2.0).solve(problem, callbacks=[cb])
print(cb.events)  # [{iteration: 5, objective: 42.3, ...}, ...]
```

---

## Repository Structure

```
heurkit/
├── core/                    # Domain-agnostic abstractions
│   ├── problem.py           #   Abstract Problem
│   ├── solution.py          #   Abstract Solution
│   ├── move.py              #   Abstract Move
│   ├── evaluator.py         #   Evaluator + Evaluation
│   ├── runtime.py           #   SearchAlgorithm base + protocols
│   ├── result.py            #   SearchResult with serialisation
│   ├── stopping.py          #   Composable stopping criteria
│   ├── callbacks.py         #   Print / Logging / History callbacks
│   └── random_state.py      #   Deterministic RNG factory
│
├── algorithms/              # Generic search algorithms
│   ├── greedy.py            #   Constructor wrapper
│   ├── hill_climb.py        #   First-improvement hill climbing
│   ├── tabu.py              #   Tabu search
│   ├── simulated_annealing.py
│   ├── iterated_local_search.py
│   └── vns.py               #   Variable Neighbourhood Search
│
├── kernels/                 # Domain-specific problem kernels
│   ├── tsp/                 #   Travelling Salesman Problem
│   ├── cvrp/                #   Capacitated Vehicle Routing
│   └── binpacking/          #   1-D Bin Packing
│
├── portfolio/               # Auto-solver layer
│   ├── auto.py              #   AutoSolver
│   └── presets.py            #   Algorithm presets per problem type
│
├── benchmark/               # Benchmark runner
│   └── runner.py            #   BenchmarkConfig → BenchmarkResult
│
└── utils/                   # Utilities
    ├── metrics.py           #   ASCII table formatting
    ├── plotting.py          #   Convergence plots (matplotlib)
    ├── distance.py          #   Distance helpers
    └── io.py                #   Instance generators

examples/                    # Runnable demos
tests/                       # pytest test suite (90+ tests)
```

---

## How to Extend

### Adding a New Kernel (Problem Type)

1. Create `heurkit/kernels/your_problem/`
2. Implement these files following the existing pattern:

| File | Implements | Purpose |
|------|-----------|---------|
| `problem.py` | `Problem` ABC | Instance data + factory methods + `default_*()` |
| `solution.py` | `Solution` ABC | Solution representation + `copy()` |
| `evaluator.py` | `Evaluator` ABC | Objective + feasibility computation |
| `moves.py` | `Move` ABC | Concrete move operators |
| `constructors.py` | `Constructor` protocol | Initial solution builders |
| `neighbors.py` | `NeighborhoodGenerator` protocol | Move sampling logic |

3. Add the problem type to `portfolio/presets.py`
4. All existing algorithms will work automatically

### Adding a New Algorithm

1. Create `heurkit/algorithms/your_algo.py`
2. Subclass `SearchAlgorithm` and implement `solve()`
3. Use `self._resolve_components()` to get constructor/evaluator/neighbourhood
4. Use `StoppingCriteria` for consistent stopping
5. Return a `SearchResult`
6. Call `self._fire_new_best()` and `self._fire_iteration()` for callback support

---

## Running

```bash
# Run all tests
make test
# or: pytest tests/ -v

# Run individual demos
make examples

# Run benchmark
make benchmark

# Clean up
make clean
```

---

## Result Serialisation

```python
result = algo.solve(problem)

# To dict (excludes solution object)
d = result.to_dict()

# To JSON string
j = result.to_json()
```

---

## Input Validation

All problem constructors validate inputs and raise `ValueError` with clear messages:

```python
TSPProblem(np.zeros((2, 2)))      # ValueError: TSP requires at least 3 cities
CVRPProblem(..., capacity=0)       # ValueError: Vehicle capacity must be positive
BinPackingProblem(np.array([200]), bin_capacity=100)  # ValueError: exceeds bin capacity
```

---

## Current Limitations

- **Not production-optimized.** Full re-evaluation per move (no delta evaluation).
- **Small instances.** Designed for tens to low hundreds of entities.
- **Simple algorithm implementations.** Correct and readable, not state-of-the-art.
- **No parallel execution.** All algorithms run sequentially.
- **No standard format I/O.** No TSPLIB/CVRPLIB parsers (only random generators).

---

## Roadmap

- [ ] Delta evaluation for faster inner loops
- [ ] Large Neighbourhood Search (LNS) with pluggable destroy/repair
- [ ] Standard instance format parsers (TSPLIB, CVRPLIB, OR-Library)
- [ ] Additional kernels: Job Shop Scheduling, Knapsack
- [ ] Parallel portfolio execution
- [ ] CLI with `typer` for running benchmarks from the command line
- [ ] Sphinx API documentation
- [ ] GitHub Actions CI

---

## License

MIT
