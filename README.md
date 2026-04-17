# HeurKit

A lightweight Python framework for heuristic optimization.

**One shared runtime, three usage levels:**
1. built-in kernels (TSP, CVRP, Bin Packing)
2. low-code custom problems (callbacks only)
3. full custom kernels (advanced extension)

---

## Why HeurKit?

Most heuristic codebases mix algorithm logic with domain logic in one script.
HeurKit separates these concerns:

| Layer | Responsibility | Domain-aware? |
|---|---|:---:|
| Algorithms | Search logic (HC, Tabu, SA, ILS, VNS) | No |
| Kernels / Custom | Problem-specific representation and moves | Yes |
| Core | Runtime contracts, stopping, results | No |

The same algorithm implementation runs unchanged across built-in kernels and callback-defined custom problems.

---

## Architecture

```text
┌──────────────────────────────────────────────────────────────┐
│ User API                                                     │
│ AutoSolver · Benchmark Runner · ProblemBuilder              │
├──────────────────────────────────────────────────────────────┤
│ Generic Algorithms                                           │
│ Greedy · Hill Climbing · Tabu · SA · ILS · VNS              │
├──────────────────────────┬───────────────────────────────────┤
│ Built-in Kernels         │ Low-code CustomProblem           │
│ TSP · CVRP · BinPacking  │ constructor/objective/moves      │
│                          │ (+optional feasibility/repair)   │
├──────────────────────────┴───────────────────────────────────┤
│ Core Abstractions                                            │
│ Problem · Solution · Move · Evaluator · SearchResult         │
└──────────────────────────────────────────────────────────────┘
```

---

## Supported Built-in Problems

| Problem | Objective |
|---|---|
| TSP | Minimize tour distance |
| CVRP | Minimize route distance |
| Bin Packing | Minimize number of bins |

## Supported Algorithms

- `GreedyConstructor`
- `HillClimbing`
- `TabuSearch`
- `SimulatedAnnealing`
- `IteratedLocalSearch`
- `VariableNeighborhoodSearch`

---

## Installation

```bash
git clone https://github.com/kongwoang/heurkit2.git
cd heurkit2
pip install -e .
```

Requirements: Python 3.11+, `numpy`.
Optional: `pytest`, `matplotlib`.

---

## Quickstart (Built-in Kernel)

```python
from heurkit.kernels.tsp.problem import TSPProblem
from heurkit.algorithms.simulated_annealing import SimulatedAnnealing

problem = TSPProblem.generate_random(n_cities=30, seed=42)
result = SimulatedAnnealing(time_limit=2.0, seed=42).solve(problem)
print(result.best_objective)
```

---

## Using Your Own Problem

### Which path should you choose?

- Use a **built-in kernel** when your problem is TSP/CVRP/Bin Packing.
- Use **CustomProblem (low-code)** when your domain is custom but you can define a state, objective, and moves.
- Build a **full custom kernel** when you need advanced control (specialized move objects, incremental delta evaluation, custom neighborhoods, etc.).

### Minimal callback contract

Required:
- `constructor(problem_data, rng) -> state`
- `objective(problem_data, state) -> float | Score`
- `generate_moves(problem_data, state, rng) -> iterable[move]`
- `apply_move(problem_data, state, move) -> new_state`

Optional:
- `feasibility(problem_data, state) -> bool`
- `repair(problem_data, state, rng) -> state`
- `pretty_print(problem_data, state) -> str`
- `state_copy(state) -> state`

### Complete example

```python
from dataclasses import dataclass
from heurkit.custom import ProblemBuilder, Score
from heurkit.portfolio.auto import AutoSolver

@dataclass
class MyState:
    assignment: list[int]

def make_initial(problem_data, rng):
    return MyState(assignment=[0 for _ in problem_data["items"]])

def evaluate(problem_data, state):
    loads = [0.0] * problem_data["num_bins"]
    for i, b in enumerate(state.assignment):
        loads[b] += problem_data["items"][i]

    hard_violations = sum(1 for load in loads if load > problem_data["capacity"])
    imbalance = max(loads) - min(loads)

    return Score(
        objective=max(loads),
        hard_violations=hard_violations,
        soft_penalty=imbalance,
    )

def is_feasible(problem_data, state):
    loads = [0.0] * problem_data["num_bins"]
    for i, b in enumerate(state.assignment):
        loads[b] += problem_data["items"][i]
    return all(load <= problem_data["capacity"] for load in loads)

def generate_moves(problem_data, state, rng):
    for i in range(len(state.assignment)):
        for b in range(problem_data["num_bins"]):
            if b != state.assignment[i]:
                yield ("assign", i, b)

def apply_move(problem_data, state, move):
    _, i, b = move
    out = state.assignment.copy()
    out[i] = b
    return MyState(assignment=out)

problem = (
    ProblemBuilder("my_custom_problem")
    .data({
        "items": [4, 7, 2, 9],
        "num_bins": 3,
        "capacity": 10,
    })
    .constructor(make_initial)
    .objective(evaluate, sense="min", hard_violation_weight=1_000.0)
    .feasibility(is_feasible)
    .moves(generate_moves, apply_move)
    .build()
)

result = AutoSolver(time_limit=5, seed=42).solve(problem)
print(result.best_objective)
```

### `Score` model

`objective` can return either:
- a `float`
- a `Score(objective, hard_violations=0, soft_penalty=0)`

`hard_violations` are multiplied by `hard_violation_weight` from `.objective(...)`.
This allows quick constraint handling without manually hard-coding giant penalties in your callback.

### How AutoSolver works for custom problems

`AutoSolver` now detects `CustomProblem` automatically.

Default strategy:
- no feasibility/repair: `HillClimbing` + `TabuSearch`
- with feasibility or repair: `TabuSearch` + `SimulatedAnnealing`

You can still override picks:

```python
result = AutoSolver(
    time_limit=5,
    seed=42,
    picks=["TabuSearch", "VNS"],
).solve(problem)
```

---

## Examples

Run built-in examples:

```bash
python examples/tsp_demo.py
python examples/cvrp_demo.py
python examples/binpacking_demo.py
python examples/auto_demo.py
```

Run custom examples:

```bash
python examples/custom_problem_minimal_demo.py
python examples/custom_assignment_demo.py
python examples/custom_partition_demo.py
```

---

## Tests

```bash
pytest -v
```

---

## Extending to a Full Kernel

When callbacks are no longer enough, implement a full kernel under `heurkit/kernels/<your_problem>/` with:

- `problem.py`
- `solution.py`
- `evaluator.py`
- `moves.py`
- `constructors.py`
- `neighbors.py`

Then register presets in `heurkit/portfolio/presets.py`.

---

## Current Limitations (Low-code API)

- Full re-evaluation per move (no delta evaluation hooks yet).
- Performance depends heavily on your callback efficiency.
- Move generators should stay bounded; generating huge neighborhoods each iteration can be slow.
- Callback layer does not auto-infer neighborhoods from formulas.
- No external solver adapters (Pyomo/Gurobi/OR-Tools) by design.

---

## License

MIT
