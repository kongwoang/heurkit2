.PHONY: install test examples benchmark lint clean

install:
	pip install -e .

test:
	python -m pytest tests/ -v

examples:
	python examples/tsp_demo.py
	python examples/cvrp_demo.py
	python examples/binpacking_demo.py
	python examples/auto_demo.py
	python examples/custom_problem_minimal_demo.py
	python examples/custom_assignment_demo.py
	python examples/custom_partition_demo.py

benchmark:
	python examples/compare_algorithms.py

lint:
	python -m py_compile heurkit/__init__.py
	python -c "import heurkit; print(f'heurkit v{heurkit.__version__} OK')"

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type d -name .pytest_cache -exec rm -rf {} +
	rm -rf output/ *.png build/ dist/ *.egg-info
