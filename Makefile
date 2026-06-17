.PHONY: check compile pilot robust ac ieee33 matpower ieee123 public reserve acn

check:
	python -m py_compile simulations/*.py tools/*.py
	python tools/check_submission_readiness.py

compile:
	python -m py_compile simulations/*.py tools/*.py

pilot:
	python simulations/pilot_grid_fedgat.py

robust:
	python simulations/robust_grid_fedgat.py

ac:
	python simulations/ac_validation.py

ieee33:
	python simulations/ieee33_validation.py

matpower:
	python simulations/matpower_validation.py

ieee123:
	python simulations/ieee123_validation.py

public:
	python simulations/public_ev_validation.py

reserve:
	python simulations/reserve_pareto_sensitivity.py

acn:
	python simulations/acn_data_validation.py
