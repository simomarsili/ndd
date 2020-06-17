INSTALL_PATH=/usr/local
PYTHON=python3
VERSION=$(shell python3 -c "import ndd; print(ndd.__version__)")
.PHONY: default build install dev_install test clean veryclean f2py timings dist ta

default:
	make dev_install; make test
build:
	$(PYTHON) setup.py build
install:
	make veryclean ; pip3 install .
dev_install:
	make veryclean ; pip3 install -e .
test:
	(cd tests; pytest)
clean:
	$(RM) ndd/exts/*.o ndd/exts/*.mod ndd/fnsb*
veryclean:
	make clean
	$(RM) -- **/*~
	$(RM) -- **/*#
	$(RM) -r .tox .cache .pytest_cache .libs __pycache__ ndd.egg-info dist build ndd/__pycache__
	pip3 uninstall ndd
f2py:
	(cd ndd/exts; $(RM) nsb.pyf; f2py estimators.f90 -m fnsb -h nsb.pyf; mv nsb.pyf ../)
timings:
	$(PYTHON) utils/timings.py
dist:
	@if [ "$(shell git rev-parse --abbrev-ref HEAD)" != "master" ]; then (echo "run `make dist` only in master branch"; exit 1); fi
	$(RM) -f dist/*
	$(PYTHON) setup.py sdist
tag:
	@if [ "$(shell git rev-parse --abbrev-ref HEAD)" != "master" ]; then (echo "run `make tag` only in master branch"; exit 1); fi
	@echo "tagging v$(VERSION)..."
	git tag v$(VERSION)
	git push origin v$(VERSION)
