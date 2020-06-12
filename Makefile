INSTALL_PATH=/usr/local
PYTHON=python3
VERSION=$(shell python3 -c "import ndd; print(ndd.__version__)")

default:
	$(PYTHON) setup.py install --prefix $(INSTALL_PATH)
build:
	$(PYTHON) setup.py build
install:
	$(PYTHON) setup.py install --prefix $(INSTALL_PATH)
user:
	$(PYTHON) setup.py install --prefix $(HOME)/.local
dist:
	@if [ "$(shell git rev-parse --abbrev-ref HEAD)" != "master" ]; then (echo "run `make dist` only in master branch"; exit 1); fi
	rm -f dist/*
	$(PYTHON) setup.py sdist
tag:
	@if [ "$(shell git rev-parse --abbrev-ref HEAD)" != "master" ]; then (echo "run `make tag` only in master branch"; exit 1); fi
	@echo "tagging v$(VERSION)..."
	git tag v$(VERSION)
	git push origin v$(VERSION)
test:
	(cd tests; pytest -v)
clean:
	rm -r build README.rst
f2py:
	(cd ndd/exts; rm nsb.pyf; f2py estimators.f90 -m fnsb -h nsb.pyf; mv nsb.pyf ../)
