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
README.rst: README.md
	pandoc README.md -o README.rst
dist: README.rst
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
