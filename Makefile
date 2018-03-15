INSTALL_PATH=/usr/local
PYTHON=python3

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
	$(PYTHON) setup.py sdist
test:
	(cd tests; pytest -v)
clean:
	rm -r build
