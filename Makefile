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
env:
	(pip install -r requirements.txt; $(PYTHON) setup.py install --prefix ./venv)
test:
	(cd tests; pytest -v)
basic_test:
	(cd tests; pytest -v)
clean:
	rm -r build
