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
	(cd tests; $(PYTHON) test_basic.py)
basic_test:
	(cd tests; $(PYTHON) test_basic.py)
advanced_test:
	(cd tests; $(PYTHON) test_advanced.py > out; diff out OUT_1.0)
clean:
	rm -r build
