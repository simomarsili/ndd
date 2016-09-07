INSTALL_PATH = /usr/local

default:
	python setup.py install --prefix $(INSTALL_PATH)
build:
	python setup.py build
install:
	python setup.py install --prefix $(INSTALL_PATH)
user: 
	python setup.py install --prefix $(HOME)/.local
env:
	(pip install -r requirements.txt; python setup.py install --prefix ./venv)
test:
	(cd tests; python test_basic.py)
basic_test:
	(cd tests; python test_basic.py)
advanced_test:
	(cd tests; python test_advanced.py > out; diff out OUT_1.0)
clean:
	rm -r build
