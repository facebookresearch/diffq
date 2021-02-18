default: tests

all: default docs upload dist

tests:
	coverage run -m unittest discover -s diffq/tests || exit 1
	coverage report --include 'diffq/*'

docs:
	pdoc3 --html -o docs -f diffq

dist: docs
	python3 setup.py sdist

upload: docs
		rsync -ar docs bob:www/share/diffq/

clean:
	rm -r docs dist build *.egg-info

live:
	pdoc3 --http : diffq


.PHONY: tests docs dist
