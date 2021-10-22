default: tests

all: linter tests docs dist

linter:
	flake8 diffq

tests:
	python3 setup.py build_ext --inplace
	coverage run -m unittest discover -s diffq/tests || exit 1
	coverage report --include 'diffq/*'

docs:
	pdoc3 --html -o docs -f diffq

dist: docs
	python3 setup.py sdist

clean:
	rm -r docs dist build *.egg-info

live:
	pdoc3 --http : diffq

examples:
	./examples/setup_repo.sh fairseq
	./examples/setup_repo.sh deit

patches:
	./examples/update_patch.sh fairseq
	./examples/update_patch.sh deit

reset:
	./examples/reset_repo.sh fairseq
	./examples/reset_repo.sh deit

wheels:
	python3 get_build.py
	cd dist && unzip build.zip

.PHONY: linter tests docs dist examples patches reset
