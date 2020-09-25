# -*- Makefile -*-

TEST_LOCAT=test/

clean:
	@echo "Find and remove pycache."
	@echo "Remove past Python built."
	rm -rf *.egg-info/
	rm -rf dist/
	rm -rf *tmp/
	
	@echo "Remove Python caches (__pycache__, .pyc, .pyo)."
	# From: https://stackoverflow.com/questions/28991015/python3-project-remove-pycache-folders-and-pyc-files/46822695
	find . | grep -E "(__pycache__|\.pyc|\.pyo$)" | xargs rm -rf
test:
	pytest $(TEST_LOCAT)

all:
	@echo "Test"

install:
	@echo "Downloading all dependencies."
