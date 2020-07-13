# -*- Makefile -*-

TEST_LOCAT=test/

clean:
	@echo "Find and remove pycache."
	
	@echo "Remove past Python built."
	rm -rf *.egg-info/
	rm -rf dist/
	
	@echo "Remove Python caches."
test:
	pytest $(TEST_LOCAT)

all:
	@echo "Test"

install:
	@echo "Downloading all dependencies."
