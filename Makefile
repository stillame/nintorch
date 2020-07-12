# -*- Makefile -*-

TEST_LOCAT=test/

clean:
	@echo "Find and remove pycache."
	
	@echo "Remove past Python built."
	rm -rf *.egg-info
    
test:
	pytest $(TEST_LOCAT)

all:
	@echo "Test"

install:
	@echo "Downloading all dependencies."
