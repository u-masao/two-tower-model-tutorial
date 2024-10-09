#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_NAME = two-tower-model-tutorial
PYTHON_VERSION = 3.10
PYTHON_INTERPRETER = python

#################################################################################
# COMMANDS                                                                      #
#################################################################################

## run dvc repro
.PHONY: repro
repro: check_commit PIPELINE.md
	poetry run dvc repro
	git commit dvc.lock -m 'run dvc repro' || true

## check commit
.PHONY: check_commit
check_commit: lint
	git status
	git diff --exit-code
	git diff --exit-code --staged

## make pipeline file
PIPELINE.md: dvc.yaml params.yaml
	echo '# pipeline DAG\n\n' > $@
	echo '## process DAG\n\n' >> $@
	poetry run dvc dag --md >> $@
	echo '\n\n## output file DAG\n\n' >> $@
	poetry run dvc dag --md --out >> $@
	git commit $@ -m 'update dvc pipeline'

## Install Python Dependencies
.PHONY: requirements
requirements:
	$(PYTHON_INTERPRETER) -m pip install -U pip
	$(PYTHON_INTERPRETER) -m pip install -r requirements.txt

## Delete all compiled Python files
.PHONY: clean
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete

## lint and formatting
.PHONY: lint
lint:
	poetry run isort src
	poetry run black src -l 79
	poetry run flake8 src

## Format source code with black
.PHONY: format
format:
	black --config pyproject.toml src






#################################################################################
# PROJECT RULES                                                                 #
#################################################################################


## Make Dataset
.PHONY: data
data: requirements
	$(PYTHON_INTERPRETER) src/dataset.py


#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

.DEFAULT_GOAL := help

define PRINT_HELP_PYSCRIPT
import re, sys; \
lines = '\n'.join([line for line in sys.stdin]); \
matches = re.findall(r'\n## (.*)\n[\s\S]+?\n([a-zA-Z_-]+):', lines); \
print('Available rules:\n'); \
print('\n'.join(['{:25}{}'.format(*reversed(match)) for match in matches]))
endef
export PRINT_HELP_PYSCRIPT

help:
	@$(PYTHON_INTERPRETER) -c "${PRINT_HELP_PYSCRIPT}" < $(MAKEFILE_LIST)
