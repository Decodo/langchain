
.PHONY: all lint lint_package lint_tests format test tests integration_tests help

PYTEST_EXTRA ?=

all: help

######################
# LINTING AND FORMATTING
######################

lint format: PYTHON_FILES=.
lint_package: PYTHON_FILES=langchain_decodo
lint_tests: PYTHON_FILES=tests
lint_tests: MYPY_CACHE=.mypy_cache_test
MYPY_CACHE=.mypy_cache

lint lint_package lint_tests:
	uv run --group lint --group typing ruff check $(PYTHON_FILES)
	uv run --group lint --group typing ruff format $(PYTHON_FILES) --diff
	mkdir -p $(MYPY_CACHE) && uv run --group lint --group typing mypy $(PYTHON_FILES) --cache-dir $(MYPY_CACHE)

format:
	uv run --group lint ruff format $(PYTHON_FILES)
	uv run --group lint ruff check --fix $(PYTHON_FILES)

######################
# TESTING
######################

tests test:
	uv run --group test pytest $(PYTEST_EXTRA) tests/unit_tests

integration_tests:
	uv run --group test --group test_integration pytest tests/integration_tests

######################
# HELP
######################

help:
	@echo '----'
	@echo 'lint                         - run linters'
	@echo 'format                       - run formatters'
	@echo 'tests                        - run unit tests'
	@echo 'integration_tests            - run integration tests (requires an API key in DECODO_API_TOKEN)'
