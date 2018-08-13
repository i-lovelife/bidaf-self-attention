import pathlib
PROJECT_ROOT = (pathlib.Path(__file__).parent / ".." / "..").resolve()  # pylint: disable=no-member
MODULE_ROOT = PROJECT_ROOT / "source"
TESTS_ROOT = PROJECT_ROOT / "tests"
FIXTURES_ROOT = TESTS_ROOT / "fixtures"
