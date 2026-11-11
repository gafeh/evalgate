"""Pure evaluation logic. No I/O, no database, no network.

Nothing in this package may import a driver, a web framework, an HTTP
client, or any other evalgate subpackage. tests/test_architecture.py
enforces that.
"""