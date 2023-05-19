coverage:
	pytest --cov=. tests/ --cov-report xml:coverage.xml --cov-report term
