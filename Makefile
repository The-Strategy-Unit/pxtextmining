coverage:
	pytest --cov=pxtextmining --cov=api tests/ --cov-report xml:coverage.xml --cov-report term
