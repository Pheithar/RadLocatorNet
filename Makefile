documentation:
	@echo "Generating documentation..."
	@cd docs && sphinx-apidoc -o . ../radlocatornet --force && make html
	
	