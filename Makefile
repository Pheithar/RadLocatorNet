documentation:
	@echo "Generating documentation..."
	@cd docs && sphinx-apidoc -o . ../radlocatornet --force && make html

updaterequirements: install_dev
	@echo "Updating requirements..."
	@pipreqs --force .

install: install_dev
	@echo "Installing requirements..."
	@pip install -r requirements.txt

install_dev:
	@echo "Installing development requirements..."
	@pip install -r requirements_dev.txt
	