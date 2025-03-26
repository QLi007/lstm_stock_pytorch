# Create and activate virtual environment
python -m venv venv
.\venv\Scripts\Activate.ps1

# Upgrade pip
python -m pip install --upgrade pip

# Install requirements
pip install -r requirements.txt

# Create necessary directories
mkdir -Force data
mkdir -Force models
mkdir -Force results
mkdir -Force logs

Write-Host "Virtual environment setup completed successfully!"
Write-Host "To activate the environment, run: .\venv\Scripts\Activate.ps1" 