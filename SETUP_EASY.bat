@echo off
REM ============================================================================
REM Fed-AuditGAN Easy Setup Script for Windows
REM ============================================================================
REM This script automatically sets up the Fed-AuditGAN environment on Windows
REM ============================================================================

echo.
echo ================================================================================
echo Fed-AuditGAN: Fairness-Aware Federated Learning - Easy Setup
echo ================================================================================
echo.
echo This script will:
echo   1. Create a conda environment named 'fed-audit-gan'
echo   2. Install PyTorch with CUDA support
echo   3. Install all required dependencies
echo   4. Verify the installation
echo.
echo Estimated time: 5-15 minutes
echo.
pause

REM Check if Anaconda/Miniconda is installed
echo [1/5] Checking for Anaconda/Miniconda...
where conda >nul 2>nul
if %errorlevel% neq 0 (
    echo.
    echo ERROR: Conda not found!
    echo Please install Anaconda or Miniconda first:
    echo https://www.anaconda.com/products/distribution
    echo.
    pause
    exit /b 1
)
echo ✓ Conda found

REM Initialize conda for batch script
echo [2/5] Initializing conda...
call C:\Users\%USERNAME%\anaconda3\condabin\conda.bat activate base
if %errorlevel% neq 0 (
    REM Try Miniconda path
    call C:\ProgramData\Miniconda3\condabin\conda.bat activate base
)

REM Create conda environment
echo [3/5] Creating conda environment 'fed-audit-gan'...
call conda env create -f environment.yml
if %errorlevel% neq 0 (
    echo.
    echo Environment creation failed. Trying manual installation...
    call conda create -n fed-audit-gan python=3.9 -y
    call conda activate fed-audit-gan
    
    REM Install PyTorch with CUDA
    echo Installing PyTorch with CUDA support...
    call conda install pytorch torchvision torchaudio pytorch-cuda=11.6 -c pytorch -c nvidia -y
    if %errorlevel% neq 0 (
        echo CUDA installation failed. Installing CPU version...
        call conda install pytorch torchvision torchaudio cpuonly -c pytorch -y
    )
    
    REM Install other dependencies
    echo Installing other dependencies...
    call conda install numpy matplotlib -y
    call pip install tqdm wandb
)

REM Activate environment
echo [4/5] Activating environment...
call conda activate fed-audit-gan

REM Verify installation
echo [5/5] Verifying installation...
python -c "import torch; print('✓ PyTorch version:', torch.__version__); print('✓ CUDA available:', torch.cuda.is_available())"
python -c "import numpy; print('✓ NumPy version:', numpy.__version__)"
python -c "import torchvision; print('✓ TorchVision version:', torchvision.__version__)"
python -c "import matplotlib; print('✓ Matplotlib version:', matplotlib.__version__)"
python -c "import tqdm; print('✓ tqdm installed')"

echo.
echo ================================================================================
echo Setup Complete!
echo ================================================================================
echo.
echo To use Fed-AuditGAN:
echo   1. Activate environment: conda activate fed-audit-gan
echo   2. Run training: python fed_audit_gan.py --dataset mnist --use_audit_gan
echo   3. Or use the launcher: start_fed_audit_gan.bat
echo.
echo For help: python fed_audit_gan.py --help
echo.
pause
