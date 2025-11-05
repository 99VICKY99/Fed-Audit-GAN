@echo off
REM ============================================================================
REM Fed-AuditGAN Simple Launcher for Windows
REM ============================================================================
REM Direct Python execution without conda activation
REM ============================================================================

set PYTHON_PATH=C:\Users\vicky\anaconda3\envs\fedavg\python.exe

:MENU
cls
echo.
echo ================================================================================
echo Fed-AuditGAN: Fairness-Aware Federated Learning
echo ================================================================================
echo.
echo Select an experiment to run:
echo.
echo MNIST Quick Tests (2 rounds):
echo   [1] MNIST - Standard FedAvg (quick test)
echo   [2] MNIST - Fed-AuditGAN gamma=0.5 (quick test)
echo.
echo MNIST Full Experiments (50 rounds):
echo   [3] MNIST - IID - Standard FedAvg
echo   [4] MNIST - IID - Fed-AuditGAN (gamma=0.3, accuracy-focused)
echo   [5] MNIST - IID - Fed-AuditGAN (gamma=0.5, balanced)
echo   [6] MNIST - IID - Fed-AuditGAN (gamma=0.7, fairness-focused)
echo.
echo CIFAR-10 Experiments:
echo   [7] CIFAR-10 - IID - Standard FedAvg
echo   [8] CIFAR-10 - IID - Fed-AuditGAN (balanced)
echo.
echo Other:
echo   [Q] Quit
echo   [H] Help / Custom Parameters
echo.
set /p choice="Enter your choice: "

REM Process choice
if /i "%choice%"=="1" (
    echo Running MNIST - Standard FedAvg - quick test...
    "%PYTHON_PATH%" fed_audit_gan.py --dataset mnist --n_clients 3 --n_epochs 2 --n_client_epochs 1 --batch_size 32 --device cpu --exp_name "MNIST_FedAvg_test"
    goto END
)
if /i "%choice%"=="2" (
    echo Running MNIST - Fed-AuditGAN gamma=0.5 - quick test...
    "%PYTHON_PATH%" fed_audit_gan.py --dataset mnist --use_audit_gan --gamma 0.5 --n_clients 3 --n_epochs 2 --n_client_epochs 1 --n_audit_steps 3 --n_probes 100 --batch_size 32 --device cpu --exp_name "MNIST_AuditGAN_test"
    goto END
)
if /i "%choice%"=="3" (
    echo Running MNIST - IID - Standard FedAvg - 50 rounds...
    "%PYTHON_PATH%" fed_audit_gan.py --dataset mnist --partition_mode iid --n_epochs 50 --exp_name "MNIST_IID_FedAvg"
    goto END
)
if /i "%choice%"=="4" (
    echo Running MNIST - IID - Fed-AuditGAN - gamma=0.3...
    "%PYTHON_PATH%" fed_audit_gan.py --dataset mnist --partition_mode iid --use_audit_gan --gamma 0.3 --n_epochs 50 --wandb --exp_name "MNIST_IID_AuditGAN_g03"
    goto END
)
if /i "%choice%"=="5" (
    echo Running MNIST - IID - Fed-AuditGAN - gamma=0.5...
    "%PYTHON_PATH%" fed_audit_gan.py --dataset mnist --partition_mode iid --use_audit_gan --gamma 0.5 --n_epochs 50 --wandb --exp_name "MNIST_IID_AuditGAN_g05"
    goto END
)
if /i "%choice%"=="6" (
    echo Running MNIST - IID - Fed-AuditGAN - gamma=0.7...
    "%PYTHON_PATH%" fed_audit_gan.py --dataset mnist --partition_mode iid --use_audit_gan --gamma 0.7 --n_epochs 50 --wandb --exp_name "MNIST_IID_AuditGAN_g07"
    goto END
)
if /i "%choice%"=="7" (
    echo Running CIFAR-10 - IID - Standard FedAvg...
    "%PYTHON_PATH%" fed_audit_gan.py --dataset cifar10 --partition_mode iid --n_epochs 60 --exp_name "CIFAR10_IID_FedAvg"
    goto END
)
if /i "%choice%"=="8" (
    echo Running CIFAR-10 - IID - Fed-AuditGAN - balanced...
    "%PYTHON_PATH%" fed_audit_gan.py --dataset cifar10 --partition_mode iid --use_audit_gan --gamma 0.5 --n_epochs 60 --wandb --exp_name "CIFAR10_IID_AuditGAN"
    goto END
)
if /i "%choice%"=="Q" (
    echo Exiting...
    exit /b 0
)
if /i "%choice%"=="H" (
    echo.
    echo ========================================================================
    echo Custom Parameters Example:
    echo ========================================================================
    echo.
    echo "%PYTHON_PATH%" fed_audit_gan.py ^
    echo     --dataset mnist ^
    echo     --model_name cnn ^
    echo     --partition_mode shard ^
    echo     --n_clients 10 ^
    echo     --n_epochs 50 ^
    echo     --use_audit_gan ^
    echo     --gamma 0.5 ^
    echo     --n_audit_steps 100 ^
    echo     --exp_name "My_Experiment"
    echo.
    echo For full help: "%PYTHON_PATH%" fed_audit_gan.py --help
    echo.
    pause
    goto MENU
)

echo Invalid choice. Please try again.
pause
goto MENU

:END
echo.
echo ================================================================================
echo Experiment completed!
echo Results saved in ./results/
echo ================================================================================
echo.
pause
