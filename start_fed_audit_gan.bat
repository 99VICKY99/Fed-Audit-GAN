@echo off
REM ============================================================================
REM Fed-AuditGAN Launcher for Windows
REM ============================================================================
REM Interactive menu for running Fed-AuditGAN experiments
REM ============================================================================

:MENU
cls
echo.
echo ================================================================================
echo Fed-AuditGAN: Fairness-Aware Federated Learning
echo ================================================================================
echo.
echo Select an experiment to run:
echo.
echo MNIST Experiments:
echo   [1] MNIST - IID - Standard FedAvg
echo   [2] MNIST - IID - Fed-AuditGAN (gamma=0.3, accuracy-focused)
echo   [3] MNIST - IID - Fed-AuditGAN (gamma=0.5, balanced)
echo   [4] MNIST - IID - Fed-AuditGAN (gamma=0.7, fairness-focused)
echo   [5] MNIST - Shard Non-IID - Fed-AuditGAN (balanced)
echo   [6] MNIST - Dirichlet Non-IID - Fed-AuditGAN (balanced)
echo.
echo CIFAR-10 Experiments:
echo   [7] CIFAR-10 - IID - Standard FedAvg
echo   [8] CIFAR-10 - IID - Fed-AuditGAN (balanced)
echo   [9] CIFAR-10 - Shard Non-IID - Fed-AuditGAN (balanced)
echo   [A] CIFAR-10 - Dirichlet Non-IID - Fed-AuditGAN (balanced)
echo.
echo CIFAR-100 Experiments:
echo   [B] CIFAR-100 - IID - Fed-AuditGAN (balanced)
echo   [C] CIFAR-100 - Shard Non-IID - Fed-AuditGAN (balanced)
echo.
echo Other:
echo   [Q] Quit
echo   [H] Help / Custom Parameters
echo.
set /p choice="Enter your choice: "

REM Initialize conda
call C:\Users\%USERNAME%\anaconda3\condabin\conda.bat activate base
if %errorlevel% neq 0 (
    call C:\ProgramData\Miniconda3\condabin\conda.bat activate base
)
call conda activate fed-audit-gan

REM Process choice
if /i "%choice%"=="1" (
    echo Running MNIST - IID - Standard FedAvg...
    python fed_audit_gan.py --dataset mnist --partition_mode iid --n_epochs 50 --exp_name "MNIST_IID_FedAvg"
    goto END
)
if /i "%choice%"=="2" (
    echo Running MNIST - IID - Fed-AuditGAN (gamma=0.3)...
    python fed_audit_gan.py --dataset mnist --partition_mode iid --use_audit_gan --gamma 0.3 --n_epochs 50 --wandb --exp_name "MNIST_IID_AuditGAN_g03"
    goto END
)
if /i "%choice%"=="3" (
    echo Running MNIST - IID - Fed-AuditGAN (gamma=0.5)...
    python fed_audit_gan.py --dataset mnist --partition_mode iid --use_audit_gan --gamma 0.5 --n_epochs 50 --wandb --exp_name "MNIST_IID_AuditGAN_g05"
    goto END
)
if /i "%choice%"=="4" (
    echo Running MNIST - IID - Fed-AuditGAN (gamma=0.7)...
    python fed_audit_gan.py --dataset mnist --partition_mode iid --use_audit_gan --gamma 0.7 --n_epochs 50 --wandb --exp_name "MNIST_IID_AuditGAN_g07"
    goto END
)
if /i "%choice%"=="5" (
    echo Running MNIST - Shard Non-IID - Fed-AuditGAN...
    python fed_audit_gan.py --dataset mnist --partition_mode shard --use_audit_gan --gamma 0.5 --n_epochs 50 --wandb --exp_name "MNIST_Shard_AuditGAN"
    goto END
)
if /i "%choice%"=="6" (
    echo Running MNIST - Dirichlet Non-IID - Fed-AuditGAN...
    python fed_audit_gan.py --dataset mnist --partition_mode dirichlet --dirichlet_alpha 0.1 --use_audit_gan --gamma 0.5 --n_epochs 50 --wandb --exp_name "MNIST_Dirichlet_AuditGAN"
    goto END
)
if /i "%choice%"=="7" (
    echo Running CIFAR-10 - IID - Standard FedAvg...
    python fed_audit_gan.py --dataset cifar10 --partition_mode iid --n_epochs 60 --exp_name "CIFAR10_IID_FedAvg"
    goto END
)
if /i "%choice%"=="8" (
    echo Running CIFAR-10 - IID - Fed-AuditGAN...
    python fed_audit_gan.py --dataset cifar10 --partition_mode iid --use_audit_gan --gamma 0.5 --n_epochs 60 --wandb --exp_name "CIFAR10_IID_AuditGAN"
    goto END
)
if /i "%choice%"=="9" (
    echo Running CIFAR-10 - Shard Non-IID - Fed-AuditGAN...
    python fed_audit_gan.py --dataset cifar10 --partition_mode shard --use_audit_gan --gamma 0.5 --n_epochs 60 --wandb --exp_name "CIFAR10_Shard_AuditGAN"
    goto END
)
if /i "%choice%"=="A" (
    echo Running CIFAR-10 - Dirichlet Non-IID - Fed-AuditGAN...
    python fed_audit_gan.py --dataset cifar10 --partition_mode dirichlet --dirichlet_alpha 0.1 --use_audit_gan --gamma 0.5 --n_epochs 60 --wandb --exp_name "CIFAR10_Dirichlet_AuditGAN"
    goto END
)
if /i "%choice%"=="B" (
    echo Running CIFAR-100 - IID - Fed-AuditGAN...
    python fed_audit_gan.py --dataset cifar100 --partition_mode iid --use_audit_gan --gamma 0.5 --n_epochs 80 --wandb --exp_name "CIFAR100_IID_AuditGAN"
    goto END
)
if /i "%choice%"=="C" (
    echo Running CIFAR-100 - Shard Non-IID - Fed-AuditGAN...
    python fed_audit_gan.py --dataset cifar100 --partition_mode shard --use_audit_gan --gamma 0.5 --n_epochs 80 --wandb --exp_name "CIFAR100_Shard_AuditGAN"
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
    echo python fed_audit_gan.py ^
    echo     --dataset mnist ^
    echo     --model_name cnn ^
    echo     --partition_mode shard ^
    echo     --n_clients 10 ^
    echo     --n_epochs 50 ^
    echo     --use_audit_gan ^
    echo     --gamma 0.5 ^
    echo     --n_audit_steps 100 ^
    echo     --wandb ^
    echo     --exp_name "My_Experiment"
    echo.
    echo For full help: python fed_audit_gan.py --help
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
