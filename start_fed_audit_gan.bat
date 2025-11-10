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
echo Quick Tests (2 rounds):
echo   [1] MNIST-IID - Standard FedAvg
echo   [2] MNIST-IID - Fed-AuditGAN gamma=0.5
echo.
echo MNIST-IID Gamma Comparison (50 rounds):
echo   [3] Run ALL gamma values - 0.0, 0.3, 0.5, 0.7, 1.0 (IID)
echo   [4] Gamma=0.0 - Pure Accuracy
echo   [5] Gamma=0.3 - Accuracy-Focused
echo   [6] Gamma=0.5 - Balanced
echo   [7] Gamma=0.7 - Fairness-Focused
echo   [8] Gamma=1.0 - Pure Fairness
echo.
echo MNIST Non-IID Experiments (50 rounds):
echo   [A] Shard Non-IID - Gamma=0.5 balanced
echo   [B] Shard Non-IID - Run ALL gamma values
echo   [D] Dirichlet Non-IID - Gamma=0.5 balanced
echo   [E] Dirichlet Non-IID - Run ALL gamma values
echo.
echo CIFAR-10 Experiments (60 rounds):
echo   [F] CIFAR-10 IID - Gamma=0.5 balanced
echo   [G] CIFAR-10 IID - Run ALL gamma values
echo   [I] CIFAR-10 Shard Non-IID - Gamma=0.5
echo   [K] CIFAR-10 Shard Non-IID - Run ALL gamma values
echo   [J] CIFAR-10 Dirichlet Non-IID - Gamma=0.5
echo   [L] CIFAR-10 Dirichlet Non-IID - Run ALL gamma values
echo.
echo Baseline (No Fed-AuditGAN):
echo   [9] MNIST-IID - Standard FedAvg baseline
echo   [M] MNIST Shard Non-IID - Standard FedAvg baseline
echo   [N] MNIST Dirichlet Non-IID - Standard FedAvg baseline
echo   [C] CIFAR-10 IID - Standard FedAvg baseline
echo   [O] CIFAR-10 Shard Non-IID - Standard FedAvg baseline
echo   [P] CIFAR-10 Dirichlet Non-IID - Standard FedAvg baseline
echo.
echo Other:
echo   [Q] Quit
echo   [H] Help / Custom Parameters
echo   [T] Test Gamma Parameter (Debug)
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
    echo.
    echo ========================================================================
    echo Running COMPLETE GAMMA COMPARISON STUDY
    echo ========================================================================
    echo This will run 5 experiments sequentially:
    echo   1. Gamma=0.0 - Pure Accuracy
    echo   2. Gamma=0.3 - Accuracy-Focused  
    echo   3. Gamma=0.5 - Balanced
    echo   4. Gamma=0.7 - Fairness-Focused
    echo   5. Gamma=1.0 - Pure Fairness
    echo.
    echo Total time: ~5-10 hours
    echo Results will be logged to WandB for comparison
    echo.
    pause
    
    echo.
    echo [1/5] Running Gamma=0.0 - Pure Accuracy...
    "%PYTHON_PATH%" fed_audit_gan.py --dataset mnist --partition_mode iid --use_audit_gan --gamma 0.0 --n_epochs 50 --wandb --exp_name "MNIST_Gamma_0.0_PureAccuracy"
    
    echo.
    echo [2/5] Running Gamma=0.3 - Accuracy-Focused...
    "%PYTHON_PATH%" fed_audit_gan.py --dataset mnist --partition_mode iid --use_audit_gan --gamma 0.3 --n_epochs 50 --wandb --exp_name "MNIST_Gamma_0.3_AccuracyFocused"
    
    echo.
    echo [3/5] Running Gamma=0.5 - Balanced...
    "%PYTHON_PATH%" fed_audit_gan.py --dataset mnist --partition_mode iid --use_audit_gan --gamma 0.5 --n_epochs 50 --wandb --exp_name "MNIST_Gamma_0.5_Balanced"
    
    echo.
    echo [4/5] Running Gamma=0.7 - Fairness-Focused...
    "%PYTHON_PATH%" fed_audit_gan.py --dataset mnist --partition_mode iid --use_audit_gan --gamma 0.7 --n_epochs 50 --wandb --exp_name "MNIST_Gamma_0.7_FairnessFocused"
    
    echo.
    echo [5/5] Running Gamma=1.0 - Pure Fairness...
    "%PYTHON_PATH%" fed_audit_gan.py --dataset mnist --partition_mode iid --use_audit_gan --gamma 1.0 --n_epochs 50 --wandb --exp_name "MNIST_Gamma_1.0_PureFairness"
    
    echo.
    echo ========================================================================
    echo All 5 experiments complete!
    echo Compare results on WandB to see gamma impact
    echo ========================================================================
    goto END
)
if /i "%choice%"=="4" (
    echo Running Gamma=0.0 - Pure Accuracy - NO fairness optimization...
    echo This uses standard FedAvg with Fed-AuditGAN infrastructure but 100%% accuracy focus
    "%PYTHON_PATH%" fed_audit_gan.py --dataset mnist --partition_mode iid --use_audit_gan --gamma 0.0 --n_epochs 50 --wandb --exp_name "MNIST_Gamma_0.0"
    goto END
)
if /i "%choice%"=="5" (
    echo Running Gamma=0.3 - Accuracy-Focused - 30%% fairness, 70%% accuracy...
    "%PYTHON_PATH%" fed_audit_gan.py --dataset mnist --partition_mode iid --use_audit_gan --gamma 0.3 --n_epochs 50 --wandb --exp_name "MNIST_Gamma_0.3"
    goto END
)
if /i "%choice%"=="6" (
    echo Running Gamma=0.5 - Balanced - 50%% fairness, 50%% accuracy...
    "%PYTHON_PATH%" fed_audit_gan.py --dataset mnist --partition_mode iid --use_audit_gan --gamma 0.5 --n_epochs 50 --wandb --exp_name "MNIST_Gamma_0.5"
    goto END
)
if /i "%choice%"=="7" (
    echo Running Gamma=0.7 - Fairness-Focused - 70%% fairness, 30%% accuracy...
    "%PYTHON_PATH%" fed_audit_gan.py --dataset mnist --partition_mode iid --use_audit_gan --gamma 0.7 --n_epochs 50 --wandb --exp_name "MNIST_Gamma_0.7"
    goto END
)
if /i "%choice%"=="8" (
    echo Running Gamma=1.0 - Pure Fairness - 100%% fairness optimization...
    echo This maximizes fairness at the cost of some accuracy
    "%PYTHON_PATH%" fed_audit_gan.py --dataset mnist --partition_mode iid --use_audit_gan --gamma 1.0 --n_epochs 50 --wandb --exp_name "MNIST_Gamma_1.0"
    goto END
)
if /i "%choice%"=="9" (
    echo Running MNIST-IID - Standard FedAvg - NO Fed-AuditGAN baseline...
    "%PYTHON_PATH%" fed_audit_gan.py --dataset mnist --partition_mode iid --n_epochs 50 --wandb --exp_name "MNIST_IID_FedAvg_Baseline"
    goto END
)
if /i "%choice%"=="M" (
    echo Running MNIST Shard Non-IID - Standard FedAvg baseline...
    echo NO Fed-AuditGAN - pure FedAvg for comparison
    "%PYTHON_PATH%" fed_audit_gan.py --dataset mnist --partition_mode shard --n_epochs 50 --wandb --exp_name "MNIST_Shard_FedAvg_Baseline"
    goto END
)
if /i "%choice%"=="N" (
    echo Running MNIST Dirichlet Non-IID - Standard FedAvg baseline...
    echo NO Fed-AuditGAN - pure FedAvg for comparison
    "%PYTHON_PATH%" fed_audit_gan.py --dataset mnist --partition_mode dirichlet --dirichlet_alpha 0.1 --n_epochs 50 --wandb --exp_name "MNIST_Dirichlet_FedAvg_Baseline"
    goto END
)
if /i "%choice%"=="C" (
    echo Running CIFAR-10 IID - Standard FedAvg baseline...
    echo NO Fed-AuditGAN - pure FedAvg for comparison
    "%PYTHON_PATH%" fed_audit_gan.py --dataset cifar10 --partition_mode iid --n_epochs 60 --wandb --exp_name "CIFAR10_IID_FedAvg_Baseline"
    goto END
)
if /i "%choice%"=="O" (
    echo Running CIFAR-10 Shard Non-IID - Standard FedAvg baseline...
    echo NO Fed-AuditGAN - pure FedAvg for comparison
    "%PYTHON_PATH%" fed_audit_gan.py --dataset cifar10 --partition_mode shard --n_epochs 60 --wandb --exp_name "CIFAR10_Shard_FedAvg_Baseline"
    goto END
)
if /i "%choice%"=="P" (
    echo Running CIFAR-10 Dirichlet Non-IID - Standard FedAvg baseline...
    echo NO Fed-AuditGAN - pure FedAvg for comparison
    "%PYTHON_PATH%" fed_audit_gan.py --dataset cifar10 --partition_mode dirichlet --dirichlet_alpha 0.1 --n_epochs 60 --wandb --exp_name "CIFAR10_Dirichlet_FedAvg_Baseline"
    goto END
)
REM ============================================================================
REM MNIST Non-IID Experiments
REM ============================================================================
if /i "%choice%"=="A" (
    echo Running MNIST Shard Non-IID - Gamma=0.5 balanced...
    echo Each client gets 2 class shards - realistic heterogeneity
    "%PYTHON_PATH%" fed_audit_gan.py --dataset mnist --partition_mode shard --use_audit_gan --gamma 0.5 --n_epochs 50 --wandb --exp_name "MNIST_Shard_Gamma_0.5"
    goto END
)
if /i "%choice%"=="B" (
    echo.
    echo ========================================================================
    echo Running MNIST Shard Non-IID - ALL GAMMA VALUES
    echo ========================================================================
    echo This will run 5 experiments with Shard Non-IID partitioning
    echo.
    pause
    
    echo [1/5] Shard Non-IID - Gamma=0.0...
    "%PYTHON_PATH%" fed_audit_gan.py --dataset mnist --partition_mode shard --use_audit_gan --gamma 0.0 --n_epochs 50 --wandb --exp_name "MNIST_Shard_Gamma_0.0"
    
    echo [2/5] Shard Non-IID - Gamma=0.3...
    "%PYTHON_PATH%" fed_audit_gan.py --dataset mnist --partition_mode shard --use_audit_gan --gamma 0.3 --n_epochs 50 --wandb --exp_name "MNIST_Shard_Gamma_0.3"
    
    echo [3/5] Shard Non-IID - Gamma=0.5...
    "%PYTHON_PATH%" fed_audit_gan.py --dataset mnist --partition_mode shard --use_audit_gan --gamma 0.5 --n_epochs 50 --wandb --exp_name "MNIST_Shard_Gamma_0.5"
    
    echo [4/5] Shard Non-IID - Gamma=0.7...
    "%PYTHON_PATH%" fed_audit_gan.py --dataset mnist --partition_mode shard --use_audit_gan --gamma 0.7 --n_epochs 50 --wandb --exp_name "MNIST_Shard_Gamma_0.7"
    
    echo [5/5] Shard Non-IID - Gamma=1.0...
    "%PYTHON_PATH%" fed_audit_gan.py --dataset mnist --partition_mode shard --use_audit_gan --gamma 1.0 --n_epochs 50 --wandb --exp_name "MNIST_Shard_Gamma_1.0"
    
    echo.
    echo All Shard Non-IID experiments complete!
    goto END
)
if /i "%choice%"=="D" (
    echo Running MNIST Dirichlet Non-IID - Gamma=0.5 balanced...
    echo Using Dirichlet alpha=0.1 for flexible heterogeneity
    "%PYTHON_PATH%" fed_audit_gan.py --dataset mnist --partition_mode dirichlet --dirichlet_alpha 0.1 --use_audit_gan --gamma 0.5 --n_epochs 50 --wandb --exp_name "MNIST_Dirichlet_Gamma_0.5"
    goto END
)
if /i "%choice%"=="E" (
    echo.
    echo ========================================================================
    echo Running MNIST Dirichlet Non-IID - ALL GAMMA VALUES
    echo ========================================================================
    echo This will run 5 experiments with Dirichlet Non-IID partitioning
    echo.
    pause
    
    echo [1/5] Dirichlet Non-IID - Gamma=0.0...
    "%PYTHON_PATH%" fed_audit_gan.py --dataset mnist --partition_mode dirichlet --dirichlet_alpha 0.1 --use_audit_gan --gamma 0.0 --n_epochs 50 --wandb --exp_name "MNIST_Dirichlet_Gamma_0.0"
    
    echo [2/5] Dirichlet Non-IID - Gamma=0.3...
    "%PYTHON_PATH%" fed_audit_gan.py --dataset mnist --partition_mode dirichlet --dirichlet_alpha 0.1 --use_audit_gan --gamma 0.3 --n_epochs 50 --wandb --exp_name "MNIST_Dirichlet_Gamma_0.3"
    
    echo [3/5] Dirichlet Non-IID - Gamma=0.5...
    "%PYTHON_PATH%" fed_audit_gan.py --dataset mnist --partition_mode dirichlet --dirichlet_alpha 0.1 --use_audit_gan --gamma 0.5 --n_epochs 50 --wandb --exp_name "MNIST_Dirichlet_Gamma_0.5"
    
    echo [4/5] Dirichlet Non-IID - Gamma=0.7...
    "%PYTHON_PATH%" fed_audit_gan.py --dataset mnist --partition_mode dirichlet --dirichlet_alpha 0.1 --use_audit_gan --gamma 0.7 --n_epochs 50 --wandb --exp_name "MNIST_Dirichlet_Gamma_0.7"
    
    echo [5/5] Dirichlet Non-IID - Gamma=1.0...
    "%PYTHON_PATH%" fed_audit_gan.py --dataset mnist --partition_mode dirichlet --dirichlet_alpha 0.1 --use_audit_gan --gamma 1.0 --n_epochs 50 --wandb --exp_name "MNIST_Dirichlet_Gamma_1.0"
    
    echo.
    echo All Dirichlet Non-IID experiments complete!
    goto END
)
REM ============================================================================
REM CIFAR-10 Experiments
REM ============================================================================
if /i "%choice%"=="F" (
    echo Running CIFAR-10 IID - Gamma=0.5 balanced...
    echo CIFAR-10 has 32x32 color images - more complex than MNIST
    "%PYTHON_PATH%" fed_audit_gan.py --dataset cifar10 --partition_mode iid --use_audit_gan --gamma 0.5 --n_epochs 60 --wandb --exp_name "CIFAR10_IID_Gamma_0.5"
    goto END
)
if /i "%choice%"=="G" (
    echo.
    echo ========================================================================
    echo Running CIFAR-10 IID - ALL GAMMA VALUES
    echo ========================================================================
    echo This will run 5 experiments with CIFAR-10 IID partitioning
    echo Note: CIFAR-10 takes longer than MNIST due to image complexity
    echo.
    pause
    
    echo [1/5] CIFAR-10 IID - Gamma=0.0...
    "%PYTHON_PATH%" fed_audit_gan.py --dataset cifar10 --partition_mode iid --use_audit_gan --gamma 0.0 --n_epochs 60 --wandb --exp_name "CIFAR10_IID_Gamma_0.0"
    
    echo [2/5] CIFAR-10 IID - Gamma=0.3...
    "%PYTHON_PATH%" fed_audit_gan.py --dataset cifar10 --partition_mode iid --use_audit_gan --gamma 0.3 --n_epochs 60 --wandb --exp_name "CIFAR10_IID_Gamma_0.3"
    
    echo [3/5] CIFAR-10 IID - Gamma=0.5...
    "%PYTHON_PATH%" fed_audit_gan.py --dataset cifar10 --partition_mode iid --use_audit_gan --gamma 0.5 --n_epochs 60 --wandb --exp_name "CIFAR10_IID_Gamma_0.5"
    
    echo [4/5] CIFAR-10 IID - Gamma=0.7...
    "%PYTHON_PATH%" fed_audit_gan.py --dataset cifar10 --partition_mode iid --use_audit_gan --gamma 0.7 --n_epochs 60 --wandb --exp_name "CIFAR10_IID_Gamma_0.7"
    
    echo [5/5] CIFAR-10 IID - Gamma=1.0...
    "%PYTHON_PATH%" fed_audit_gan.py --dataset cifar10 --partition_mode iid --use_audit_gan --gamma 1.0 --n_epochs 60 --wandb --exp_name "CIFAR10_IID_Gamma_1.0"
    
    echo.
    echo All CIFAR-10 IID experiments complete!
    goto END
)
if /i "%choice%"=="I" (
    echo Running CIFAR-10 Shard Non-IID - Gamma=0.5...
    echo CIFAR-10 with Shard Non-IID partitioning
    "%PYTHON_PATH%" fed_audit_gan.py --dataset cifar10 --partition_mode shard --use_audit_gan --gamma 0.5 --n_epochs 60 --wandb --exp_name "CIFAR10_Shard_Gamma_0.5"
    goto END
)
if /i "%choice%"=="K" (
    echo.
    echo ========================================================================
    echo Running CIFAR-10 Shard Non-IID - ALL GAMMA VALUES
    echo ========================================================================
    echo This will run 5 experiments with CIFAR-10 Shard Non-IID partitioning
    echo Note: CIFAR-10 takes longer than MNIST - Total time ~10-15 hours
    echo.
    pause
    
    echo [1/5] CIFAR-10 Shard Non-IID - Gamma=0.0...
    "%PYTHON_PATH%" fed_audit_gan.py --dataset cifar10 --partition_mode shard --use_audit_gan --gamma 0.0 --n_epochs 60 --wandb --exp_name "CIFAR10_Shard_Gamma_0.0"
    
    echo [2/5] CIFAR-10 Shard Non-IID - Gamma=0.3...
    "%PYTHON_PATH%" fed_audit_gan.py --dataset cifar10 --partition_mode shard --use_audit_gan --gamma 0.3 --n_epochs 60 --wandb --exp_name "CIFAR10_Shard_Gamma_0.3"
    
    echo [3/5] CIFAR-10 Shard Non-IID - Gamma=0.5...
    "%PYTHON_PATH%" fed_audit_gan.py --dataset cifar10 --partition_mode shard --use_audit_gan --gamma 0.5 --n_epochs 60 --wandb --exp_name "CIFAR10_Shard_Gamma_0.5"
    
    echo [4/5] CIFAR-10 Shard Non-IID - Gamma=0.7...
    "%PYTHON_PATH%" fed_audit_gan.py --dataset cifar10 --partition_mode shard --use_audit_gan --gamma 0.7 --n_epochs 60 --wandb --exp_name "CIFAR10_Shard_Gamma_0.7"
    
    echo [5/5] CIFAR-10 Shard Non-IID - Gamma=1.0...
    "%PYTHON_PATH%" fed_audit_gan.py --dataset cifar10 --partition_mode shard --use_audit_gan --gamma 1.0 --n_epochs 60 --wandb --exp_name "CIFAR10_Shard_Gamma_1.0"
    
    echo.
    echo All CIFAR-10 Shard Non-IID experiments complete!
    goto END
)
if /i "%choice%"=="J" (
    echo Running CIFAR-10 Dirichlet Non-IID - Gamma=0.5...
    echo CIFAR-10 with Dirichlet Non-IID partitioning - alpha=0.1
    "%PYTHON_PATH%" fed_audit_gan.py --dataset cifar10 --partition_mode dirichlet --dirichlet_alpha 0.1 --use_audit_gan --gamma 0.5 --n_epochs 60 --wandb --exp_name "CIFAR10_Dirichlet_Gamma_0.5"
    goto END
)
if /i "%choice%"=="L" (
    echo.
    echo ========================================================================
    echo Running CIFAR-10 Dirichlet Non-IID - ALL GAMMA VALUES
    echo ========================================================================
    echo This will run 5 experiments with CIFAR-10 Dirichlet Non-IID partitioning
    echo Using Dirichlet alpha=0.1 for high heterogeneity
    echo Note: CIFAR-10 takes longer than MNIST - Total time ~10-15 hours
    echo.
    pause
    
    echo [1/5] CIFAR-10 Dirichlet Non-IID - Gamma=0.0...
    "%PYTHON_PATH%" fed_audit_gan.py --dataset cifar10 --partition_mode dirichlet --dirichlet_alpha 0.1 --use_audit_gan --gamma 0.0 --n_epochs 60 --wandb --exp_name "CIFAR10_Dirichlet_Gamma_0.0"
    
    echo [2/5] CIFAR-10 Dirichlet Non-IID - Gamma=0.3...
    "%PYTHON_PATH%" fed_audit_gan.py --dataset cifar10 --partition_mode dirichlet --dirichlet_alpha 0.1 --use_audit_gan --gamma 0.3 --n_epochs 60 --wandb --exp_name "CIFAR10_Dirichlet_Gamma_0.3"
    
    echo [3/5] CIFAR-10 Dirichlet Non-IID - Gamma=0.5...
    "%PYTHON_PATH%" fed_audit_gan.py --dataset cifar10 --partition_mode dirichlet --dirichlet_alpha 0.1 --use_audit_gan --gamma 0.5 --n_epochs 60 --wandb --exp_name "CIFAR10_Dirichlet_Gamma_0.5"
    
    echo [4/5] CIFAR-10 Dirichlet Non-IID - Gamma=0.7...
    "%PYTHON_PATH%" fed_audit_gan.py --dataset cifar10 --partition_mode dirichlet --dirichlet_alpha 0.1 --use_audit_gan --gamma 0.7 --n_epochs 60 --wandb --exp_name "CIFAR10_Dirichlet_Gamma_0.7"
    
    echo [5/5] CIFAR-10 Dirichlet Non-IID - Gamma=1.0...
    "%PYTHON_PATH%" fed_audit_gan.py --dataset cifar10 --partition_mode dirichlet --dirichlet_alpha 0.1 --use_audit_gan --gamma 1.0 --n_epochs 60 --wandb --exp_name "CIFAR10_Dirichlet_Gamma_1.0"
    
    echo.
    echo All CIFAR-10 Dirichlet Non-IID experiments complete!
    goto END
)
if /i "%choice%"=="T" (
    echo.
    echo ========================================================================
    echo Testing Gamma Parameter
    echo ========================================================================
    echo This will run a debug script to verify the gamma parameter is working
    echo correctly by testing with simulated client contributions.
    echo.
    pause
    "%PYTHON_PATH%" scripts\debug_gamma_effect.py
    echo.
    pause
    goto MENU
)
if /i "%choice%"=="Q" (
    echo Exiting...
    exit /b 0
)
if /i "%choice%"=="H" (
    echo.
    echo ========================================================================
    echo Fed-AuditGAN Help
    echo ========================================================================
    echo.
    echo KEY CONCEPTS:
    echo.
    echo 1. GAMMA PARAMETER - Controls fairness vs accuracy trade-off:
    echo    gamma=0.0: Pure accuracy, no fairness optimization
    echo    gamma=0.3: 70%% accuracy, 30%% fairness
    echo    gamma=0.5: Balanced 50-50
    echo    gamma=0.7: 30%% accuracy, 70%% fairness
    echo    gamma=1.0: Pure fairness, maximum fairness focus
    echo.
    echo 2. DATA PARTITIONING:
    echo    IID: Uniform random distribution across clients
    echo    Shard: Each client gets 2 class shards - realistic heterogeneity
    echo    Dirichlet: Flexible heterogeneity using Dirichlet distribution
    echo.
    echo 3. DATASETS:
    echo    MNIST: 28x28 grayscale digits, faster training
    echo    CIFAR-10: 32x32 color images, slower but more complex
    echo.
    echo ========================================================================
    echo Custom Command Example:
    echo ========================================================================
    echo.
    echo "%PYTHON_PATH%" fed_audit_gan.py ^
    echo     --dataset mnist ^
    echo     --partition_mode dirichlet ^
    echo     --dirichlet_alpha 0.1 ^
    echo     --use_audit_gan ^
    echo     --gamma 0.7 ^
    echo     --n_clients 10 ^
    echo     --n_epochs 50 ^
    echo     --n_audit_steps 100 ^
    echo     --n_probes 3000 ^
    echo     --wandb ^
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
