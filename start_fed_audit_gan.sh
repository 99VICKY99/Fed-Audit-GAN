#!/bin/bash
# ============================================================================
# Fed-AuditGAN Launcher for Linux/Mac
# ============================================================================
# Interactive menu for running Fed-AuditGAN experiments
# ============================================================================

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

while true; do
    clear
    echo ""
    echo "================================================================================"
    echo "Fed-AuditGAN: Fairness-Aware Federated Learning"
    echo "================================================================================"
    echo ""
    echo "Select an experiment to run:"
    echo ""
    echo "MNIST Experiments:"
    echo "  [1] MNIST - IID - Standard FedAvg"
    echo "  [2] MNIST - IID - Fed-AuditGAN (gamma=0.3, accuracy-focused)"
    echo "  [3] MNIST - IID - Fed-AuditGAN (gamma=0.5, balanced)"
    echo "  [4] MNIST - IID - Fed-AuditGAN (gamma=0.7, fairness-focused)"
    echo "  [5] MNIST - Shard Non-IID - Fed-AuditGAN (balanced)"
    echo "  [6] MNIST - Dirichlet Non-IID - Fed-AuditGAN (balanced)"
    echo ""
    echo "CIFAR-10 Experiments:"
    echo "  [7] CIFAR-10 - IID - Standard FedAvg"
    echo "  [8] CIFAR-10 - IID - Fed-AuditGAN (balanced)"
    echo "  [9] CIFAR-10 - Shard Non-IID - Fed-AuditGAN (balanced)"
    echo "  [10] CIFAR-10 - Dirichlet Non-IID - Fed-AuditGAN (balanced)"
    echo ""
    echo "CIFAR-100 Experiments:"
    echo "  [11] CIFAR-100 - IID - Fed-AuditGAN (balanced)"
    echo "  [12] CIFAR-100 - Shard Non-IID - Fed-AuditGAN (balanced)"
    echo ""
    echo "Other:"
    echo "  [Q] Quit"
    echo "  [H] Help / Custom Parameters"
    echo ""
    read -p "Enter your choice: " choice

    # Activate conda environment
    eval "$(conda shell.bash hook)"
    conda activate fed-audit-gan

    case $choice in
        1)
            echo -e "${GREEN}Running MNIST - IID - Standard FedAvg...${NC}"
            python fed_audit_gan.py --dataset mnist --partition_mode iid --n_epochs 50 --exp_name "MNIST_IID_FedAvg"
            break
            ;;
        2)
            echo -e "${GREEN}Running MNIST - IID - Fed-AuditGAN (gamma=0.3)...${NC}"
            python fed_audit_gan.py --dataset mnist --partition_mode iid --use_audit_gan --gamma 0.3 --n_epochs 50 --wandb --exp_name "MNIST_IID_AuditGAN_g03"
            break
            ;;
        3)
            echo -e "${GREEN}Running MNIST - IID - Fed-AuditGAN (gamma=0.5)...${NC}"
            python fed_audit_gan.py --dataset mnist --partition_mode iid --use_audit_gan --gamma 0.5 --n_epochs 50 --wandb --exp_name "MNIST_IID_AuditGAN_g05"
            break
            ;;
        4)
            echo -e "${GREEN}Running MNIST - IID - Fed-AuditGAN (gamma=0.7)...${NC}"
            python fed_audit_gan.py --dataset mnist --partition_mode iid --use_audit_gan --gamma 0.7 --n_epochs 50 --wandb --exp_name "MNIST_IID_AuditGAN_g07"
            break
            ;;
        5)
            echo -e "${GREEN}Running MNIST - Shard Non-IID - Fed-AuditGAN...${NC}"
            python fed_audit_gan.py --dataset mnist --partition_mode shard --use_audit_gan --gamma 0.5 --n_epochs 50 --wandb --exp_name "MNIST_Shard_AuditGAN"
            break
            ;;
        6)
            echo -e "${GREEN}Running MNIST - Dirichlet Non-IID - Fed-AuditGAN...${NC}"
            python fed_audit_gan.py --dataset mnist --partition_mode dirichlet --dirichlet_alpha 0.1 --use_audit_gan --gamma 0.5 --n_epochs 50 --wandb --exp_name "MNIST_Dirichlet_AuditGAN"
            break
            ;;
        7)
            echo -e "${GREEN}Running CIFAR-10 - IID - Standard FedAvg...${NC}"
            python fed_audit_gan.py --dataset cifar10 --partition_mode iid --n_epochs 60 --exp_name "CIFAR10_IID_FedAvg"
            break
            ;;
        8)
            echo -e "${GREEN}Running CIFAR-10 - IID - Fed-AuditGAN...${NC}"
            python fed_audit_gan.py --dataset cifar10 --partition_mode iid --use_audit_gan --gamma 0.5 --n_epochs 60 --wandb --exp_name "CIFAR10_IID_AuditGAN"
            break
            ;;
        9)
            echo -e "${GREEN}Running CIFAR-10 - Shard Non-IID - Fed-AuditGAN...${NC}"
            python fed_audit_gan.py --dataset cifar10 --partition_mode shard --use_audit_gan --gamma 0.5 --n_epochs 60 --wandb --exp_name "CIFAR10_Shard_AuditGAN"
            break
            ;;
        10)
            echo -e "${GREEN}Running CIFAR-10 - Dirichlet Non-IID - Fed-AuditGAN...${NC}"
            python fed_audit_gan.py --dataset cifar10 --partition_mode dirichlet --dirichlet_alpha 0.1 --use_audit_gan --gamma 0.5 --n_epochs 60 --wandb --exp_name "CIFAR10_Dirichlet_AuditGAN"
            break
            ;;
        11)
            echo -e "${GREEN}Running CIFAR-100 - IID - Fed-AuditGAN...${NC}"
            python fed_audit_gan.py --dataset cifar100 --partition_mode iid --use_audit_gan --gamma 0.5 --n_epochs 80 --wandb --exp_name "CIFAR100_IID_AuditGAN"
            break
            ;;
        12)
            echo -e "${GREEN}Running CIFAR-100 - Shard Non-IID - Fed-AuditGAN...${NC}"
            python fed_audit_gan.py --dataset cifar100 --partition_mode shard --use_audit_gan --gamma 0.5 --n_epochs 80 --wandb --exp_name "CIFAR100_Shard_AuditGAN"
            break
            ;;
        [Qq])
            echo "Exiting..."
            exit 0
            ;;
        [Hh])
            echo ""
            echo "========================================================================"
            echo "Custom Parameters Example:"
            echo "========================================================================"
            echo ""
            echo "python fed_audit_gan.py \\"
            echo "    --dataset mnist \\"
            echo "    --model_name cnn \\"
            echo "    --partition_mode shard \\"
            echo "    --n_clients 10 \\"
            echo "    --n_epochs 50 \\"
            echo "    --use_audit_gan \\"
            echo "    --gamma 0.5 \\"
            echo "    --n_audit_steps 100 \\"
            echo "    --wandb \\"
            echo "    --exp_name \"My_Experiment\""
            echo ""
            echo "For full help: python fed_audit_gan.py --help"
            echo ""
            read -p "Press Enter to continue..."
            ;;
        *)
            echo -e "${RED}Invalid choice. Please try again.${NC}"
            sleep 2
            ;;
    esac
done

echo ""
echo "================================================================================"
echo "Experiment completed!"
echo "Results saved in ./results/"
echo "================================================================================"
echo ""
