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
    echo "MNIST Quick Tests (2 rounds):"
    echo "  [1] MNIST - Standard FedAvg (quick test)"
    echo "  [2] MNIST - Fed-AuditGAN gamma=0.5 (quick test)"
    echo ""
    echo "MNIST Full Experiments (50 rounds):"
    echo "  [3] MNIST - IID - Standard FedAvg"
    echo "  [4] MNIST - IID - Fed-AuditGAN (gamma=0.3, accuracy-focused)"
    echo "  [5] MNIST - IID - Fed-AuditGAN (gamma=0.5, balanced)"
    echo "  [6] MNIST - IID - Fed-AuditGAN (gamma=0.7, fairness-focused)"
    echo ""
    echo "CIFAR-10 Experiments:"
    echo "  [7] CIFAR-10 - IID - Standard FedAvg"
    echo "  [8] CIFAR-10 - IID - Fed-AuditGAN (balanced)"
    echo ""
    echo "Other:"
    echo "  [Q] Quit"
    echo "  [H] Help / Custom Parameters"
    echo ""
    read -p "Enter your choice: " choice

    # Activate conda environment
    # Note: Users should adjust the environment name if different
    # Common names: fed-audit-gan, fedavg, federated-learning
    eval "$(conda shell.bash hook)"
    if conda env list | grep -q "^fedavg "; then
        conda activate fedavg
    elif conda env list | grep -q "^fed-audit-gan "; then
        conda activate fed-audit-gan
    else
        echo -e "${YELLOW}Warning: Could not find fedavg or fed-audit-gan environment${NC}"
        echo -e "${YELLOW}Make sure to activate your environment manually or create it:${NC}"
        echo -e "${YELLOW}  conda env create -f environment.yml${NC}"
        echo ""
    fi

    case $choice in
        1)
            echo -e "${GREEN}Running MNIST - Standard FedAvg - quick test...${NC}"
            python fed_audit_gan.py --dataset mnist --n_clients 3 --n_epochs 2 --n_client_epochs 1 --batch_size 32 --device cpu --exp_name "MNIST_FedAvg_test"
            break
            ;;
        2)
            echo -e "${GREEN}Running MNIST - Fed-AuditGAN gamma=0.5 - quick test...${NC}"
            python fed_audit_gan.py --dataset mnist --use_audit_gan --gamma 0.5 --n_clients 3 --n_epochs 2 --n_client_epochs 1 --n_audit_steps 3 --n_probes 100 --batch_size 32 --device cpu --exp_name "MNIST_AuditGAN_test"
            break
            ;;
        3)
            echo -e "${GREEN}Running MNIST - IID - Standard FedAvg - 50 rounds...${NC}"
            python fed_audit_gan.py --dataset mnist --partition_mode iid --n_epochs 50 --exp_name "MNIST_IID_FedAvg"
            break
            ;;
        4)
            echo -e "${GREEN}Running MNIST - IID - Fed-AuditGAN - gamma=0.3...${NC}"
            python fed_audit_gan.py --dataset mnist --partition_mode iid --use_audit_gan --gamma 0.3 --n_epochs 50 --wandb --exp_name "MNIST_IID_AuditGAN_g03"
            break
            ;;
        5)
            echo -e "${GREEN}Running MNIST - IID - Fed-AuditGAN - gamma=0.5...${NC}"
            python fed_audit_gan.py --dataset mnist --partition_mode iid --use_audit_gan --gamma 0.5 --n_epochs 50 --wandb --exp_name "MNIST_IID_AuditGAN_g05"
            break
            ;;
        6)
            echo -e "${GREEN}Running MNIST - IID - Fed-AuditGAN - gamma=0.7...${NC}"
            python fed_audit_gan.py --dataset mnist --partition_mode iid --use_audit_gan --gamma 0.7 --n_epochs 50 --wandb --exp_name "MNIST_IID_AuditGAN_g07"
            break
            ;;
        7)
            echo -e "${GREEN}Running CIFAR-10 - IID - Standard FedAvg...${NC}"
            python fed_audit_gan.py --dataset cifar10 --partition_mode iid --n_epochs 60 --exp_name "CIFAR10_IID_FedAvg"
            break
            ;;
        8)
            echo -e "${GREEN}Running CIFAR-10 - IID - Fed-AuditGAN - balanced...${NC}"
            python fed_audit_gan.py --dataset cifar10 --partition_mode iid --use_audit_gan --gamma 0.5 --n_epochs 60 --wandb --exp_name "CIFAR10_IID_AuditGAN"
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
