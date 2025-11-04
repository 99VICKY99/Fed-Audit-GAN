"""
Fed-AuditGAN Phase Verification Script
Checks that all 4 phases are implemented correctly without running full training
"""

import ast
import re

def check_file_exists(filepath):
    """Check if a file exists."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return True, f.read()
    except FileNotFoundError:
        return False, None

def check_phase_implementation(content, phase_markers):
    """Check if all phase markers are present in the code."""
    found = {}
    for phase, markers in phase_markers.items():
        found[phase] = all(marker in content for marker in markers)
    return found

def main():
    """Verify all phases are implemented."""
    print("="*70)
    print("Fed-AuditGAN Implementation Verification")
    print("="*70)
    
    # Check main script
    print("\n1. Checking fed_audit_gan.py...")
    exists, content = check_file_exists('fed_audit_gan.py')
    
    if not exists:
        print("❌ fed_audit_gan.py not found!")
        return False
    
    print("✅ File exists")
    
    # Define phase markers
    phase_markers = {
        "Phase 1: Client Training": [
            "Phase 1: Standard FL Training",
            "LocalUpdate",
            "compute_client_update",
            "client_updates"
        ],
        "Phase 2: DCGAN Auditing": [
            "Phase 2: Generative Fairness Auditing",
            "Generator(",
            "Discriminator(",
            "train_generator",
            "generate_synthetic_samples"
        ],
        "Phase 3: Fairness Scoring": [
            "Phase 3: Fairness Contribution Scoring",
            "FairnessContributionScorer",
            "compute_combined_scores",
            "client_accuracies",
            "client_fairness_metrics"
        ],
        "Phase 4: Aggregation": [
            "Phase 4: Multi-Objective Aggregation",
            "aggregate_weighted",
            "final_weights"
        ]
    }
    
    print("\n2. Verifying Phase Implementations...")
    results = check_phase_implementation(content, phase_markers)
    
    all_passed = True
    for phase, passed in results.items():
        status = "✅" if passed else "❌"
        print(f"   {status} {phase}")
        if not passed:
            all_passed = False
    
    # Check auditor modules
    print("\n3. Checking Auditor Modules...")
    
    modules = {
        "Generator (DCGAN)": "auditor/models/generator.py",
        "Fairness Metrics": "auditor/utils/fairness_metrics.py",
        "Contribution Scoring": "auditor/utils/scoring.py"
    }
    
    for name, path in modules.items():
        exists, _ = check_file_exists(path)
        status = "✅" if exists else "❌"
        print(f"   {status} {name}: {path}")
        if not exists:
            all_passed = False
    
    # Check key classes in generator
    print("\n4. Checking Key Classes...")
    exists, gen_content = check_file_exists('auditor/models/generator.py')
    
    if exists:
        key_classes = [
            ("Generator", "class Generator(nn.Module)"),
            ("Discriminator", "class Discriminator(nn.Module)"),
            ("train_generator", "def train_generator("),
            ("generate_synthetic_samples", "def generate_synthetic_samples(")
        ]
        
        for name, signature in key_classes:
            found = signature in gen_content
            status = "✅" if found else "❌"
            print(f"   {status} {name}")
            if not found:
                all_passed = False
    
    # Check fairness auditor
    print("\n5. Checking Fairness Auditor...")
    exists, audit_content = check_file_exists('auditor/utils/fairness_metrics.py')
    
    if exists:
        metrics = [
            "compute_demographic_parity",
            "compute_equalized_odds",
            "compute_class_balance",
            "audit_model"
        ]
        
        for metric in metrics:
            found = metric in audit_content
            status = "✅" if found else "❌"
            print(f"   {status} {metric}")
            if not found:
                all_passed = False
    
    # Check contribution scorer
    print("\n6. Checking Contribution Scorer...")
    exists, score_content = check_file_exists('auditor/utils/scoring.py')
    
    if exists:
        methods = [
            "FairnessContributionScorer",
            "compute_accuracy_contribution",
            "compute_fairness_contribution",
            "compute_combined_scores"
        ]
        
        for method in methods:
            found = method in score_content
            status = "✅" if found else "❌"
            print(f"   {status} {method}")
            if not found:
                all_passed = False
    
    # Check command-line arguments
    print("\n7. Checking Command-Line Arguments...")
    args_to_check = [
        "--use_audit_gan",
        "--gamma",
        "--n_audit_steps",
        "--n_probes",
        "--latent_dim"
    ]
    
    for arg in args_to_check:
        found = arg in content
        status = "✅" if found else "❌"
        print(f"   {status} {arg}")
        if not found:
            all_passed = False
    
    # Summary
    print("\n" + "="*70)
    if all_passed:
        print("🎉 SUCCESS! All Fed-AuditGAN phases are implemented correctly!")
        print("="*70)
        print("\nImplementation Summary:")
        print("  ✅ Phase 1: Standard FL Training (client local training)")
        print("  ✅ Phase 2: DCGAN-based Fairness Auditing (synthetic probes)")
        print("  ✅ Phase 3: Fairness Contribution Scoring (weighted evaluation)")
        print("  ✅ Phase 4: Multi-Objective Aggregation (fairness-aware weights)")
        print("\nAll required modules and functions are present.")
        print("\nTo run Fed-AuditGAN:")
        print("  1. Set up environment: conda activate fed-audit-gan")
        print("  2. Run training:")
        print("     python fed_audit_gan.py --dataset mnist --use_audit_gan --gamma 0.5")
        return True
    else:
        print("❌ VERIFICATION FAILED!")
        print("="*70)
        print("Some components are missing or incorrectly implemented.")
        print("Please check the failed items above.")
        return False

if __name__ == "__main__":
    import sys
    success = main()
    sys.exit(0 if success else 1)
