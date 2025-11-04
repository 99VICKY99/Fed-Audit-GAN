"""
Quick Test Script for Fed-AuditGAN
Tests all 4 phases with minimal settings
"""

import subprocess
import sys

def run_test(test_name, command):
    """Run a test command and report results."""
    print(f"\n{'='*70}")
    print(f"Testing: {test_name}")
    print(f"{'='*70}")
    print(f"Command: {command}\n")
    
    result = subprocess.run(
        command,
        shell=True,
        capture_output=False,
        text=True
    )
    
    if result.returncode == 0:
        print(f"\n✅ {test_name} - PASSED")
        return True
    else:
        print(f"\n❌ {test_name} - FAILED (exit code: {result.returncode})")
        return False

def main():
    """Run all tests."""
    print("="*70)
    print("Fed-AuditGAN Test Suite")
    print("="*70)
    
    tests = []
    
    # Test 1: Baseline FedAvg (no Fed-AuditGAN)
    tests.append((
        "Baseline FedAvg (2 rounds)",
        "python fed_audit_gan.py --dataset mnist --n_clients 5 --n_epochs 2 --n_client_epochs 1 --batch_size 32"
    ))
    
    # Test 2: Fed-AuditGAN with gamma=0.5
    tests.append((
        "Fed-AuditGAN (gamma=0.5, 2 rounds)",
        "python fed_audit_gan.py --dataset mnist --use_audit_gan --gamma 0.5 --n_clients 5 --n_epochs 2 --n_client_epochs 1 --n_audit_steps 5 --n_probes 100 --batch_size 32"
    ))
    
    # Test 3: Fed-AuditGAN with gamma=1.0 (pure fairness)
    tests.append((
        "Fed-AuditGAN (gamma=1.0, pure fairness)",
        "python fed_audit_gan.py --dataset mnist --use_audit_gan --gamma 1.0 --n_clients 5 --n_epochs 1 --n_client_epochs 1 --n_audit_steps 5 --n_probes 100 --batch_size 32"
    ))
    
    # Run all tests
    results = []
    for test_name, command in tests:
        passed = run_test(test_name, command)
        results.append((test_name, passed))
    
    # Summary
    print("\n" + "="*70)
    print("Test Summary")
    print("="*70)
    
    for test_name, passed in results:
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{status} - {test_name}")
    
    total = len(results)
    passed_count = sum(1 for _, p in results if p)
    
    print(f"\n{passed_count}/{total} tests passed")
    
    if passed_count == total:
        print("\n🎉 All tests passed! Fed-AuditGAN is working correctly!")
        return 0
    else:
        print(f"\n⚠️  {total - passed_count} test(s) failed. Check the output above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
