B.Tech. Major Project (CS499) – Weekly Progress Report 10
Student Details:
Name: Vicky Prasad and Shivansh
Roll No.: 22CSE1040 and 22CSE1034
Supervisor's Name: Dr. Pravati Swain
Reporting Period: [25th October 2025 – 30th October 2025]
1. Progress Summary*
This week, our primary focus was on establishing the development environment for implementing the Fed-AuditGAN framework. As outlined in our previous report, we planned to begin with a simple federated learning simulation using Python and PyTorch. However, we encountered significant challenges during the environment setup phase using Miniconda, which prevented us from proceeding with the actual implementation. Despite our efforts to resolve these issues, we were unable to achieve a stable and fully functional development environment by the end of the reporting period.
2. Tasks Completed This Week*
We began by setting up Miniconda 23.10.0 as our package and environment management system, selecting it for its lightweight nature and ability to manage complex dependencies in machine learning projects. The initial setup involved creating a virtual environment named fed-auditgan-env with Python 3.10 and attempting to install core dependencies, including PyTorch 2.1.0, NumPy, Pandas, and scikit-learn. However, we encountered several issues. The first was a package dependency conflict: PyTorch 2.1.0 (with CUDA 11.8) required a NumPy version incompatible with other dependencies. Attempts to resolve this by manually specifying versions or using pip as an alternative installer only exacerbated compatibility issues with torchvision and torchaudio. We also faced challenges with CUDA installation, where the conda-installed version (11.8) conflicted with the system-level CUDA installation (12.1), and environment variables were not properly set, leading to GPU detection issues in PyTorch. Additionally, inconsistencies in environment activation meant that installed packages were not accessible within Python despite appearing in the conda list, suggesting issues with the environment's site-packages path or executable linking. Further complications arose when trying to install GAN-specific libraries like torch-fidelity and preprocessing tools for the Adult Income dataset, which broke existing configurations due to missing C++ build tools and conflicting pandas version requirements. Extensive troubleshooting efforts, including environment recreation, channel prioritization, and system-level cleanup, proved ineffective. We also briefly explored Docker containerization but lacked the necessary expertise to implement it swiftly. Despite reviewing PyTorch installation guides, conda documentation, and relevant Stack Overflow threads, we were unable to establish a stable, conflict-free environment.

3. Plan for the Upcoming Week*
Given the time invested in environment setup, we’ve revised our strategy for the coming week. Our primary approach is to set up the environment using Python venv with pip, instead of conda, to simplify dependency management. As a backup, we’ll explore Google Colab, which offers pre-configured PyTorch and CUDA environments, ensuring a smoother development process. If both approaches fail, we’ll try another performative device for assistance with system-level configurations. In parallel, we will continue with other tasks, such as finalizing the architectural design of our GAN auditor component, preparing pseudocode for the fairness contribution scoring mechanism, and researching bias metrics like demographic parity and equalized odds. Additionally, we will download and preliminarily examine the structure of the Adult Income dataset.

4. Additional Notes*

N/A
Student’s Signature: ___________________
Supervisor’s Remarks: The research progress report is satisfactory / not-satisfactory (if not satisfactory, specific reasons must be furnished separately)
__________________________________________________________________________
Supervisor’s Signature: ___________________
