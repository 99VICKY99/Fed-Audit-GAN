B.Tech. Major Project (CS499) – Weekly Progress Report 12
Student Details:
Name: Vicky Prasad and Shivansh
Roll No.: 22CSE1040 and 22CSE1034
Supervisor's Name: Dr. Pravati Swain
Reporting Period: [8th October 2025 – 14th November 2025]
1. Progress Summary*
This week, we conducted a series of experiments to investigate the influence of the gamma hyperparameter on model fairness. Specifically, we evaluated gamma values of 0.3, 0.5, and 0.7 under identical datasets and training configurations, monitoring three fairness metrics throughout the training process: Jain’s Fairness Index (JFI), performance variance across groups, and the min–max performance gap between the best- and worst-performing groups. Across all settings, fairness metrics showed consistent improvement over training rounds, indicating the model’s ability to enhance equity over time. However, higher gamma values were associated with slower or less stable convergence dynamics. Future work will focus on stabilizing convergence and implementing system-level scaling and hardening to ensure robustness under larger-scale or more variable conditions.
2. Tasks Completed This Week*
This week, we conducted a controlled set of experiments to examine the impact of the gamma hyperparameter on fairness performance and training stability. Three configurations were tested: γ = 0.3, 0.5, and 0.7, using identical datasets (MNIST), random seeds, client splits, and optimizer settings to ensure comparability. The only varying factor across runs was the gamma value. For each configuration, we recorded metrics after every federated round, including per-group accuracies (used to compute fairness statistics), global accuracy, and training loss. From the per-group results, we derived Jain’s Fairness Index (JFI), variance across group accuracies, and the min–max performance gap (difference between the best- and worst-performing groups).
Key Findings: Across all gamma values, fairness metrics improved steadily during training. JFI consistently increased, indicating enhanced equity across groups, while both variance and min–max gap decreased over time, reflecting reduced disparity between group performances. The progression was most stable for γ = 0.5, which achieved the best balance between fairness improvement and convergence stability. Lower gamma (γ = 0.3) converged more quickly but reached a slightly lower final fairness, whereas higher gamma (γ = 0.7) led to slower, occasionally unstable convergence characterized by training oscillations and brief regressions.
Metric-Level Trends:
JFI: γ = 0.3 improved rapidly but plateaued; γ = 0.5 showed the smoothest and highest final gain; γ = 0.7 improved more erratically.
Variance: γ = 0.5 achieved the lowest final variance, while γ = 0.7 exhibited intermittent spikes.
Min–Max Gap: γ = 0.5 again achieved the smallest gap, confirming stronger group alignment.
Interpretation: These results confirm that fairness-oriented components in the training process effectively reduce inter-group disparities as training progresses. However, gamma serves as a critical tuning parameter: too low limits fairness gains, while too high destabilizes optimization. In this configuration, γ = 0.5 provided the most effective trade-off between fairness promotion and training reliability. Moving forward, addressing the convergence instability observed for higher gamma values will be essential before scaling or production deployment.

The outputs are as follows:


3. Plan for the Upcoming Week*
To address convergence issues at γ = 0.7, we will test:
Learning rate adjustments: step or cosine schedule with lower base LR.
Gamma ramping: start at 0.3 and increase to target over initial rounds.
Stability measures: gradient clipping and fewer local epochs.
We’ll re-run γ = 0.3, 0.5, 0.7 with 3 seeds each, log results to WandB, and plot mean ± std for JFI, variance, and min–max gap over rounds.
 If convergence stabilizes, we’ll run a scaling test with more clients to assess robustness.
4. Additional Notes*

N/A
Student’s Signature: ___________________
Supervisor’s Remarks: The research progress report is satisfactory / not-satisfactory (if not satisfactory, specific reasons must be furnished separately)
__________________________________________________________________________
Supervisor’s Signature: ___________________
