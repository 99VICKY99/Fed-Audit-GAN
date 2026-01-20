B.Tech. Major Project (CS499) – Weekly Progress Report 9
Student Details:
Name: Vicky Prasad and Shivansh
Roll No.: 22CSE1040 and 22CSE1034
Supervisor's Name: Dr. Pravati Swain
Reporting Period: [19th October 2025 – 24th October 2025]
1. Progress Summary*
As discussed earlier, our current objective was to formulate a precise problem statement that clearly defines the issue we intend to address. Alongside this, we were required to develop a flow diagram that visually illustrates the working mechanism of our proposed solution, outlining how different components interact and how the overall process functions from start to finish. Furthermore, upon finalizing the problem statement, we needed to outline a structured plan for future work, detailing the next steps we will undertake. This includes identifying the specific tasks, methodologies, and milestones that will guide the subsequent phases of our project, ensuring a systematic and goal-oriented progression from problem definition to solution implementation.
2. Tasks Completed This Week*
This week, we tried to focus onto the problem statement

Proposed Solution (Fed-AuditGAN)

This framework introduces an active, generative auditing process to create a more holistic and dynamic measure of client contribution. It uses a GAN not to generate training data, but to act as an intelligent "red team" that actively probes the global model for fairness vulnerabilities.
Phase 1 (Standard FL Round): The server and a selected cohort of clients perform a standard round of federated training. Each client computes a local update, and the server aggregates them to produce a candidate global model for that round.
Phase 2 (Generative Fairness Auditing): The server maintains a separate "auditor" GAN. This GAN is trained adversarially against the current global model. Its goal is not to generate realistic images, but to generate counterfactual fairness probes. These are data points specifically designed to expose the model's biases. For example, the auditor GAN might learn to generate pairs of samples $(x, x')$ that are nearly identical in all task-relevant features but differ only in a sensitive attribute (e.g., two loan applications with identical financial details but different synthesized demographic profiles), for which the current global model produces different predictions. This leverages the adversarial nature of GANs for auditing purposes.
Phase 3 (Fairness Contribution Scoring): The server uses these generated probes to audit the model. It first measures the bias of the global model before applying the clients' updates. Then, for each client's update, it hypothetically applies it and re-measures the bias. A client's "fairness contribution score" is then calculated based on how much their individual update reduces the model's discriminatory behavior on the adversarial probes. Clients whose updates make the model more equitable receive a high fairness score, while those whose updates increase bias receive a low or negative score.
Phase 4 (Multi-Objective Incentive Distribution): The server calculates the final reward for each client based on a weighted combination of their traditional accuracy contribution (e.g., loss reduction) and their newly calculated fairness contribution score. This creates a direct and tangible incentive for clients to provide data that is not just accurate, but also diverse and bias-mitigating.
Novelty and Significance
Fed-AuditGAN reframes contribution fairness from a static, accuracy-focused metric to a dynamic, multi-objective one. It creates the first concrete mechanism to quantify and reward a client's contribution to model fairness. By using a generative model as an adaptive auditor, the system can continuously discover and prioritize the mitigation of the model's most salient biases as it evolves. This encourages the long-term health and equity of the federation by properly valuing clients who provide critical, bias-reducing data.

3. Plan for the Upcoming Week*
Next week, we will begin by setting up a simple federated learning simulation using Python and PyTorch. We will create three to five virtual clients, each with a small portion of a dataset such as the Adult Income dataset, where we will treat gender as the sensitive attribute. A basic logistic regression or small neural network will serve as the global model. Each client will perform a few local training steps and send model updates to the server for aggregation. On the server side, we will implement a simple GAN: the generator will take random noise and produce synthetic data pairs (x,x’)(x, x’)(x,x’) that differ only in the sensitive attribute, while the discriminator will help ensure realism in these probes. We will then pass these generated pairs through the global model to measure prediction differences as a proxy for bias. After each training round, we will reapply individual client updates, re-evaluate bias on these probes, and compute a basic fairness contribution score by comparing pre- and post-update bias metrics (e.g., difference in positive prediction rates between groups). This first prototype will help validate that the training, probing, and scoring components interact correctly in practice.

4. Additional Notes*

N/A
Student’s Signature: ___________________
Supervisor’s Remarks: The research progress report is satisfactory / not-satisfactory (if not satisfactory, specific reasons must be furnished separately)
__________________________________________________________________________
Supervisor’s Signature: ___________________
