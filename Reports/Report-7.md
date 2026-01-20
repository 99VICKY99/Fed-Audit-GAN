B.Tech. Major Project (CS499) – Weekly Progress Report 7
Student Details:
Name: Vicky Prasad and Shivansh
Roll No.: 22CSE1040 and 22CSE1034
Supervisor's Name: Dr. Pravati Swain
Reporting Period: [4th October 2025 – 10th October 2025]
1. Progress Summary*
Building upon our earlier exploration of fairness in federated learning, this week we reviewed two additional papers,“Towards Generalization Fairness in Federated Learning” (FedDragon) and “Accelerating Fair Federated Learning: Adaptive Federated Adam (AdaFedAdam), to deepen our understanding of how fairness can be embedded within optimization and generalization frameworks. FedDragon focused on achieving dual fairness, balancing performance both across clients (domain-wise) and within each client (class-wise) through knowledge distillation and dynamic weighting. AdaFedAdam, on the other hand, emphasized adaptive optimization, using gradient-based scaling to ensure fair convergence among heterogeneous clients. Together, these works highlighted that fairness is not merely a post-processing constraint but an integral aspect of model training dynamics. The insights from these papers reinforce our direction toward integrating Generative AI with Federated Learning, where GenAI can act as an adaptive reasoning layer to monitor, explain, and dynamically enforce fairness across evolving client distributions.

Towards Generalization Fairness in Federated Learning
(Mang Ye, Yuhang Chen, Wenke Huang, Hui Cai, and Laizhong Cui)

Abstract -- Federated learning (FL) has emerged as a new paradigm for privacy-preserving collaborative training in mobile computing, where fairness holds paramount importance. Traditional efforts have largely concentrated on ensuring fairness across clients with different data distributions (i.e., domain-wise generalization fairness). However, the aspect of fairness within classes (class-wise generalization fairness) remains largely unexplored. Thus, an important question arises: is it possible to simultaneously achieve domainwise and class-wise generalization fairness? Moreover, current approaches often improve model performance on weaker distributions at the cost of compromising performance on stronger ones, introducing another dimension of unfairness. So, can we boost performance on weaker distributions without compromising that on stronger ones? To this end, we introduce a global classifier with a super logits distillation strategy to achieve comprehensive generalization fairness. The basic idea is to guide local updating by selecting reliable global supervision via a unified global classifier across multiple clients. First, we use these super logits to improve underperforming distributions while preserving the performance of well-performing distributions, promoting domain-wise generalization fairness. Second, we dynamically allocate local distillation loss weights according to the intra-client and inter-client class fairness, accelerating the training of underperforming classes and enhancing class-wise generalization fairness. Comprehensive experiments demonstrate the enhanced fairness and superior performance of our method, highlighting the significance of fairness in mobile computing environments.

Accelerating Fair Federated Learning: Adaptive Federated Adam (LI JU , TIANRU ZHANG , SALMAN TOOR , AND ANDREAS HELLANDER)

Abstract -- Federated learning is a distributed and privacy-preserving approach to train a statistical model collaboratively from decentralized data held by different parties. However, when the datasets are not independent and identically distributed, models trained by naive federated algorithms may be biased towards certain participants, and model performance across participants is non-uniform. This is known as the fairness problem in federated learning. In this paper, we formulate fairness-controlled federated learning as a dynamical multi-objective optimization problem to ensure the fairness and convergence with theoretical guarantee. To solve the problem efficiently, we study the convergence and bias of Adam as the server optimizer in federated learning, and propose Adaptive Federated Adam (AdaFedAdam) to accelerate fair federated learning with alleviated bias. We validated the effectiveness, Pareto optimality and robustness of AdaFedAdam with numerical experiments and show that AdaFedAdam outperforms existing algorithms, providing better convergence and fairness properties of the federated scheme.

2. Tasks Completed This Week*
“Towards Generalization Fairness in Federated Learning” (FedDragon) and “Accelerating Fair Federated Learning: Adaptive Federated Adam (AdaFedAdam).” FedDragon introduced a dual fairness framework that addresses both domain-wise and class-wise generalization fairness through super logits distillation and dynamic loss weighting, while AdaFedAdam approached fairness as a multi-objective optimization problem, using adaptive gradient scaling to reduce bias and improve convergence across heterogeneous clients. These complementary perspectives highlighted fairness as an integral part of model training, not just a post-processing concern. Alongside these readings, we also worked on preparing our presentation, integrating key insights and aligning them with our broader research direction leveraging Generative AI as an adaptive reasoning layer to monitor, explain, and enforce fairness in evolving federated learning environments.

Alongside this, we are also preparing our presentation, which will summarize the entire progress we had throughout the seven weeks.

Presentation Preview:



3. Plan for the Upcoming Week*
Fairness is a vast domain and its implication can be found on various architectural levels, be it data, client or algorithm itself. Hence, we must choose carefully how and where GenAI is to be augmented, as the context would be dependent on the environment the model would be augmented. Another obstacle may be to define the constraints of “Fairness” to a model, definitions and semantics become quite important when ethics come into question. So, our direction might be decided, but the steps are still to be polished into something viable.
4. Additional Notes*

N/A
Student’s Signature: ___________________
Supervisor’s Remarks: The research progress report is satisfactory / not-satisfactory (if not satisfactory, specific reasons must be furnished separately)
__________________________________________________________________________
Supervisor’s Signature: ___________________


| Paper / Method | Core Idea / Approach | Fairness Mechanism | Strengths | Limitations / Gaps | Conceptual Title (“Fairness as …”) |
| --- | --- | --- | --- | --- | --- |
| SRA / RSRA (Fairness-Aware FL with Unreliable Links in IoT) | Reweights client updates by inverse link reliability; RSRA stabilizes aggregation under packet loss. | Statistical reweighting of updates (1/p_i) + fairness objective (q-loss). | Handles unreliable links, improves fairness for weak clients, resource-efficient. | Simulation-only validation; relies on accurate link probability estimates; limited scalability. | Fairness as Reliability Compensation |
| FairFedCS | Enhances fairness via adaptive client selection and balanced sampling based on local loss and contribution. | Client scoring using local model divergence and utility; selects underrepresented clients more often. | Dynamic fairness improvement; better participation diversity; improved global accuracy. | Overhead from frequent client evaluation; assumes reliable communication. | Fairness as Client Selection |
| FairEquityFL | Ensures equity by modifying aggregation weights to minimize loss disparity among clients. | Weighted optimization using fairness-adjusted aggregation coefficients. | Reduces loss variance; theoretically grounded fairness control. | Slower convergence; sensitive to weight-tuning; lacks privacy protection. | Fairness as Equity Optimization |
| FairFed | Promotes group-level fairness by minimizing inter-group performance disparity (e.g., demographic or domain groups). | Group-based loss regularization and reweighted optimization. | Addresses social or demographic fairness; simple to integrate. | Requires known group labels; may reduce accuracy on dominant groups. | Fairness as Group Equity |
| FedDragon (Towards Generalization Fairness in FL) | Dual fairness via domain- and class-wise balancing using knowledge distillation and global logits. | Global classifier + fairness-weighted dual objectives (domain and class). | Strong generalization fairness; communication-efficient variant (FedDragon-Q). | Needs domain labels; privacy leakage risk; no real-world deployment. | Fairness as Dual Generalization |
| AdaFedAdam (Accelerating Fair FL) | Adaptive Adam optimizer that balances convergence speed and fairness via gradient scaling and learning-rate adaptation. | Client-wise adaptive optimization with fairness-aware momentum control. | Fast convergence; balanced updates; optimization-based fairness. | Lacks real-world tests; unstable under extreme heterogeneity; no adversarial defense. | Fairness as Adaptive Optimization |

