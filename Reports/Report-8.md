B.Tech. Major Project (CS499) – Weekly Progress Report 8
Student Details:
Name: Vicky Prasad and Shivansh
Roll No.: 22CSE1040 and 22CSE1034
Supervisor's Name: Dr. Pravati Swain
Reporting Period: [13th October 2025 – 18th October 2025]
1. Progress Summary*
After delivering our presentation, we took the time to revisit and thoroughly review the research papers we had previously analyzed. During this process, we identified various issues and limitations present in each piece of literature. These problems ranged from methodological shortcomings to gaps in addressing key challenges, particularly around the topic of fairness. Recognizing these concerns, we decided to refine our focus and explore how Generative AI (GenAI) can be leveraged to enhance fairness within Federated Learning systems. Our aim was to investigate innovative ways in which GenAI could mitigate biases, improve data representation, and support more equitable outcomes across decentralized learning environments.

2. Tasks Completed This Week*
This week, we identified a key challenge related to the computation of the Shapley Value in our research. Specifically, we observed that calculating the Shapley Value is highly resource-intensive and computationally expensive, especially in large-scale federated learning systems. This inefficiency poses a barrier to practical implementation and scalability. Based on this insight, we formulated a problem statement to guide our ongoing work and explore more efficient approaches to achieve fair contribution evaluation in federated learning.

Achieving Dynamic Contribution Fairness with Generative Auditing (Fed-AuditGAN)

Problem Definition: Existing mechanisms for contribution fairness typically reward clients based on their impact on the global model's accuracy or loss reduction. This narrow definition of "contribution" creates a perverse incentive. It fails to recognize or reward a client whose data, while perhaps not improving overall accuracy, is critical for improving the model's fairness. For example, a client with a small but demographically diverse dataset might significantly reduce the model's bias against a minority group. Under a standard contribution fairness scheme, this client would be undervalued and poorly incentivized, potentially causing them to leave the federation and making the model even less fair over time.

Proposed Solution (Fed-AuditGAN) This framework introduces an active, generative auditing process to create a more holistic and dynamic measure of client contribution. It uses a GAN not to generate training data, but to act as an intelligent "red team" that actively probes the global model for fairness vulnerabilities.


3. Plan for the Upcoming Week*
Going forward, we plan to systematically identify additional vulnerabilities in our federated learning pipeline and document them thoroughly. To do this, we will develop a comprehensive flow diagram that maps components, data flows, threat surfaces, and mitigation points. Alongside the diagram, we will create a detailed implementation scheme that translates identified fixes into concrete engineering tasks, responsibilities, and success criteria. This scheme will be paired with a realistic timeline and milestone plan to ensure steady progress and accountability from this point onward.

4. Additional Notes*

N/A
Student’s Signature: ___________________
Supervisor’s Remarks: The research progress report is satisfactory / not-satisfactory (if not satisfactory, specific reasons must be furnished separately)
__________________________________________________________________________
Supervisor’s Signature: ___________________