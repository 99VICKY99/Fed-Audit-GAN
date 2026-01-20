B.Tech. Major Project (CS499) – Weekly Progress Report 3
Student Details:
Name: Vicky Prasad and Shivansh
Roll No.: 22CSE1040 and 22CSE1034
Supervisor's Name: Dr. Pravati Swain
Reporting Period: [29 August 2025 – 03 September 2025]
1. Progress Summary*
This time, our goal was not only to work on solving the non-IID (non-Independent and Identically Distributed) data problem, which is one of the major challenges in Federated Learning (FL), but also to explore some of the other important issues that come up in real-world FL systems. While Federated Learning allows multiple devices or clients to train a shared model without sharing raw data, it still faces several practical drawbacks. One key issue we were asked to focus on was the problem of preserving user privacy. Although FL helps in keeping data local, sensitive information can still sometimes be leaked through model updates or gradients. Therefore, we needed to look into privacy-preserving techniques such as differential privacy, secure aggregation, and homomorphic encryption that aim to protect users' data even further.
In addition to privacy, we also tried to understand other potential problems such as communication overhead, client heterogeneity (differences in computing power, data size, or availability of clients), and how to handle dropped or unreliable clients during training.
Apart from our main focus on Federated Learning, we were also asked to briefly explore the latest research trends in Generative AI (GenAI). This included looking at how GenAI models like large language models (LLMs) and generative models for images and audio are being improved and applied in different fields. To support this, we also revised and studied some basic machine learning concepts and terminologies. For example, we reviewed how Convolutional Neural Networks (CNNs) work, as they are a foundational architecture commonly used in computer vision tasks.
Overall, the objective was to get a broader understanding of both the technical challenges in Federated Learning and the current developments in GenAI, while also strengthening our foundation in essential machine learning topics.

2. Tasks Completed This Week*

We found some papers that might be of interest:

# Federated Knowledge Recycling: Privacy-preserving synthetic data sharing (Eugenio Lomurno, Matteo Matteucci)

Abstract -- Federated learning has emerged as a paradigm for collaborative learning, enabling the development of robust models without the need to centralize sensitive data. However, conventional federated learning techniques have privacy and security vulnerabilities due to the exposure of models, parameters or updates, which can be exploited as an attack surface. This paper presents Federated Knowledge Recycling (FedKR), a cross-silo federated learning approach that uses locally generated synthetic data to facilitate collaboration between institutions. FedKR combines advanced data generation techniques with a dynamic aggregation process to provide greater security against privacy attacks than existing methods, significantly reducing the attack surface. Experimental results on generic and medical datasets show that FedKR achieves competitive performance, with an average improvement in accuracy of 4.24% compared to training models from local data, demonstrating particular effectiveness in data scarcity scenarios.
GraphFedMIG: Tackling Class Imbalance in Federated Graph Learning via Mutual Information-Guided Generation (Xinrui Li, Qilin Fan1, Tianfu Wang, Kaiwen Wei, Ke Yu, Xu Zhang)
Abstract -- Federated graph learning (FGL) enables multiple clients to collaboratively train powerful graph neural networks without sharing their private, decentralized graph data. Inherited from generic federated learning, FGL is critically challenged by statistical heterogeneity, where non-IID data distributions across clients can severely impair model performance. A particularly destructive form of this is class imbalance, which causes the global model to become biased towards majority classes and fail at identifying rare but critical events. This issue is exacerbated in FGL, as nodes from a minority class are often surrounded by biased neighborhood information, hindering the learning of expressive embeddings. To grapple with this challenge, we propose GraphFedMIG, a novel FGL framework that reframes the problem as a federated generative data augmentation task. GraphFedMIG employs a hierarchical generative adversarial network where each client trains a local generator to synthesize high-fidelity feature representations. To provide tailored supervision, clients are grouped into clusters, each sharing a dedicated discriminator. Crucially, the framework designs a mutual information-guided mechanism to steer the evolution of these client generators. By calculating each client’s unique informational value, this mechanism corrects the local generator parameters, ensuring that subsequent rounds of mutual information-guided generation are focused on producing high-value, minority-class features. We conduct extensive experiments on four real-world datasets, and the results demonstrate the superiority of the proposed GraphFedMIG compared with other baselines.


3. Plan for the Upcoming Week*

Given the problems we have identified in Federated Learning, along with the existing solutions that researchers have proposed, the next objective is to find a specific problem statement where there is still room for improvement. This means carefully analyzing the current challenges—such as non-IID data, privacy concerns, communication costs, and client variability—and identifying an area where the current solutions are either not efficient enough, not practical, or not fully developed.
Once a suitable problem statement is selected, the next step will be to plan out the implementation process. This involves deciding on the right methods, tools, and frameworks to use, as well as designing the architecture or model that can help solve the problem. We may also need to experiment with different techniques, compare results, and evaluate the performance of our solution against existing benchmarks. The goal is to come up with an approach that is both effective and practical, and which contributes something meaningful to the ongoing research in this field.

4. Additional Notes*
N/A





Student’s Signature: ___________________
Supervisor’s Remarks: The research progress report is satisfactory / not-satisfactory (if not satisfactory, specific reasons must be furnished separately)



Supervisor’s Signature: ___________________
