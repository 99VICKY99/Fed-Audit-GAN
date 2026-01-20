B.Tech. Major Project (CS499) – Weekly Progress Report
Student Details:
Name: Vicky Prasad and Shivansh
Roll No.: 22CSE1040 and 22CSE1034
Supervisor's Name: Dr. Pravati Swain
Reporting Period: [18 August 2025 – 22 August 2025]
1. Progress Summary*
Upon being assigned under Dr. Pravati, she had suggested to explore Federated Learning and Generative AI, how they work, and any method via which both of these domains can be combined to find a better solution for the flaws that currently exist in FL. Hence, we were tasked with collecting the summaries and abstracts of relevant papers. Among the ones reviewed, two were found to closely align with our original objective:

Federated Learning for Diffusion Models (Zihao Peng, Xijun Wang, Member, IEEE, Shengbo Chen*, Member, IEEE, Hong Rao*, Cong Shen, Senior Member, IEEE)

Abstract -- Diffusion models are powerful generative models that can produce highly realistic samples for various tasks . Typically, these models are constructed using centralized, independently and identically distributed (IID) training data. However, in practical scenarios, data is often distributed across multiple clients and frequently manifests non-IID characteristics. Federated Learning (FL) can leverage this distributed data to train diffusion models, but the performance of existing FL methods is unsatisfactory in non-IID scenarios. To address this, we propose FEDDDPM—Federated Learning with Denoising Diffusion Probabilistic Models, which leverages the data generative capability of diffusion models to facilitate model training. In particular, the server uses well-trained local diffusion models uploaded by each client before FL training to generat e auxiliary data that can approximately represent the global data distribution. Following each round of model aggregation, the server further optimizes the global model using the auxiliary dataset to alleviate the impact of heterogeneous data on model performance. We provide a rigorous convergence analysis of FEDDDPM and propose an enhanced algorithm, FEDDDPM+, to reduce training overheads. FEDDDPM+ detects instances of slow model learning and performs a one-shot correction using the auxiliary dataset. Experimental results validate that our proposed algorithms outperform the state-of-the-art FL algorithms on the MNIST, CIFAR10 and CIFAR100 datasets.

Generative AI-Powered Plugin for Robust Federated Learning in Heterogeneous IoT Networks (Youngjoon Lee, Jinu Gong, Joonhyuk Kang, School of Electrical Engineering, KAIST, South Korea, Department of Applied AI, Hansung University, South Korea)

Abstract -- Federated learning enables edge devices to collaboratively train a global model while maintaining data privacy by keeping data localized. However, the Non-IID nature of data distribution across devices often hinders model convergence and reduces performance. In this paper, we propose a novel plugin for federated optimization techniques that approximates NonIID data distributions to IID through generative AI-enhanced data augmentation and balanced sampling strategy. Key idea is to synthesize additional data for underrepresented classes on each edge device, leveraging generative AI to create a more balanced dataset across the FL network. Additionally, a balanced sampling approach at the central server selectively includes only the most IID-like devices, accelerating convergence while maximizing the global model’s performance. Experimental results validate that our approach significantly improves convergence speed and robustness against data imbalance, establishing a flexible, privacy-preserving FL plugin that is applicable even in data-scarce environments.
2. Tasks Completed This Week*
After finding the relevant papers, we had discussed about the findings with our guide, and later she also reviewed a presentation on GenAI. We had shown the papers we had collected and what GenAI is about. Upon review, she recommended to thoroughly look into the workings of GenAI, and explore into questions such as:
How GenAI produces varied and diverse synthetic data, if the data set is limited.
The inner workings of GenAI, though terms such as “LLMs” and “Diffusion Models” are often used in an abstract manner, one should know what they mean beyond layman terms.
GenAI is useful for creating synthetic data for non-IID data of clients, which solves a large problem, but, there may be other drawbacks in FL which can be solved, it might be useful if we also look into these situations as well.
From the two papers mentioned above, our guide also suggested that their methodology may follow a similar pattern, but their inner framework can differ, this difference can be used to find how we can work onto a problem
Identifying more relevant research that explains the "why", "how", and "where" of GenAI integration with FL to ensure its usage is driven by necessity and effectiveness, rather than trend.
3. Plan for the Upcoming Week*
We shall now revisit the relevant papers again, for finding the answers for the above queries, and properly identify a solution to one of the flaws of FL that can be solved using GenAI, and also support the fact that GenAI is actually used for its purpose and not because of its recent popularity. This shall then narrow our problem statement, where we would need to find how we will explore potential solution directions.
4. Additional Notes*

A suggestion by our project guide was to revisit the paper properly and look through the concepts again. Though overwhelming, it helps to better recognize the inner workings and algorithms that come into play, which makes the foundation and understanding of the research better and more confident.








Student’s Signature: ___________________
Supervisor’s Remarks: The research progress report is satisfactory / not-satisfactory (if not satisfactory, specific reasons must be furnished separately)



Supervisor’s Signature: ___________________
