B.Tech. Major Project (CS499) – Weekly Progress Report 5
Student Details:
Name: Vicky Prasad and Shivansh
Roll No.: 22CSE1040 and 22CSE1034
Supervisor's Name: Dr. Pravati Swain
Reporting Period: [13 September 2025 – 19 September 2025]
1. Progress Summary*
The problem statement can be articulated as follows: the primary objective is to address the existing gap between frameworks that excel in accuracy and those that focus on minimizing communication delay, with the aim of developing a novel approach that achieves advancements in both aspects simultaneously. During our review, we came across an additional study that aligns with this perspective. The paper explores the use of Hierarchical Knowledge Transfer (HKT), a technique designed to manage heterogeneous models more effectively. By leveraging logits, or soft labels, the method not only facilitates knowledge sharing across diverse clients but also helps reduce communication overhead and associated time delays. This demonstrates a promising direction for designing federated learning systems that do not sacrifice one essential property for the other, but instead move toward a balanced and optimized solution.

2. Tasks Completed This Week*

# A Hierarchical Knowledge Transfer Framework for Heterogeneous Federated Learning (Yongheng Deng, Ju Ren1, Cheng Tang, Feng Lyu, Yang Liu, Yaoxue Zhang)

Abstract -- Federated learning (FL) enables distributed clients to collaboratively learn a shared model while keeping their raw data private. To mitigate the system heterogeneity issues of FL and overcome the resource constraints of clients, we investigate a novel paradigm in which heterogeneous clients learn uniquely designed models with different architectures, and transfer knowledge to the server to train a larger server model that in turn helps to enhance client models. For efficient knowledge transfer between client models and server model, we propose FedHKT, a Hierarchical Knowledge Transfer framework for FL. The main idea of FedHKT is to allow clients with similar data distributions to collaboratively learn to specialize in certain classes, then the specialized knowledge of clients is aggregated to a super knowledge covering all specialties to train the server model, and finally the server model knowledge is distilled to client models. Specifically, we tailor a hybrid knowledge transfer mechanism for FedHKT, where the model parameters based and knowledge distillation (KD) based methods are respectively used for client-edge and edge-cloud knowledge transfer, which can harness the pros and evade the cons of these two approaches in learning performance and resource efficiency. Besides, to efficiently aggregate knowledge for conducive server model training, we propose a weighted ensemble distillation scheme with serverassisted knowledge selection, which aggregates knowledge by its prediction confidence, selects qualified knowledge during server model training, and uses selected knowledge to help improve client models. Extensive experiments demonstrate the superior performance of FedHKT compared to state-of-the-art baselines.




Current Drawbacks

The FedHKT framework provides a novel way to handle heterogeneous federated learning by relying on soft-label knowledge transfer via a public unlabeled dataset. However, it faces several significant drawbacks. First, the reliance on a large, high-quality public dataset is both impractical and inefficient; maintaining and distributing such datasets across edge servers introduces heavy storage and communication overhead. Second, since the public dataset is unlabeled, the server struggles to measure the reliability of client predictions, which complicates efficient aggregation. Third, client knowledge is often of low quality, due to limited computational capacity and skewed data distributions at the edge. This leads to disparate prediction reliability across classes, making the server’s ensemble distillation less effective. Finally, the framework lacks adaptability, as the fixed public dataset may not adequately cover evolving data distributions or underrepresented categories, reducing the global model’s robustness over time.

Possible use case for GenAI

Generative AI (GenAI) offers a powerful solution to fill these gaps by replacing static public datasets with dynamic synthetic data generation. Instead of distributing heavy datasets, the server can maintain or distribute a generative model (e.g., diffusion models, GANs, or large language models) that synthesizes diverse, domain-specific samples on demand. This approach eliminates storage and communication bottlenecks while ensuring broader coverage of underrepresented classes. Moreover, the server can adaptively generate synthetic samples tailored to each client’s weaknesses, thereby reducing label skew effects. GenAI can also provide semantic embeddings or pseudo-data that are lightweight yet effective for cross-client knowledge transfer. By combining synthetic sample generation with soft-label distillation, the FedHKT framework becomes more scalable, privacy-preserving, and resilient to evolving client data distributions. In short, integrating GenAI enhances both the efficiency and adaptability of knowledge transfer, significantly improving federated learning performance in heterogeneous environments.



3. Plan for the Upcoming Week*

As we approach the completion of our literature review, it becomes increasingly clear that a significant gap exists in the current body of research on Federated Learning. Most existing studies tend to analyze communication delay and accuracy in isolation, treating them as distinct challenges that require separate methods of improvement. While this approach has yielded useful insights, it overlooks the complex interdependence between the two factors. For instance, strategies aimed at reducing communication delay may inadvertently compromise accuracy, while methods that prioritize accuracy often intensify communication costs. This disconnect highlights the need for a more holistic perspective that regards these dimensions not as isolated trade-offs, but as intertwined challenges that can and should be addressed together. By pursuing approaches that simultaneously optimize communication efficiency and model accuracy, we can move toward innovative solutions that offer practical benefits for large-scale, real-world applications of Federated Learning.
4. Additional Notes*
N/A





Student’s Signature: ___________________
Supervisor’s Remarks: The research progress report is satisfactory / not-satisfactory (if not satisfactory, specific reasons must be furnished separately)



Supervisor’s Signature: ___________________
