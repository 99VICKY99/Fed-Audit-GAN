B.Tech. Major Project (CS499) – Weekly Progress Report 4
Student Details:
Name: Vicky Prasad and Shivansh
Roll No.: 22CSE1040 and 22CSE1034
Supervisor's Name: Dr. Pravati Swain
Reporting Period: [4 September 2025 – 12 September 2025]
1. Progress Summary*
While exploring the different problems with Federated Learning (FL), one of the first major issues we noticed was a drop in model accuracy when the data on each device was different. In real life, data is almost never the same across all users, and this uneven distribution makes it hard for the model to perform well on every device.
Apart from the accuracy problem, we also found a few other challenges that can affect the learning process. One big issue is the increase in communication between devices and the central server. To improve accuracy in non-IID settings, more frequent and detailed communication is often needed. This can lead to slower performance, higher use of resources, and make it difficult to scale the system as more devices are added.
Privacy is another important concern in Federated Learning. Even though the main idea of FL is to keep user data on the device, there are still ways attackers can try to figure out private information by analyzing the model updates. Techniques like model inversion or gradient leakage can sometimes be used to recover sensitive data. To prevent this, strong privacy methods like differential privacy or secure multi-party computation should be used to make sure user data stays safe.

2. Tasks Completed This Week*

We found some papers that might be of interest:

# A Hierarchical Knowledge Transfer Framework for Heterogeneous Federated Learning (Yongheng Deng, Ju Ren, Cheng Tang, Feng Lyu, Yang Liu, Yaoxue Zhang)

Abstract -- Federated learning (FL) enables distributed clients to collaboratively learn a shared model while keeping their raw data private. To mitigate the system heterogeneity issues of FL and overcome the resource constraints of clients, we investigate a novel paradigm in which heterogeneous clients learn uniquely designed models with different architectures, and transfer knowledge to the server to train a larger server model that in turn helps to enhance client models. For efficient knowledge transfer between client models and server model, we propose FedHKT, a Hierarchical Knowledge Transfer framework for FL. The main idea of FedHKT is to allow clients with similar data distributions to collaboratively learn to specialize in certain classes, then the specialized knowledge of clients is aggregated to a super knowledge covering all specialties to train the server model, and finally the server model knowledge is distilled to client models. Specifically, we tailor a hybrid knowledge transfer mechanism for FedHKT, where the model parameters based and knowledge distillation (KD) based methods are respectively used for client-edge and edge-cloud knowledge transfer, which can harness the pros and evade the cons of these two approaches in learning performance and resource efficiency. Besides, to efficiently aggregate knowledge for conducive server model training, we propose a weighted ensemble distillation scheme with serverassisted knowledge selection, which aggregates knowledge by its prediction confidence, selects qualified knowledge during server model training, and uses selected knowledge to help improve client models. Extensive experiments demonstrate the superior performance of FedHKT compared to state-of-the-art baselines.
Communication-Efficient Adaptive Federated Learning (Yujia Wang, Lu Lin, Jinghui Chen)
Abstract -- Federated learning is a machine learning training paradigm that enables clients to jointly train models without sharing their own localized data. However, the implementation of federated learning in practice still faces numerous challenges, such as the large communication overhead due to the repetitive server-client synchronization and the lack of adaptivity by SGD-based model updates. Despite that various methods have been proposed for reducing the communication cost by gradient compression or quantization, and the federated versions of adaptive optimizers such as FedAdam are proposed to add more adaptivity, the current federated learning framework still cannot solve the aforementioned challenges all at once. In this paper, we propose a novel communication-efficient adaptive federated learning method (FedCAMS) with theoretical convergence guarantees. We show that in the nonconvex stochastic optimization setting, our proposed FedCAMS achieves the same convergence rate of O( √ 1 TKm ) as its non-compressed counterparts. Extensive experiments on various benchmarks verify our theoretical analysis.

3. Plan for the Upcoming Week*

Right now, we are planning to design a pipeline that can help improve the accuracy in non-IID settings while also reducing communication overhead. We've explored some promising approaches, such as using frameworks like FedDDPM and the use of Generative AI, which seem helpful for this goal. In addition, techniques like knowledge distillation and other optimization algorithms could help reduce the delay and cost of sharing updates or outputs during training.
What remains now is to carefully choose a set of frameworks and methods that work well together. The goal is to create a balanced system where each part supports the others and helps close the remaining gaps in performance and efficiency.

4. Additional Notes*
N/A





Student’s Signature: ___________________
Supervisor’s Remarks: The research progress report is satisfactory / not-satisfactory (if not satisfactory, specific reasons must be furnished separately)



Supervisor’s Signature: ___________________
