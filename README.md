Problem Statement: Medical image classification has achieved significant progress with the help of deep learning 
techniques, particularly convolutional neural networks. However, these models rely heavily on 
large-scale annotated datasets for effective training. In real-world medical scenarios, especially 
for rare diseases, such datasets are difficult to obtain due to privacy concerns, limited patient cases, and the high cost of expert annotation. This results in data scarcity, which significantly 
impacts the performance and reliability of traditional machine learning models. 
Another major challenge is class imbalance, where certain diseases have thousands of samples 
while others have very few. This imbalance causes models to become biased toward majority 
classes, leading to poor performance on rare conditions. Additionally, domain variation across 
datasets further reduces model generalization, making it difficult for trained models to perform 
consistently on new or unseen data. 
Few-shot learning has been proposed as a potential solution to address these issues by enabling 
models to learn from a very small number of samples. However, existing few-shot learning 
models still face challenges such as unstable performance, sensitivity to data distribution, and 
difficulty in capturing meaningful feature representations with extremely limited data. 
Therefore, there is a need to develop a robust system that can accurately classify medical images 
with minimal training samples while maintaining good generalisation. This project focuses on 
analysing how model performance degrades as data decreases and identifying strategies to 
improve reliability under such conditions. 
2.3 EXISTING SYSTEM 
Existing medical image classification systems are primarily based on deep learning models such 
as convolutional neural networks, including architectures like ResNet, VGG, and DenseNet. 
These models are typically pretrained on large datasets and then fine-tuned for specific medical 
imaging tasks. While they have demonstrated high accuracy in well-balanced datasets, their 
performance is highly dependent on the availability of large amounts of labelled data. 
In practical medical scenarios, especially for rare diseases, such data availability is limited. As a 
result, these models tend to overfit when trained on small datasets, learning noise instead of 
meaningful patterns. Furthermore, they struggle to generalise across different datasets due to 
variations in imaging conditions, patient demographics, and equipment used for data acquisition. 
Another limitation of existing systems is their inability to handle class imbalance effectively. 
Models trained on imbalanced datasets tend to prioritise majority classes, resulting in poor 
detection of rare conditions. Additionally, traditional deep learning approaches require significant 
computational resources and long training times, making them less suitable for low-resource 
environments. 
Although techniques such as transfer learning and data augmentation have been used to improve 
performance, they still rely on the availability of a reasonably large dataset. Therefore, existing 
systems are not well-suited for scenarios where only a few samples per class are available, 
highlighting the need for alternative approaches.
2.4 PROPOSED SYSTEM 
The proposed system aims to overcome the limitations of traditional medical image classification 
methods by utilising few-shot learning techniques. Few-shot learning enables models to learn from 
a very small number of samples by focusing on extracting meaningful feature representations and 
learning relationships between data points rather than relying on large datasets. 
In this system, two metric-based few-shot learning models are implemented: Prototypical 
Networks and Siamese Networks. Prototypical Networks represent each class using a prototype, 
which is calculated as the mean of feature embeddings, and classify new samples based on their 
distance to these prototypes. Siamese Networks, on the other hand, learn a similarity function that 
compares pairs of images and determines whether they belong to the same class. 
The proposed system also incorporates data augmentation techniques such as rotation, flipping, 
Mixup, and CutMix to increase dataset diversity and improve model generalisation. These techniques help reduce overfitting and enhance the robustness of the model under low-data 
conditions. 
In addition, the system evaluates model performance by gradually reducing the number of training 
samples per class, allowing for the analysis of performance degradation. This approach provides 
insights into how different models behave under extreme data scarcity and helps identify the most 
effective strategies for maintaining accuracy. 
Overall, the proposed system provides a scalable and efficient solution for medical image 
classification in scenarios with limited data, making it highly relevant for real-world healthcare 
applications. 
