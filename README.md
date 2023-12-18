# Courses I found interesting : 
- [https://developers.google.com/machine-learning/crash-course/ml-intro](url)
- 


# ML-Interview-preparation
These are the tasks I completed and I revised to prepare for ML interview. I researched about the ML questions that are normally asked in an interview in Bangladesh.

Basic Software and problem solving skills : 
1. OOP(Must)
2. Problem Solving (Basics).
3. Recursion.
4. Sorting algorithms.(Good to have knowledge on these)
5. BFS,DFS,DP (Basics, Not required for most of the cases).
6. python list comprehension(Necessary), decorators. (Is not required for most of the cases)
7. Linked-List.(Usually necessary)
8. pointers and how they work ( Not always required).

After the above then you can have good knowledge on the below : 

Bias : Jokhon model underfit obosthay thake. Mane jokhon training error and validation error same pattern follow kore.

Variance : Jokhon model overfit obosthay thake. Mane jokhon training error and validation error different pattern follow kore.

Regularization : Regularization use kora hoe jate model overfit or underfit na kore.

Scaling : 
1.	scaling kora hole training time reduce hoe karon gradient quickly converge kore.
2.	Scaling kora hole ekta specific feature er proti unnecessary importance reduce hoe. Mane emon suppose ekta dataset e normally feature gular range (0-1) er moddhe. Ekhon jodi emon ekta feature thake jetar range -200,200. Tahole shavabik model etake beshi importance deoar chance thakbe although eta important nao hote pare. Scaling kora hole emon scenario reduce hoe.
3.	Standardization performs well when the dataset may contain outliers.
4.	Min-Max Scaler performs well when you know about the features of your data.

Batch Normalization : Jekono ML task e amra initially data scale kore thaki. However, jokhon model train houa start hoe and weights and bias update hote thake tokhon ar normalize kora hoe nah. E khetre bhivinno problem arise korte pare, jemon slow convergence. Ejonne prottek steps e weights gula normalize kora ke “Batch Normalization” bole.

Hypothesis Testing : Dhora jak, ekta assumption kora holo je ekta class er average student marks hocche 70. To ei assumption ta correct jodi hoe taile sheta hobe “Null hypothesis”(Ho). Ar er opposite ta jodi correct hoe sheta hocche “Alternate hypothesis”(Ha)..

Type 1 error : Rejecting Ho when it is actually True.
Type 2 error : Accepting Ho when it is actually False.

ACID : Atomicity, Consistency - Constraints , Isolation- Concurrency, Durability - Backup

Embeddings : embeddings word embeddings hote pare, sentence embeddings o hote pare. 
Word embeddings. Similarity measure korar ways : 
1.	Dot product
2.	Cosine Similarity

Attention : Recap - https://docs.cohere.com/docs/the-attention-mechanism

Neural Network : Recap - https://www.youtube.com/watch?v=MRZVIuos3Pk&list=PLKdU0fuY4OFdFUCFcUp-7VD4bLXr50hgb&index=6

Activation Functions : https://www.v7labs.com/blog/neural-networks-activation-functions
1. Activation function in regression : https://chat.openai.com/share/74f56371-2e61-46d4-adeb-3606f865b9f7
2. Sigmoid Function : For binary classification. Suffers from “vanishing gradient problem”. This can be partially reduced by using “Data Scaling” and “Batch Normalization”.
3. Tanh :  Tan hyperbolic. Output is within [-1,1].
4. ReLU : Computationally efficient.
5. Dying ReLU : jehetu relu negative weight gula ke kokhono activate kortese na,infact zero kore dicche so emon hote pare je most of the weights have become zero, This can create dead neurons.
6. Leaku ReLU : ekta small part negative side e thakbe ,slope. Jetar karone dying relu problem overcome hoe.
7. Softmax :  Used for muliclass classification.

Precision : Algorithm jotogula positive ke identify korse er moddhe kotogula actualy posive.

Precision = TP/(TP + FP)

Recall : Total jotogula positives ase er moddhe algorithm kotogula ke positive bole identify korse.

Recall = TP/(TP + FN)

Average Precision (AP) : The area under precision-recall curve.

ROC: Curve of TPR(True Positive Rate) VS FPR(False Positive Rate). TPR = Recall. FPR = 1 - Precision.

IoU ( Intersection over Union) : 
![bbox](https://github.com/abdullahmoosa/ML-Interview-preparation/assets/67234038/53cf87c4-85b9-4c9a-ba0a-405f0292e0c5)


Average Precision : https://lh5.googleusercontent.com/VaGm6clve2pqkHmP54gqbAJwfQO0lMx7sGAEKRpCAO-sMOqehIsxRazwTtaweIBqmhqsDsyEM2AtfCvwziqIPPVTeWtdiDi55TavEdYsIIdR7F5AxDQSAuMOKsCkgOngp4emWRy5oOCgtAdZj9ppvtMuEN1_sBa0tcup5_pbP7bZUtqnbCWqcbKEtUpqPA

mAP (mean Average Precision ) : bivinno IoU threshold er jonne average precision ber kore shetar average neoa hocche mAP.


Multi-CLass Cross-Entropy :  Used in multiclass problems when the target column is one hot encoded.

Multi Class Sparse Categorical Cross Entropy :  Used when the target column is not one hot encoded

How to Deal with imbalanced dataset : 

1. Oversampling : SMOTE ( Synthetic Minority Oversampling Technique ) .
2. SMOTE hocche randomly ekta minority class example select kore. Then she knn apply kore. Mane k nearest minority neighbor choose kore egula connect kore. Then connection point er moddhe randomly selected ekta point new sample hisebe generate hoe.
3. Assigning higher weights to the minority class inside the model.
4. One class classification method : Another interesting algorithm-level solution is to apply one-class learning or one-class classification(OCC for short) that focuses on the target group, creating a data description. This way we eliminate bias towards any group, as we concentrate only on a single set of objects.OCC can be useful in imbalanced classification problems because it provides techniques for outlier and anomaly detection. It does this by fitting the model on the majority class data (also known as positive examples) and predicting whether new data belong to the majority class or belong to the minority class(also known as negative examples) meaning it’s an outlier/anomaly.

RNN : RNN can store the memory or the hidden state of previous states. RNN is called Recurrent because the information of the current state is passed to the previous state. The information is cycled in the network’s hidden layers. That is because the simplest RNN model has a major drawback, called vanishing gradient problem, which prevents it from being accurate.
In a nutshell, the problem comes from the fact that at each time step during training we are using the same weights to calculate y_t. That multiplication is also done during back-propagation. The further we move backwards, the bigger or smaller our error signal becomes. This means that the network experiences difficulty in memorizing words from far away in the sequence and makes predictions based on only the most recent ones.

LSTM : LSTM contains forget gates. It can store valuable information and get rid of unimportant ones. Also LSTM contains a Gating mechanism which is used to check whether an information will be stored or not. That is why more information can be stored in LSTM and most of it is valuable.

Bernoulli Distribution : eta mane hocche kono ekta binary outcome event e jekono ekta event houar probability ke indicate kore.

Binomial Distribution : dhora jak 5 bar coin flip kora hoise. Next bar coin flip korle ki utte pare eta ei distribution diye bujha jabe.

Poisson Distribution : Poisson distribution deals with the frequency with which an event occurs within a specific interval. Instead of the probability of an event, Poisson distribution requires knowing how often it happens in a particular period or distance. For example, a cricket chirps two times in 7 seconds on average. We can use the Poisson distribution to determine the likelihood of it chirping five times in 15 seconds.

Beta Distribution : Ekta range er moddhe kono ekta event occur korar probability ber korar jonne use kora hoe

Probability Density Function (PDF) : Function ta kono ekta random value ekta certain value er soman houar probability indicate kore.

Cumulative Density Function (CDF): Kono ekta value 0 theke shei value porjonto houar probability ke indicate kore.

Deployment Pattern in ML : 
1. Shadow mode deployment : ML systems run parallel with human inspection/ Previous models. The output of the ML system is not being used for any decisions. This is used to evaluate the effectiveness of ML models in the real world.
2. Canary Deployment : Small fraction of the traffic is used for prediction using ML. Suppose, initially using only 5% and gradually increasing the traffic amount if the algorithm performs as expected.
3. Blue/Green Deployment : Blue system is the old version, Green is the new version. The phone images are passed to the blue version for prediction through a router. Then , configuring the router, you can also send the images to the Green version. Using this method, you can monitor both methods and also rollback to the blue version if something goes south.

Optimizers : Optimizers are used to select the appropriate learning rates for gradient descent.

How CNN works : https://towardsdatascience.com/convolutional-neural-networks-explained-9cc5188c4939

How YOLO works : 
https://www.v7labs.com/blog/yolo-object-detection

Anchor Boxes and why they are important : 
https://blog.roboflow.com/what-is-an-anchor-box/

How YOLov8 works : 
https://blog.roboflow.com/whats-new-in-yolov8/

non-maximum suppression (NMS) : 
Jei bounding box er IoU high thakbe shetay selected hobe.
https://medium.com/mlearning-ai/a-deep-dive-into-non-maximum-suppression-nms-understanding-the-math-behind-object-detection-765ff48392e5

How Gradient Descent works : https://www.analyticsvidhya.com/blog/2020/10/how-does-the-gradient-descent-algorithm-work-in-machine-learning/

Cross Validation , importance and implementation : https://www.analyticsvidhya.com/blog/2021/05/4-ways-to-evaluate-your-machine-learning-model-cross-validation-techniques-with-python-code/

Cross Validation for Time Series Data : https://medium.com/@soumyachess1496/cross-validation-in-time-series-566ae4981ce4

PCA and How it works : https://builtin.com/data-science/step-step-explanation-principal-component-analysis

Clusters and how to choose the right number of ones : https://towardsdatascience.com/10-tips-for-choosing-the-optimal-number-of-clusters-277e93d72d92#

Concating models in ML : https://control.com/technical-articles/combining-two-deep-learning-models/

Git workflow explained with example : https://chat.openai.com/share/d8f2d545-ac8f-4e0e-9589-129729d0b282





