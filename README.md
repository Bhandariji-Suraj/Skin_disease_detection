Skin Disease Detection 


Project
Skin Disease Detection Using CNN.


Abstract
Skin diseases affect a vast majority of the world population and its timely diagnosis plays a major role in better treatment. In areas like Africa, where there is a lack of healthy lifestyle, many diseases are prone to occur as well as the areas where climatic conditions are not moderate.
In recent years, Convolutional Neural Networks have demonstrated great success in various image classification and analysis tasks. Hence, Convolutional Neural Networks are used in the medical field for various disease detection and classification for early treatment.
The proposed system uses a large dataset of various types of skin lesions to study and predict the type of skin disease.This paper proposes a very basic and simple method of skin disease detection using CNNs. Its aim is to provide a reliable, cost-effective and automated system to healthcare workers and dermatologists.


Keywords : Dermatitis, Convolutional Neural Networks, Skin disease, Support Vector Machine, dermatology, Ensemble Learning.



Introduction

Skin disease is one of the most significant diseases that affects our everyday life. It is very dangerous if not treated on time.Skin diseases can be caused due to several reasons including unhealthy lifestyle, cosmetic products or some biological reasons. 

The early and accurate detection of skin disease is very important, as ignorance can lead to several unavoidable consequences.

In recent years, the application of convolutional neural networks (CNNs) in skin care routine(dermatology) has attracted significant attention by providing a sufficiently large number of solutions for skin disease detection.There are many methods and algorithms which can be used to classify an image as a dermatitis or non-dermatitis. Some commonly used machine learning tools are CNNs, Random Forest, Support Vector Machines, Ensemble Learning etc.

Among all the available methods for skin disease detection, CNN is considered as the best as it works best on image datasets. We can use different methods to enhance our results by applying ensemble learning.




Literature Review

Skin disease detection is an important area of research in the field of dermatology, which aims in providing timely and accurate detection of various types of skin diseases. There has been a vast growing interest in advancements in the field of machine learning and computer technology to enhance the efficiency and accuracy of skin disease detection systems. In this literature survey, our focus is to explore the current state of work done so far and what can be done in future to get best outcomes.



An Introduction to Convolutional Neural Networks[1] - 

This paper basically talks about the recent development in the field of machine learning with the introduction of Convolutional Neural Networks(CNN). CNN is basically used for reading the image data and pattern recognition tasks. This paper also talks about the limitation of Convolutional Neural Networks ie. the computational power of CNN and how it can be solved by avoiding overfitting of dataset and improving predictive performance.


A Method of Skin Disease Detection Using Image Processing And Machine Learning[2] - 

This paper elaborates the method of skin disease detection using image processing and machine learning. Skin diseases are common to all human beings and can be caused by multiple factors such as fungal infection, allergy, or viruses. The expensive advanced technologies for skin disease detection creates a problem for common people, hence, the introduction of image processing was required to solve this problem. The image processing only uses a camera and a computer to study the skin patches of a patient and detect the type of disease caused. 


Survey of Texture Based Feature Extraction For Skin Disease Detection[3] - 

This paper talks about the importance of automated systems which can be used in monitoring the patient, for evaluating the risks for different types of skin diseases using images of their skin lesions. In this paper, the author reviewed different preprocessing methods and classification techniques used by experts to identify whether a skin image suffers from any disease or not. The author explained the use of texture based model which is derived from GLCM(Grey Level Co-occurrence Matrix)
Matrix used for disease detection of skin. In this paper, the author has also compared the performance of Support Vector Machine and K-nearest neighbor classifiers for evaluating the type of disease present in a skin lesion.





Skin Disease Detection model using image processing[4] - 

This paper concentrates on a model for skin disease detection that uses image processing techniques and can be accessed in remote areas also. In this paper, the author used a sample dataset with six common diseases to train its model. Using mobile phones, the patient provides images of the infected area and the model in turn predicts the type of skin disease.



Image based skin disease detection using hybrid neural network coupled Bag-of-Features[5] - 

This paper establishes a model bag-of-features based on NN-NSGA-II, to classify the type of skin disease. In this paper, the author takes only two types of diseases under consideration namely Basel Cell Carcinoma and Skin Angioma. Here, the author has used ANN which is trained using NSGA-II, which has been trained with bag-of -features obtained by implementing SIFT(Scale Invariant Feature Transform) algorithm. Finally, the result has been compared with other methods of ANN. 



Skin disease detection using machine learning[6] - 

This paper discusses the use of convolutional neural networks along with some ensemble learning. In this paper, the proposed model consists of five models such as CNN, VGG16, DenseNet, Inception model and ensemble of VGG16, Inception and DenseNet, for detecting certain types of skin diseases using dermatoscopic images of skin lesions. The model used gives a result with 85.02% accuracy with VGG16, DenseNet and Inception in ensemble method. This paper also discusses some limitations such as limited dataset and presence of more than one disease in one image as it can detect only one disease at one time.


Title
Authors
Methods and models 
Result
An Introduction to Convolutional Neural Networks 
Keiron O’Shea and Ryan Nash
Convolutional Neural Network created with multiple layers such as input layer, convolutional layer, pooling layer, fully-connected layers
An foundation to CNNs 
A Method of Skin Disease Detection Using Image Processing And Machine Learning
Nawal Soliman ALKolifi ALEnezi
Preprocessing, Feature Extraction, and Classification using Support Vector Machine Classifier
On giving the image as input to the model it classify the image into the types of diseases using SVM Classifier
Survey of Texture Based Feature Extraction For Skin Disease Detection
Seema Kolkur, D.R. Kalbande
Grey Level Co-occurrence Matrix, SVM, Neural Network
Texture based classification of input image of skin lesions.
Skin Disease Detection model using image processing
Archana Ajith, Vrinda Goel, Priyanka Vazirani, Dr. M. Mani Roja
Image processing with the help of Discrete Cosine Transform(DCT), Discrete Wavelet Transform (DVT), Singular Value Decomposition(SVD)
Diseases first detected with DCT with 8, 16, 32, and 64 coefficients.
DWT with different levels of decomposition and SVD with different number of singular values
Image based skin disease detection using hybrid neural network coupled Bag-of-Features
Shouvik Chakraborty, Kalyani Mali, Sankhadeep Chatterjee, Sumit Anand, Aavery Basu, Soumen Banerjee, Mitali Das, Abhishek Bhattacharya
Bag-of-features based model with NN-NSGA-II for detecting and classification of skin disease. Scale Invariant Feature Trans-Form method to extract features.
NN-NSGA-II is superior  to NN-PSO and NN-CS as it achieved accuracy of 90.56%, precision of 88.26%, 93.64% recall and 90.87% F-measure.
Skin disease detection using machine learning
Dr. T. Kameswara Rao, P. Chamanthi, N. Tharun Kumar, R. Lakshmi Amulya, M. Uday Sagar
Models used are Convolutional Neural Networks, VGG16, DenseNet, Inception, and ensemble of VGG16, DenseNet, and Inception models.
Accuracy of all models calculated and compared, out of which ensemble model gained highest accuracy. CNN gained accuracy of 74.59%, while VGG16 got 80.33%, DenseNet 82.83%, Inception 80.43% and ensemble model with 85.02% accuracy.


Problem Identification

In the early days, due to lack of knowledge and accurate medical checkups, several people had to suffer from skin diseases which could be cured if treated on time. But due to advancement in technology and introduction of machine learning, it is now easy to detect the type of skin disease and its stage with proper study. Use of CNN, for skin disease detection can help us in accurate detection. Hence, this paper aims to generate a model which can classify the type of skin disease using CNN along with some ensemble learning.
