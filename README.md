# Machine-Learning
CSCI 447 Projects from Montana State Univeristy

Instructor: Dr. John W. Sheppard

Course Objective: To learn why and how different machine learning algorithms operate.

* [Projects Overview](#projects-overview)
* [Project 1: Naive Bayes](#project-1-naive-bayes)
* [Project 2: K Nearest Neighbors](#project-2-k-nearest-neighbor)
* [Project 3: Neural Networks](#project-3-neural-networks)
* [Project 4: Swarm Based](#project-4-swarm-based)
* [Extra Credit: Stacked Autoencoder](#extra-credit-stacked-autoencoder)
* [Contributors](#contributers)

## Projects Overview
Brief description of projects 1-4 and an additional extra credit project.
All of the projects use [Pandas](https://pandas.pydata.org/pandas-docs/stable/) in Python3.

Project 1 uses different data sets than projects 2-4 (and the extra credit). The data sets for projects 2-4 and extra credit consist of classifcation data and regression data as shown below.
* [Abalone](https://archive.ics.uci.edu/ml/datasets/Abalone), Classification
* [Car Evaluation](https://archive.ics.uci.edu/ml/datasets/Car+Evaluation), Classification
* [Image Segmentation](https://archive.ics.uci.edu/ml/datasets/Image+Segmentation), Classification
* [Computer Hardware](https://archive.ics.uci.edu/ml/datasets/Computer+Hardware), Regression
* [Forest Fires](https://archive.ics.uci.edu/ml/datasets/Forest+Fires), Regression
* [Wine Quality](https://archive.ics.uci.edu/ml/datasets/Wine+Quality), Regression

### Project 1: Naive Bayes
[Working Repo](https://github.com/AlexanderHarry/CSCI_447_Machine_Learning.git)


Project 1 is a gentle introduction to machine learning. This project uses the following discrete data sets:
* [Breast Cancer](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+/%28Original/%29)
* [Glass](https://archive.ics.uci.edu/ml/datasets/Glass+Identification)
* [Iris](https://archive.ics.uci.edu/ml/datasets/Iris)
* [Soybean (small)](https://archive.ics.uci.edu/ml/datasets/Soybean+/%28Small/%29)
* [Vote](https://archive.ics.uci.edu/ml/datasets/Congressional+Voting+Records)

These data sets are to be pre-processed to: 
1. Handle missing attributes by removing them from the data,
2. Character data handeling,
3. Split into 80% train data and 20% test data.



#### Naive Bayes
Naive Bayes is the algorithm to be implemented. Using the original data, and randomizing 10% of the data to add noise, Naive Bayes is to be run on both to compare the results. Zero-one loss and log likelihood are used to compare the results of the original data and the data with noise.

K-fold cross validation with a K of 10 is also implemented. 


### Project 2: K Nearest Neighbor
[Working Repo](https://github.com/Jrkeeling23/CSCI-447-Machine-Learning-P2.git)


Project 2 focuses on the different implementations of KNN. 
The data is pre-processed similarily to that of project one. 

#### KNN Algorithms Implemented
1. Traditional k-nearest neighbor
2. Edited k-nearest neighbor
3. Condensed k-nearest neighbor
4. K-means clustering around centroids for reducing the data sets
5. Partition around Medoids (PAM or k-medoids) for reducing the data sets

Comparisons between edited and condensed as well as centroids and medoids are made. All the algorithms are also compared to traditional KNN.


### Project 3: Neural Networks
[Working Repo](https://github.com/Jrkeeling23/CSCI-447-Machine-Learning-P3.git)


The data is pre-processed similarily to that of project 1. 
The key concepts that are to implemented in this project are feed forward neural network and back propagation.
Using 



### Project 4: Swarm Based 
[Working Repo](https://github.com/Jrkeeling23/CSCI-447-Machine-Learning-P4.git)


The data is pre-processed similarily to that of project 1.



### Extra Credit: Stacked Autoencoder
[Working Repo](https://github.com/Jrkeeling23/CSCI-447-Machine-Learning-Extra.git)


The extra credit assignment focuses on a single algorithm; SAE.  
The data is pre-processed similarily to that of project 1, but normalizing the data is implemented for better results.


## Contributers
Justin Keeling
* jrkeeling23@gmail.com
* github username: Jrkeeling23

John Lambrecht
* jclambrecht97@msn.com
* github username: JohnLambrecht1

Alex Harry
* alex.harry1@gmail.com
* github username: AlexanderHarry

Andrew Smith
* a.smith@troutmoon.com
* github username: TheGreatFatCat
