# Toxic-Comments
It is very common to express personal opinions on the Internet.  Some comments can be positive and friendly, however, some comments can be very rude and offensive. 
These toxic comments are those that contain  slurs, identity attacks, insults, threats or other uncomfortable comments. The comment dataset used contains both labeled and unlabeled datasets. The labeled dataset is divided into train set, develop set, and test set. train set and develop set are used to train the model required, and test set is used to make the final experimental predictions. For each of train set, develop set, and test set, there are three different data representations. Raw data contains the raw text of the corresponding comments, that is one comment per line with ID, toxicity label, identities, and comments. Term frequency inverse document frequency(tfidf) data is provided that applied term frequency-inverse document frequency pre-processing to the comments for feature selection. In tfidf data, each comment is represented as a 1000 dimensional feature vector, each dimension corresponding to one  of the 1000 words. Embedding dataset is provided  that mapped each comment to a 384-dimensional embedding computed with a pre-trained language model. Vector captures the meaning of each comment so that  the comments are closely aligned in a 384-dimensional space. For the labeled data, 1 means the comment is toxic, and 0 means the comment is non-toxic. There  are 140000 instances in every train set, 15000 instances in every develop set, and 200000 instances in very unlabeled data set. We will focus on  whether unlabeled data can affect the structure of data classification and whether it can improve classification with drug theory. 

Model                           Accuracy

Logistic Regression Implemented 81%

Bernoulli Naive Bayes(sklearn)  69%

Logistic Regression(sklearn)    84%

Perceptron(sklearn)             75%
