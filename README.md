# Naive-Bayes-Spam-Filter
Naive Bayes is a popular algorithm used in message filtering, including spam detection. The "naive" in Naive Bayes comes from the assumption that features used to describe instances are conditionally independent, given the class label. In the context of message filtering:

Training:

Data Collection: Gather a labeled dataset of messages, where each message is tagged as either spam or not spam (ham).
Preprocessing: Clean and preprocess the text data, removing irrelevant information, and converting text into a format suitable for analysis (e.g., tokenization, stemming).
Feature Extraction: Represent the messages as feature vectors. Commonly used features include the frequency of each word in the message.

Model Training:

Use the labeled dataset to train the Naive Bayes classifier.
The classifier calculates the probability of a message being spam or ham based on the occurrences of words in the message.

Classification:

Given a new, unseen message, the classifier calculates the probability of it being spam or ham.
The message is classified as spam if the probability of being spam is higher than a predefined threshold; otherwise, it is classified as ham.

Evaluation:

Assess the performance of the model using metrics such as accuracy, precision, recall, and F1-score on a separate test dataset.
