# Sentiment-analysis

# Overview
This dataset contains real customer reviews from Amazon, featuring fields such as review text, reviewer name, star ratings, helpful votes, and review dates. It has been enhanced with sentiment polarity scores (positive, neutral, negative, and compound) using the VADER sentiment analyzer.

🔍 Key Features:

Real-world e-commerce review data

Text-based feedback for NLP tasks

Includes both user ratings and calculated sentiment scores

Perfect for projects in sentiment analysis, text classification, and customer feedback analytics.

# Objective and Key features:
To perform sentiment analysis on textual data using Natural Language Processing (NLP) techniques to classify opinions or statements as positive, negative, or neutral.

Key Features:
Text Preprocessing: Includes steps like tokenization, stopword removal, and stemming/lemmatization.

Feature Extraction: Utilizes methods like TF-IDF to convert text into numerical format.

Model Training: Implements machine learning models such as Logistic Regression or Naive Bayes for sentiment classification.

Evaluation Metrics: Assesses model performance using accuracy, precision, recall, and confusion matrix.

Visualization: May include word clouds or charts for understanding sentiment distribution.

# Tools and Technologies
This project leverages a range of modern NLP and data analysis tools to perform accurate sentiment classification:

Hugging Face Transformers
Utilized the pre-trained cardiffnlp/twitter-roberta-base-sentiment model for sentiment analysis, along with AutoTokenizer and AutoModelForSequenceClassification for efficient text encoding and classification.

PyTorch
Served as the deep learning backend to run the transformer model and handle tensor operations.

SciPy
Applied the softmax function from scipy.special to convert model logits into probability distributions for sentiment classes.

Pandas
Used for data handling, structuring model outputs, and displaying results in a tabular format.

Seaborn & Matplotlib
Employed for visualizing sentiment distributions and plotting key insights from the analysis.

# Visualization Queries
What is the distribution of star ratings in the dataset?
![Screenshot 2025-04-18 232100](https://github.com/user-attachments/assets/825ab8b8-631f-4390-8aa9-cb418537b6fb)

How does the compound sentiment score vary across different star ratings?
![image](https://github.com/user-attachments/assets/51b06f44-bcd7-4527-9788-9dfca58cdfb2)

How do positive, neutral, and negative VADER scores vary with star ratings?
![image](https://github.com/user-attachments/assets/1551ee7f-c59e-46c1-925a-235ef4f0381a)

What is the distribution of compound sentiment scores?
![image](https://github.com/user-attachments/assets/de244ef9-b0de-432f-bee6-ddd63185f81d)

How does the RoBERTa model distribute probabilities across sentiment classes?
![image](https://github.com/user-attachments/assets/603c314a-f6f1-449a-943e-83cf99d3a999)

How are sentiment probabilities distributed in a radar format for comparison?
![image](https://github.com/user-attachments/assets/fd2e11f0-f395-4a9f-9db8-9e5800ca6126)







