# SENTIMENT-ANALYSIS-WITH-NLP

*COMPANY* : CODTECH IT SOLUTIONS

*NAME* : SUBRATA SAMANTA

*INTERN ID* : CT08DG249

*DOMAIN* : MACHINE LEARNING

*MENTOR* : NEELA SANTOSH

*📌 Project Title: Sentiment Analysis Using Machine Learning*
As part of my internship project in the domain of Natural Language Processing (NLP), I worked on Sentiment Analysis, one of the most common and practical applications of NLP in real-world scenarios such as customer feedback monitoring, social media analysis, and product reviews.

This task involved building a machine learning pipeline to classify the sentiment of text data (positive or negative) using classical NLP techniques and classification models.

*🔍 Project Overview*:
The project began with the import of essential Python libraries like pandas, numpy, matplotlib, seaborn, and scikit-learn modules such as CountVectorizer, TfidfVectorizer, and LogisticRegression. These tools were used for data manipulation, text vectorization, visualization, and model training.

I worked with a text dataset containing labeled sentiments, where each data point included a piece of text and its corresponding sentiment label (positive or negative). The objective was to train a model to predict the sentiment of unseen text.

*🧹 Data Preprocessing*:
Preprocessing was a critical step in this project. The steps included:

Lowercasing the text

Removing special characters, punctuations, and stopwords

Tokenization: Splitting sentences into individual words

Stemming and Lemmatization: Reducing words to their base/root form

These operations cleaned the raw text and made it suitable for machine learning models.

*🧠 Feature Engineering*:
Text data cannot be used directly by machine learning models, so it was converted into numerical format using:

CountVectorizer: Converts text to a matrix of token counts.

TF-IDF (Term Frequency-Inverse Document Frequency): Captures word importance based on frequency and uniqueness across documents.

These vectorized forms were then used as features to train the model.

*🤖 Model Training and Evaluation*:
The primary model used for classification was Logistic Regression, a powerful and interpretable model for binary classification tasks. The dataset was split into training and testing sets using train_test_split(), and the model was trained using the training data.

Evaluation of the model included:

Accuracy Score

Confusion Matrix

Classification Report (Precision, Recall, F1-Score)

The results provided insight into how well the model could distinguish between positive and negative sentiments.

*📊 Visualization*:
I visualized the confusion matrix using seaborn’s heatmap, which made it easy to interpret the correct and incorrect classifications. Additionally, basic EDA (exploratory data analysis) helped understand class distribution and the most frequent words in each class.

*✅ Conclusion*:
This sentiment analysis task gave me hands-on experience with key NLP and machine learning techniques. It helped me understand how raw text is transformed into a structured format, how models learn patterns in language, and how to evaluate performance meaningfully.

By completing this project, I strengthened my knowledge in:

NLP preprocessing

Feature extraction techniques (Bag-of-Words and TF-IDF)

Model training and testing in scikit-learn

Interpreting and visualizing results

#OUTPUT 

<img width="539" height="453" alt="Image" src="https://github.com/user-attachments/assets/6a0d1576-d7ec-4764-bb08-df529028a68e" />

<img width="859" height="545" alt="Image" src="https://github.com/user-attachments/assets/dca27e8c-36d1-4a8f-a51a-1804e30618b3" />

<img width="989" height="279" alt="Image" src="https://github.com/user-attachments/assets/eef74986-f16a-489e-9e88-84dcb444361d" />
