# NLP with Machine Learning



An end-to-end Natural Language Processing (NLP) project that evaluates text sentiment and predicts director demographic classification based on movie synopsis metadata.

## Project Objectives

* **Sentiment Analysis:** Compute compound polarity scores for movie synopses to rank the top 10 most positive and negative movies.


* **Text Classification:** Preprocess raw text and build machine learning classifiers to predict whether a movie was directed by a male or female director based on its plot description.



## Tools & Technologies


**Data Handling**  `pandas`<br> 
**NLP & Sentiment**  `vaderSentiment` (`SentimentIntensityAnalyzer`), `maven_text_preprocessing`<br>
**Machine Learning**  `scikit-learn` (`CountVectorizer`, `TfidfVectorizer`, `MultinomialNB`, `LogisticRegression`, `train_test_split`, `accuracy_score`, `classification_report`)

**Environment** | Jupyter Notebook (`nlp_machine_learning` environment)


## Methodology

### Task 1: Sentiment Analysis

1. **Data Ingestion:** Load `movie_reviews.csv` into a Pandas DataFrame.


2. **Scoring:** Compute VADER compound sentiment polarity scores for text in the `movie_info` column using a custom scoring function.


3. **Extraction:** Sort dataset compound scores to isolate the top 10 most negative (`head(10)`) and top 10 most positive (`tail(10)`) movies.



### Task 2: Text Classification

1. **Text Preprocessing & Normalization:** Clean and normalize the `movie_info` text using `maven_text_preprocessing.clean_and_normalize`.


2. **Feature Extraction:** Construct a bag-of-words representation using `CountVectorizer` with English stop words removed and a minimum document frequency (`min_df`) set to `0.1` (10%).


3. **Model Development:** Train Multinomial Naïve Bayes (`MultinomialNB`) and Logistic Regression (`LogisticRegression`) models to classify director gender (`director_gender`).


4. **Model Evaluation:** Evaluate model predictions against test data using accuracy metrics and classification performance reports.


5. **Inference:** Apply the highest-performing model to return the top 5 movies with the highest predicted probability of having a female director.
