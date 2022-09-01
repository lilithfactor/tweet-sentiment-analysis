# `tweet-sentiment-analysis`
![image](https://user-images.githubusercontent.com/77628011/187991242-12fa1db2-0ff4-4b8e-b02d-c42dec34cb13.png)

Sentiment Analysis of Tweets from [Kaggle Twitter Dataset](https://www.kaggle.com/datasets/jp797498e/twitter-entity-sentiment-analysis)

<b>Statement:</b> Given Tweet Content and an Entity, the task is to judge sentiment of Tweet Content about entity. There are 3 classes in this dataset: `Positive`, `Negative` and `Neutral` (messages not relevant to the entity, i.e. Irrelevant) classified as Neutral.

`Data Cleaning | Preprocessing | EDA | Defining NN Architecture | Training | Prediction | Evaluation`

<hr>

## Steps involved in Preprocessing of Raw Data (w/ `regex`)
1. Delete nans
2. Lower Text
3. Remove urls
4. Remove punctuation
5. Remove contractions (`why'd` -> `why would`)
6. Remove mentions (`@user hey` -> `hey`)
7. Remove hashtags (`#sometrend` -> `sometrend`)
8. Remove double spaces
9. Decode emojis
10. Remove stopwords (`how is the weather` -> `weather`)
11. Remove numbers (`my id 882244` -> `my id`)
12. Delete nans
13. [Lemmatize](https://nlp.stanford.edu/IR-book/html/htmledition/stemming-and-lemmatization-1.html) (`the boy's cars are different colors` -> `the boy car be differ color`)
14. Texts vectorized with TF-IDF vectorizer
15. Categorical features one-hot-encoded

<hr>

## Neural-Net Architecture
<pre>
NNSentimentClassifier(
	(softmax): Softmax(dim=1)
	(dropout): Dropout(p=0.2, inplace=False)
	(model): Sequential(
		(0): Linear(in_features=8032, out_features=1000, bias=True)
		(1): ReLU()
		(2): Dropout(p=0.2, inplace=False)
		(3): Linear(in_features=1000, out_features=100, bias=True)
		(4): Tanh()
		(5): Dropout(p=0.2, inplace=False)
		(6): Linear(in_features=100, out_features=1000, bias=True)
		(7): ReLU()
		(8): Dropout(p=0.2, inplace=False)
		(9): Linear(in_features=1000, out_features=10, bias=True)
		(10): ReLU()
		(11): Dropout(p=0.2, inplace=False)
		(12): Linear(in_features=10, out_features=4, bias=True)
	)
)
</pre>

<hr>

## Reports
### `Accuracy of Implemented Neural Network: 94%`
<img width="365" alt="image" src="https://user-images.githubusercontent.com/77628011/187987335-3a6e1a66-87d7-4e7d-abc7-9cd86ba6204b.png">
<pre>
classes = {
	'Irrelevant': 0, 
	'Negative': 1,
 	'Neutral': 2,
 	'Positive': 3
}
</pre>

[Saved NN Model](https://github.com/lilithfactor/tweet-sentiment-analysis/blob/main/net_94acc.pt)

<hr>

## `References`
1. https://www.kaggle.com/code/katearb/sentiment-analysis-in-twitter-93-test-acc
2. https://www.kaggle.com/code/vaishnavi28krishna/twitter-analysis-using-dt-and-rfdtc
3. https://www.kaggle.com/code/parisrohan/text-feature-cleaning-generation-model-building
4. https://www.kaggle.com/code/tanujdhiman/twitter-sentiment-analysis
5. https://www.kaggle.com/code/cameronwatts/bag-of-words-sentiment-analysis-with-keras-task
6. https://www.lexalytics.com/technology/sentiment-analysis/
7. https://towardsdatascience.com/sentiment-analysis-concept-analysis-and-applications-6c94d6f58c17
