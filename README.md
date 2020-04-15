In this project a Machine Learning Model that takes as input the title and the body of  news-oriented articles written in the Greek language and predicts if the article disseminates fake or legit news was built.

Data Collection:
In the Greek language there was no relative dataset so the necessary data had to be gathered first. 500 news-oriented articles covering topics in politics, economy and national defence were handpicked. The label (legit or fake) of each article was carefully decided through careful research on various sources and articles that have already been proven to be either fake or legit were chosen. From the total of 500 articles 50 were fake and 450 legit in order to model the imbalance of the two classes that exist in the real world.

Feature Engineering:
Named-entity recognition (NER) and Part-of-speech tagging(POS) were used to create meaningful features among with features based on punctuation patterns and psycholinguistics. Through this data analysis process intresting pattern emerged. For example, it became clear that geopolitical entities were used much more in fake than in legit news and on the contrary proper nouns were used more in legit than in fake news. 

Methodology:
The code was written in python. Natural Language Processing (NLP) principles were use used for cleaning the data (lemmatization, punctuation handling, case handling etc), extract usefull information from them and convert the textual content into an appropriate for the appliance of Machine Learning algorithms form (text vectorization, n-grams etc).
Several ML Algorithm (Multinomial NB, SVM, Decision Trees, Random Forest, Gradient Boosted Tree, MLP Neural Networks and more) were tested, combined with cross validation and imbalance handling methods.
In the process of performance evaluation the main metric considered was F-score since it is more appropriate for imbalance datasets than other more common metrics such as accuracy.
The best score in an unknow dataset was achieved using Multinomial NB with an F-score of 0.821 classifying correctly 104 out of 112 legit and 11 out of 13 fake news articles. 

Contribution:
The project was created bearing in mind that if it was to be used in production as a news filtering tool for social media companies or news-oriented corporations it could both contribute to the fight against false information dissemination and also create profit for the company using it, since professional journalists could cross check only the articles labeled as fake within a probability margin from the model. It could serve in other words as a first level filter that could save both money and time.  
