
# Enron Submission Free-Response Questions
## Suzannah Weinfass

## Question 1
After the collapse of Enron in 2002 and the following investigation into the widespread corporate fraud at the company, emails and financial information, amongst other information collected as part of the investigation, were entered into the public record to be used for research and academic purposes.

Machine learning can be used to analyse data and make predictions based on insights gathered from a dataset. In this project, machine learning has been used to analyse data from the Enron Corpus and to find features within the dataset whose values can highlight a person as a potential ‘person of interest’ or ‘POI’. Importantly, because the Federal Investigation into Enron’s collapse highlighted people of interest, we can use this knowledge along with the corpus to train and test a supervised learning algorithm.

Three outliers were removed from the dataset. The financial data input under `'TOTAL'` and the financial data input under `'THE TRAVEL AGENCY IN THE PARK'` were deemed to be outliers because they were not potential POIs. All values under the employee Eugene E Lockhart were `NaN` and so this data was also removed. Due to the small nature of the dataset, which originally included only 1708 datapoints, and low proportion of POIs (18 POIs out of 144 employees), an outlier detection and rejection method, such as one based on residual error, was not employed. Any `NaN` inputs (of which there were many) were replaced with `numpy.nan` when transferring the dictionary of values into a Pandas DataFrame (for easy feature manipulation) and then `numpy.nan` values were converted back to `NaN` before the dataset was transferred back into a Python dictionary.

## Question 2
I started this project by choosing, based on my intuition, what I thought would be the most significant features in determining whether someone was a POI. The chosen features were:
```python
features_list = ['poi','salary', 'bonus', 'to_messages', 'from_messages', 
                 'from_this_person_to_poi', 'from_poi_to_this_person', 'total_stock_value']
```
I then made two new features, `'proportion_to_poi'` and `'proportion_from_poi'`, which represented the proportion of messages to/from a POI out of all messages sent or received by the person in question. This feature weighted an email to a POI from a person who rarely sent emails more highly than one from someone who sent many emails.
```python
df['proportion_to_poi'] = df['from_this_person_to_poi']/df['from_messages']
df['proportion_from_poi'] = df['from_poi_to_this_person']/df['to_messages']
```
Due to the correlation of these new features with the original features `'from_this_person_to_poi'` and `'from_poi_to_this_person'`, these were removed from the final features list, resulting in:
```python
features_list_2 = ['poi','salary', 'bonus', 'to_messages', 'from_messages',
                   'total_stock_value', 'proportion_to_poi', 'proportion_from_poi']
```

I used the SelectKBest feature selection function to select the most relevant features from this list. I allowed the value of `'k'` to be in a range between 2 and 7 (7 being all features included) to ensure a large number of options were available to the GridSearchCV function. The scores obtained for each feature were as follows:
```python
['salary:15.1490411887', 'bonus:17.8573623954', 'to_messages:1.09816846672', 'from_messages:0.241116882334',
'total_stock_value:21.0589950134', 'proportion_to_poi:13.8704061022', 'proportion_from_poi:2.24877901659']
```
It can be seen that the two new features, especially `'proportion_to_poi'`, have a significant score compared to the other features.

Sklearn’s MinMaxScaler function was used to ensure that the features used spanned comparable ranges for the algorithms that required this (SVM and k-nearest neighbours).

## Question 3
In my final analysis I used an SVM algorithm, after using GridSearchCV to also test Naïve Bayes, Decision Tree and a K-Nearest neighbours algorithms. 

Three of the algorithms tested had precision and recall values both above 0.3 and all had an accuracy above 0.8.
The GaussianNB algorithm performed the worst, with a precision value of 0.328 but a recall value of only 0.264. The KNeighborsClassifier gave very similar values for precision and recall (0.333 and 0.334 respectively). In comparison, the DecisionTreeClassifier gave a precision value of 0.335 and a recall value of 0.328 (with random_state = 42). Finally, the SVC algorithm gave the best results, with a precision value of 0.396 and a recall value of 0.446. 

## Question 4
Parameters of an algorithm must be tuned to achieve the best fit whilst avoiding overfitting the data. Tuning of parameters affects things such as the influence of individual training points (for example due to their distance from one another or to a decision boundary), and in turn the influence of outliers, the smoothness of a decision boundary and the shape that a decision boundary can take, and how many times the algorithm should split the data and on what terms it should do this (i.e. to maximise information gain in the case of a decision tree classifier).

In this project I used GridSearchCV to tune the parameters for the algorithms that I tested. This allowed me to test a number of different parameters for each algorithm and to find the optimum set of parameters for my dataset. If done manually, I would have had to systematically change the value of one parameter at a time to test its effect on evaluation metrics such as the F1 score.

Parameters tuned for the `SVC` classifier include the `gamma` and `C` parameters which determine the distance of influence of each training point and the smoothness of the decision boundary respectively. Using GridSearchCV values of `C = [0.1, 1, 2, 4, 6, 8, 10]` and `gamma = [0.01, 0.1, 1, 10.0, 50.0, 100.0]` were tested.

## Question 5
Validation is the process of testing an algorithm on a set of test data, after it has been trained on a separate set of training data. Evaluation metrics, such as the F1 score and precision and recall values, can be used assess the performance of an algorithm on the test dataset. It is important to separate out the dataset into training and testing data before developing an algorithm, to ensure that the algorithm can be tested on data which was not used during the training process. If this rule is not followed, erroneously high evaluation scores will result.

In this project, validation was completed by using stratified shuffle split cross validation which splits the data into training and testing sets. The parameter `test_size` determines the proportion of the data to hold back as test data, the rest used as training data, whilst the parameter `n_splits` determines the number of times this process should be repeated before an average of the test results is reported. In this way, all data is, in the end, used for training the algorithm. This is especially useful when dealing with a small dataset such the one used for this project.

## Question 6
My final SVM algorithm obtained a precision value of 0.396 and a recall value of 0.446.

The precision value indicates that out of all the people identified as POIs, about 40 % of them actually were.  In the case of the recall metric, this value indicates that out of all the people that were in reality labelled as POIs, the algorithm was able to identify about 45 % of them.

_________
##### Footnote
I hereby confirm that this submission is my work. I have cited above the origins of any parts of the submission that were taken from Websites, books, forums, blog posts, github repositories, etc.
Resources referred to in this project submission are:
- A vast number of Udacity discussion forums! e.g. https://discussions.udacity.com/t/errors-when-running-pipeline/357381/9, https://discussions.udacity.com/t/features-selected-by-selectkbest/204361, https://discussions.udacity.com/t/outlier-removal-and-feature-selection/330723/3
- https://stackexchange.com/ 
- https://en.wikipedia.org/wiki/Enron_Corpus
- https://en.wikipedia.org/wiki/Enron
- http://scikit-learn.org/stable/documentation.html
