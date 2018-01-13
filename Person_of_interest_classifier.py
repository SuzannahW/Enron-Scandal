#!/usr/bin/python

import sys
import pickle
import matplotlib.pyplot
sys.path.append("../tools/")
import numpy as np
import pandas
from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data
from tester import test_classifier

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
# Note:  from_messages = Number of emails sent
#        to_messages = Number of emails received
features_list = ['poi','salary', 'bonus', 'to_messages', 'from_messages', \
                 'from_this_person_to_poi', 'from_poi_to_this_person', \
                 'total_stock_value']

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

### Task 2: Remove outliers
data_dict.pop("TOTAL", 0)
data_dict.pop("THE TRAVEL AGENCY IN THE PARK", 0)
data_dict.pop("LOCKHART EUGENE E", 0)

data = featureFormat(data_dict, features_list)

# Visualise data using a scatter plot. Here the relationship between 
# to_messages and from_poi_to_this_person is visualised with data points 
# coloured depending upon POI status. 
for point in data:
    poi = point[0]
    if poi == 0:
        col = 'k'
    else:
        col = 'r'
    to_messages = point[3]
    from_poi_to_this_person = point[6]
    matplotlib.pyplot.scatter( to_messages, from_poi_to_this_person, c = col )

matplotlib.pyplot.xlabel("to_messages")    
matplotlib.pyplot.ylabel("from_poi_to_this_person")
matplotlib.pyplot.show()

### Task 3: Create new feature(s)

df = pandas.DataFrame.from_records(list(data_dict.values()))
employees = pandas.Series(list(data_dict.keys()))

# set the index of df to be the employees series
df.set_index(employees, inplace=True)

# removing NaN values
df.replace(to_replace='NaN', value=np.nan, inplace=True)

# adding new features to the dataframe
# Note:  from_messages = Number of emails sent
#        to_messages = Number of emails received
df['proportion_to_poi'] = df['from_this_person_to_poi']/df['from_messages']
df['proportion_from_poi'] = df['from_poi_to_this_person']/df['to_messages']

# removing any new NaN values created from the new features
df = df.replace(np.nan, 'NaN', regex=True)

# after you create features, the column names will be your new features
# create a list of column names:
new_features_list = df.columns.values

# create a dictionary from the dataframe
df_dict = df.to_dict('index')

### Store to my_dataset for easy export below. 
my_dataset = df_dict

features_list_2 = ['poi','salary', 'bonus', 'to_messages', 'from_messages',\
                   'total_stock_value', 'proportion_to_poi',\
                   'proportion_from_poi']

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list_2, sort_keys = True)
labels, features = targetFeatureSplit(data)     

# Visualise new features 
for point in data:
    poi = point[0]
    if poi == 0:
        col = 'k'
    else:
        col = 'r'
    to_messages = point[3]
    proportion_from_poi = point[7]
    matplotlib.pyplot.scatter( to_messages, proportion_from_poi, c = col )

matplotlib.pyplot.xlabel("from_messages")
matplotlib.pyplot.ylabel("proportion_to_poi")
matplotlib.pyplot.show()

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectKBest
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import GridSearchCV

scaler = MinMaxScaler()
skb = SelectKBest()

# create the classifiers
gnb = GaussianNB()
dt = DecisionTreeClassifier()
knn = KNeighborsClassifier()
svc = SVC()

# create parameters for classifiers to be used in pipeline
k_range = range(2,7)

params_gnb = {
        'SKB__k' : k_range,
        }

params_dt = {
        'SKB__k' : k_range,
        'algorithm__min_samples_split' : [2, 4, 6, 8, 10, 15, 20, 25, 30],
        'algorithm__criterion' : ['gini', 'entropy'],
        'algorithm__random_state': [42]
        }

params_knn = {
        'SKB__k' : k_range,
        'algorithm__n_neighbors' : range(2, 10),
        'algorithm__weights' : ['uniform', 'distance'],
        'algorithm__algorithm': ['ball_tree', 'kd_tree', 'brute', 'auto']
        }
    
params_svm = {
        'SKB__k' : k_range,
        'algorithm__kernel' : ['rbf', 'linear', 'poly'],
        'algorithm__degree' : range(2, 4),
        'algorithm__C': [0.1, 1, 2, 4, 6, 8, 10], 
        'algorithm__gamma' : [0.01, 0.1, 1, 10.0, 50.0, 100.0]
        }   

### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html
def get_scores(classifier, parameters):
    steps = [('scaling',scaler), ('SKB', skb), ('algorithm', classifier)]
    pipeline = Pipeline(steps)
    sss = StratifiedShuffleSplit(n_splits=10, test_size=0.3, random_state=42)
    sss.get_n_splits(features, labels)
    gs = GridSearchCV(pipeline, parameters, n_jobs = 1, scoring="f1")
    gs.fit(features, labels)
    clf = gs.best_estimator_
    features_selected = [features_list_2[i+1] for i in
                         clf.named_steps['SKB'].get_support(indices=True)]

    feature_score = []
    for i, j in enumerate(clf.named_steps['SKB'].scores_):
        skb_feature = features_list_2[i+1]
        feature_score.append("{}:{}".format(skb_feature, j))
                         
    print '\n', '\n'
    print 'Here are the results for the classifier:', classifier, '\n'
     
    print 'The feature scores from SelectKBest:'
    print feature_score, '\n'
    
    print 'The features selected by SelectKBest:'
    print features_selected, '\n'    
    
    print 'The best parameters:'
    print gs.best_params_, '\n' 
    
    print "Tester Classification report" 
    test_classifier(clf, my_dataset, features_list_2)
    
    dump_classifier_and_data(clf, my_dataset, features_list_2)
    return clf

# run the dataset through each of the four classifiers 
#get_scores(gnb, params_gnb)
#get_scores(dt, params_dt)
#get_scores(knn, params_knn)
get_scores(svc, params_svm) #svc classifier gave best precision & recall scores

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

#dump_classifier_and_data(clf, my_dataset, features_list_2)