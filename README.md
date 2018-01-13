# Enron-Scandal

After the collapse of Enron in 2002 and the following investigation into the widespread corporate fraud at the company, emails and financial information, amongst other information collected as part of the investigation, were entered into the public record to be used for research and academic purposes.

In this project, machine learning has been used to analyse data from the Enron Corpus and to find features within the dataset whose values can highlight a person as a potential ‘person of interest’ or ‘POI’. Importantly, because the Federal Investigation into Enron’s collapse highlighted people of interest, we can use this knowledge along with the corpus to train and test a supervised learning algorithm.

This repository contains a number of files created as part of this project:

1. Person_of_interest_classifier.py
This file contains the python code used to initially load and clean the data (using the NumPy and pandas libraries) and then test a range of classifiers using the scikit-learn package.

2. Enron Submission Free-Response Questions.html
In this document I talk about data, explaining why I chose the features that I did to test the classifiers against, and the different results that I achieved with the different algorithms tested.

3.  - feature_format.py
    - final_project_dataset.pkl
    - tester.py
These files are helper files required to run Person_of_interest_classifier.py
