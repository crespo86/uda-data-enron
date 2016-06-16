'''
before I started
Because I am using Python3 I change something
- I used https://github.com/flyinactor91/ud120-projects6 dataset
  (https://discussions.udacity.com/t/course-code-updated-for-python3/31145)
- I change some path to fit my computer (../tools/ -> ./tools)
- using random_state, input sort_keys into feature_format file
'''

import sys
import pickle
import pprint
import numpy as np
import matplotlib.pyplot
import pprint
# pipeline
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectFromModel
# model
from sklearn.grid_search import GridSearchCV
from sklearn.svm import SVC
from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
# validation set
from sklearn.cross_validation import StratifiedShuffleSplit
# feature select
from sklearn.ensemble import ExtraTreesClassifier
# PCA
from sklearn.decomposition import RandomizedPCA
# get score
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn import metrics

sys.path.append("./tools/")
from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data

### purpose of this project: make identifier of POI(Person of interest) in enron dataset

### open dataset
data_dict = pickle.load(open("./final_project/final_project_dataset.pkl", "rb") )

### get information of the data
print("##### information of data #####")
print("number of features : " + str(len(data_dict["PIRO JIM"])))
print("name of features : " + str(list(data_dict["PIRO JIM"])))
print("number of people : " + str(len(data_dict)))
poi_list = []
for a in data_dict:
    if data_dict[a]['poi'] != 0:
        poi_list.append(a)
print("number of POIs : " + str(len(poi_list)))

### adding feature
'''
I made a feature named 'defer_rate_of_restrict_stock' which is restricted_stock_deferred / restricted_stock because I think
poi has more information about company so they might just have few restricted_stock_defered rate. and also it works.
In scatter plot, 17 POI's have almost 0 rate in this plot.
'''
for a in data_dict:
   if data_dict[a]['restricted_stock'] != 'NaN' and data_dict[a]['restricted_stock_deferred'] != 'NaN':
       data_dict[a]['defer_of_restrict_stock'] = (data_dict[a]['restricted_stock'])+(data_dict[a]['restricted_stock_deferred'])
   elif data_dict[a]['restricted_stock'] != 'NaN' and data_dict[a]['restricted_stock_deferred'] == 'NaN':
       data_dict[a]['defer_of_restrict_stock'] = (data_dict[a]['restricted_stock'])
   else:
       data_dict[a]['defer_of_restrict_stock'] = 'NaN'

### cleaning features 1
'''
I tried trim some features(which I assumed that are important)
- Remove 'TOTAL' : total is not a person. so I removed

- salary = MAX : 1111258 MIN : 477 MEAN : 284087.542553 STD : 176186.40936  1IQR : 211802.0  3IQR : 308606.5  MEDIAN : 258741.0
- I tried to remove outliers of salary, but 3 POIs are in the outliers so i cannot removed.
- (the outliers = LAY KENNETH L, FASTOW ANDREW S, SKILLING JEFFREY K)

- bonus = MAX : 8000000 MIN : 70000 MEAN : 1201773.07407 STD : 1432752.54483  1IQR : 425000.0  3IQR : 1200000.0  MEDIAN : 750000.0
- I tried to remove outliers of bouns, but 4 POIs are in the outliers so I cannot removed.
- (the outliers = BELDEN TIMOTHY N, LAY KENNETH L, DELAINEY DAVID W, SKILLING JEFFREY K)

- exercised_stock_options = MAX : 34348384 MIN : 3285 MEAN : 2959559.25743 STD : 5472156.87668  1IQR : 506765.0  3IQR : 2542813.0  MEDIAN : 1297049.0
- I tried to remove outliers of exercised_stock_options, but 4 POIs are in the outliers so I cannot removed.
- (the outliers = HIRKO JOSEPH, LAY KENNETH L, SKILLING JEFFREY K, RICE KENNETH D)

- from_poi_to_this_person = MAX : 528 MIN : 0 MEAN : 64.8953488372 STD : 86.4720722586  1IQR : 10.0  3IQR : 72.25  MEDIAN : 35.0
- the max number is too high (second is 305) and this people is not in POIs (LAVORATO JOHN J) So I removed.

'''
data_dict.pop('TOTAL', None) # remove 'TOTAL'
data_dict.pop('LAVORATO JOHN J', None) # remove largest 'from_poi_to_this_person'

### Demention Reduction

## make proto feature_list (remove email address because it is not number)

features_list = ['poi',
'bonus',
'deferred_income',
'to_messages',
'defer_of_restrict_stock',
'shared_receipt_with_poi',
'expenses',
'exercised_stock_options',
'long_term_incentive'
]

my_dataset = data_dict
data = featureFormat(my_dataset, features_list, sort_keys = True)

#scaled code
'''
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
data = scaler.fit_transform(data)
'''

labels, features = targetFeatureSplit(data)

## make validation set

'''
Because the data is small, I wanted to use k-fold but to make shuffle.
'''

sss = StratifiedShuffleSplit(labels, test_size=0.3, random_state=42)

for train_index, test_index in sss:
    features_train = [features[ii] for ii in train_index]
    features_test = [features[ii] for ii in test_index]
    labels_train = [labels[ii] for ii in train_index]
    labels_test = [labels[ii] for ii in test_index]

## feature selection
'''
use feature selection, I removed 6 features to get high percentage of accuracy_score which is 0.883720930233
'''

'''
if you want to do feature_selection, use it

# fit an Extra Trees model to the data
clf = tree.DecisionTreeClassifier(random_state=42)
clf.fit(features_train, labels_train)
# display the relative importance of each attribute
print(clf.feature_importances_)
pred = clf.predict(features_test)
print(classification_report(labels_test, pred, target_names=['nopoi', 'poi']))
print(recall_score(pred, labels_test))

importances = clf.feature_importances_
from sklearn.feature_selection import SelectFromModel
model = SelectFromModel(clf, prefit=True)
X_new = model.transform(features_train)
print(X_new[4])
print(features_train[4])

importance_list = []
index = 0
num =[]
for feature in importances:
	if feature >= 0.11:
		importance_list.append(feature)
		num.append(index)
	index += 1

print (importance_list, num)
'''

# PCA
'''
when I use ten components,
accuracy_score is the best ([0.86046511627906974, 0.81395348837209303, 0.79069767441860461, 0.83720930232558144, 0.81395348837209303, 0.86046511627906974, 0.88372093023255816, 0.86046511627906974, 0.88372093023255816, 0.90697674418604646, 0.88372093023255816, 0.88372093023255816])
'''

n_components = 8

pca = RandomizedPCA(n_components=n_components, whiten=True, random_state = 42).fit(features_train)

'''
features_train_pca = pca.transform(features_train)
features_test_pca = pca.transform(features_test)
clf = tree.DecisionTreeClassifier(random_state=42)
clf.fit(features_train_pca, labels_train)
pred = clf.predict(features_test_pca)
print(classification_report(labels_test, pred, target_names=['nopoi', 'poi']))
'''

### Get score

'''
I tried SVC, GaussianNB, DecisionTree, Kneighbors.
Using GridSearchSV, find best parameter and apply it.
'''

print( "======== SVC ========")
svc = SVC(kernel = 'rbf', class_weight = 'balanced')
C= [1, 10, 100, 1000, 10000]
gamma= [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1]
pipe = Pipeline(steps=[('pca', pca), ('SVC', svc)])

clf = GridSearchCV(pipe,
                   dict(SVC__C=C, SVC__gamma=gamma), scoring = 'recall')

clf = clf.fit(features_train, labels_train)
pred = clf.predict(features_test)
print(confusion_matrix(pred, labels_test, labels=range(0,2)))
print(classification_report(labels_test, pred, target_names=['nopoi', 'poi']))
print(clf.best_estimator_)

print( "======== KNeighbors ========")

knn = KNeighborsClassifier()
algorithm= ['auto', 'ball_tree', 'kd_tree', 'brute']
n_neighbors= [1,3,5,7,9,11,13,15,17,19]
leaf_size= [5,10,20,30,40,50,60]
p= [1,2]
weights= ['distance', 'uniform']
pipe = Pipeline(steps=[('pca', pca), ('knn', knn)])

clf = GridSearchCV(pipe,
                   dict(knn__algorithm=algorithm,
                        knn__n_neighbors=n_neighbors,
                        knn__leaf_size=leaf_size,
                        knn__p=p,
                        knn__weights=weights), scoring = 'recall')

clf = clf.fit(features_train, labels_train)
pred = clf.predict(features_test)
print(classification_report(labels_test, pred, target_names=['nopoi', 'poi']))
print(confusion_matrix(pred, labels_test, labels=range(0,2)))
print(clf.best_estimator_)

print( "======== Decision Tree ========")
tree = tree.DecisionTreeClassifier()
criterion= ['gini', 'entropy']
splitter= ['best', 'random']
min_samples_split= [1, 10, 20, 30, 40]
pipe = Pipeline(steps=[('pca', pca), ('tree', tree)])

clf = GridSearchCV(pipe,
                   dict(tree__criterion=criterion,
                        tree__splitter=splitter,
                        tree__min_samples_split=min_samples_split), scoring = 'recall')

clf = clf.fit(features_train, labels_train)
pred = clf.predict(features_test)
print(classification_report(labels_test, pred, target_names=['nopoi', 'poi']))
print(confusion_matrix(pred, labels_test, labels=range(0,2)))
print(clf.best_estimator_)


print( "======== GaussianNB ========")
Gaussian = GaussianNB()
clf = GaussianNB()
clf = Pipeline(steps=[('pca', pca), ('gaussian', clf)])
clf = clf.fit(features_train, labels_train)
pred = clf.predict(features_test)
print(classification_report(labels_test, pred, target_names=['nopoi', 'poi']))
print(confusion_matrix(pred, labels_test, labels=range(0,2)))


dump_classifier_and_data(clf, my_dataset, features_list)
