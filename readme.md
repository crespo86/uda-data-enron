# Identify Fraud from Enron Dataset

<p style="text-align: right;">Sanghyun Lee</p>

## Goal
In this project I wanted to identify person of interest (POI) of Enron disaster from [Enron dataset](https://www.cs.cmu.edu/~./enron/).

## Dataset Overview
The dataset have 21 features (incloud ‘poi’) and 146 people incloud 18 POIs. The features are this.

### features list

| features   | missed value rate | features   | missed value rate |
|:-----------|------------:|:-----------|------------:|
| deferred_income       |        65% |total_stock_value    |     14% |
| restricted_stock     |      25% |director_fees    |     88% |
| total_payments       |        14% |  other    |     36% |
| loan_advances       |       97% |   from_poi_to_this_person    |     41% |
| email_address    |     24$ |  restricted_stock_deferred    |     88% |
| bonus    |     44% |  exercised_stock_options    |     30% |  
| poi    |     0% |  deferral_payments    |     73% |
| salary    |     35% |  expenses    |     35% |  
| from_this_person_to_poi    |     41% |  from_messages    |     41% |
| to_messages    |     41% |  long_term_incentive    |     55% |
| shared_receipt_with_poi    |     41% |||


'email_address' is not number and I will identify 'poi', so
I can use 19 features of this dataset.

Every features which I can use have _missed value mentioned 'NaN'_. I will change 'NaN' to zero(0) to figure out.

## Data Cleaning

I tried remove outliers of some features.
I choose salary, bonus, exercised_stock_options, from_poi_to_this_person, and defer_rate_of_restrict_stock.

### 1.  remove a key named ‘TOTAL’ . It is not a person’s name

Before I removed 'Total':

<img src="https://github.com/crespo86/uda-ml/raw/master/uda_figure_1-1.png" alt="Drawing" style="width: 200px;"/>

I checked the red point and get the red is 'Total'.

After I removed 'Total':

<img src="https://github.com/crespo86/uda-ml/raw/master/uda_figure_1-2.png" alt="Drawing" style="width: 200px;"/>

It is better

### 2.  salary

| MAX | MIN | MEAN | STD | 1IQR | MEDIAN | 3IQR |
|:----|:----|:----|:----|:----|:----|:----|:----|
| 1111258 | 477 | 284087.54 | 176186.41 | 211802 | 258741 | 308606.5 |

I tried to remove outliers of salary, but 3 POIs are in the outliers so i cannot removed.
 - (LAY KENNETH L, FASTOW ANDREW S, SKILLING JEFFREY K)


### 3. bonus
| MAX | MIN | MEAN | STD | 1IQR | MEDIAN | 3IQR |
|:----|:----|:----|:----|:----|:----|:----|:----|
| 8000000 | 70000 | 1201773.07 | 1432752.54 | 425000 | 750000 | 1200000 |
I tried to remove outliers of bouns, but 4 POIs are in the outliers so I cannot removed.
-  (BELDEN TIMOTHY N, LAY KENNETH L, DELAINEY DAVID W, SKILLING JEFFREY K)

### 4. exercised_stock_options
| MAX | MIN | MEAN | STD | 1IQR | MEDIAN | 3IQR |
|:----|:----|:----|:----|:----|:----|:----|:----|
| 34348384 | 3285 | 2959559.26 | 5472156.88 | 506765 | 1297049 | 2542813 |
I tried to remove outliers of exercised_stock_options, but 4 POIs are in the outliers so I cannot removed.
-  (HIRKO JOSEPH, LAY KENNETH L, SKILLING JEFFREY K, RICE KENNETH D)

### 5. from_poi_to_this_person
| MAX | MIN | MEAN | STD | 1IQR | MEDIAN | 3IQR |
|:----|:----|:----|:----|:----|:----|:----|:----|
| 528 | 0 | 64.90 | 86.47 | 10 | 72.25 | 35 |
the max number is too high (second is 305) and this people is not in POIs (LAVORATO JOHN J) so I removed.

Before I removed 'LAVARATO JOHN J':

<img src="https://github.com/crespo86/uda-ml/raw/master/uda_figure_1-3.png" alt="Drawing" style="width: 200px;"/>

After I removed 'LAVARATO JOHN J':

<img src="https://github.com/crespo86/uda-ml/raw/master/uda_figure_1-4.png" alt="Drawing" style="width: 200px;"/>

# Feature selection

## Adding data
Before I started, I added a feature named **'defer_rate_of_restrict_stock'** . This feature is calculated like _restricted_stock_deferred  +  restricted_stock_. I thought POIs have more information about company so they might just have few restricted_stock_defered.

the Code is like this

``` python
for a in data_dict:
   if data_dict[a]['restricted_stock'] != 'NaN' and data_dict[a]['restricted_stock_deferred'] != 'NaN':
       data_dict[a]['defer_of_restrict_stock'] = (data_dict[a]['restricted_stock'])+(data_dict[a]['restricted_stock_deferred'])
   elif data_dict[a]['restricted_stock'] != 'NaN' and data_dict[a]['restricted_stock_deferred'] == 'NaN':
       data_dict[a]['defer_of_restrict_stock'] = (data_dict[a]['restricted_stock'])
   else:
       data_dict[a]['defer_of_restrict_stock'] = 'NaN'
```
- 'restricted_stock_deferred' is normally minus, so I make plus.
- Also, if 'restricted_stock_deferred' is 'NaN', it means 0, so In some case which is 'restricted_stock' is a number and 'restricted_stock_deferred' is 'NaN' I use just 'restricted_stock' because a number - 0 = a number.

## Find Important Features

to check the effectiveness of features, I used Validation Set and
**DecisionTreeClassifier**
- the data is small, I used **StratifiedShuffleSplit** which is shuffled k-fold.
- StratifiedShuffleSplit is one of cross validation iterator method. 
- First, the data is so small I had to choose shuffled validation set then k-fold.
- second, I had to choose one between StratifiedKFold and StratifiedShuffleSplit.
- third, the data is unbalanced and It can have some overlaped data. so I choose StratifiedShuffleSplit.
- (StratifiedShuffleSplit is merge of StratifiedKFold and ShuffleSplit.)

Here is the validation code

``` python
sss = StratifiedShuffleSplit(labels, test_size=0.3, random_state=42)

for train_index, test_index in sss:
    features_train = [features[ii] for ii in train_index]
    features_test = [features[ii] for ii in test_index]
    labels_train = [labels[ii] for ii in train_index]
    labels_test = [labels[ii] for ii in test_index]
```

And this is the code to get precision and recall score

``` python
clf = tree.DecisionTreeClassifier(random_state=42)
clf.fit(features_train, labels_train)
print(clf.feature_importances_)
pred = clf.predict(features_test)
print(classification_report(labels_test, pred, target_names=['nopoi', 'poi']))
```

### First, I checked my new feature

Before I put my new Feature

|| precision   | recall  | f1-score   | support |
|:--|-----------:|------------:|-----------:|------------:|
|nopoi| 0.92       |        0.92 |0.92    |     38 |
|poi| 0.40       |        0.40 |0.40    |     5 |
|avg / total| 0.86       |        0.86 |0.86    |     43 |

After I put my new Feature

|| precision   | recall  | f1-score   | support |
|:--|-----------:|------------:|-----------:|------------:|
|nopoi| 0.92       |        0.95 |0.94    |     38 |
|poi| 0.50       |        0.40 |0.44    |     5 |
|avg / total| 0.87       |        0.88 |0.88    |     43 |                           

Both precision, recall score, and f1-score of poi are increased

### Second, discard irrelevant features   

to discard irrelevant features, used **clf.feature_importances_** and **SelectFromModel**

importance list

| Features   | Importance  | Features   | Importance |
|:-----------|------------:|:-----------|------------:|
| bonus      |       25.4% |restricted_stock   |     0.0% |
| total_payments      |       13.3% |restricted_stock_deferred    |     0.0% |
| deferred_income      |       12.2% |exercised_stock_options    |     0.0% |
| to_messages      |       11.0% |restricted_stock    |     0.0% |
| defer_of_restrict_stock      |       9.1% |loan_advances    |     0.0% |
| total_stock_value      |       7.9% |deferral_payments    |     0.0% |
| shared_receipt_with_poi      |       7.7% |salary    |     0.0% |
| expenses      |       7.6% |from_this_person_to_poi    |     0.0% |
| from_messages      |       5.9% |long_term_incentive    |     0.0% |
| director_fees      |       0.0% |other    |     0.0% |

I used SelectFromModel to get important Features

the features are this

``` python
features_list = ['poi', 'bonus', 'deferred_income', 'to_messages',
                 'defer_of_restrict_stock', 'shared_receipt_with_poi',
                 'expenses', 'exercised_stock_options', 'long_term_incentive']
```
The precision and recall score to use this features_list are below

|| precision   | recall  | f1-score   | support |
|:--|-----------:|------------:|-----------:|------------:|
|nopoi| 0.92       |        0.95 |0.93    |     37 |
|poi| 0.50       |        0.40 |0.44    |     5 |
|avg / total| 0.87       |        0.88 |0.88    |     42 |  

Even though I took off lots of features, the scores are almost same.
And I think the scores are not high to think overfit.

### Third, reduce dimention using PCA

I used **RandomizedPCA** to reduce dimention. I also get the precision and recall score
from **DecisionTreeClassifier**

precision, recall, f1-score by n_components

| n_components   | precision  | recall   | f1-score |
|------------:|------------:|------------:|------------:|
|8	|0.33	|0.6	|0.43
|7	|0.27	|0.6	|0.37
|6	|0.3	|0.6	|0.4
|5	|0.4	|0.4	|0.4
|4	|0.2	|0.2	|0.2

According to the table, I used 8 components of the data

# Run algorithm

Using Selected features and PCA, I run some algorithms.
I used **SVC, DecisionTree, K-neighbors, GaussianNB **

To find best parameter, I used **GridSearchCV** and set _scoring = 'recall'_ to maximize recall score.

In **SVC, DecisionTree, K-neighbors**, I used scaled data and normal data both and use one which show more nice score.

## SVC

This is my code
- rbf kernel: I wanted to use non-linear kernel
- C: I wanted to get nore accuracy
- gamma: I used rbf so I wanted to find appropriate coefficient of this kernel

``` python
svc = SVC(kernel = 'rbf', class_weight = 'balanced')
C= [1, 10, 100, 1000, 10000]
gamma= [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1]
pipe = Pipeline(steps=[('pca', pca), ('SVC', svc)])

clf = GridSearchCV(pipe,
                   dict(SVC__C=C, SVC__gamma=gamma), scoring = 'recall')

clf = clf.fit(features_train, labels_train)
```
This is my best estimator: c = 10, gamma = 0.005

```
Pipeline(steps=[('pca', RandomizedPCA(copy=True, iterated_power=3, n_components=8, random_state=42,
       whiten=True)), ('SVC', SVC(C=10, cache_size=200, class_weight='balanced', coef0=0.0,
  decision_function_shape=None, degree=3, gamma=0.005, kernel='rbf',
  max_iter=-1, probability=False, random_state=None, shrinking=True,
  tol=0.001, verbose=False))])
```

SVC score

This score used normal data (scaled data get precision 0.18, recall 0.4)

|| precision   | recall  | f1-score   | support |
|:--|-----------:|------------:|-----------:|------------:|
|nopoi| 0.94       |        0.84 |0.89    |     37 |
|poi| 0.33       |        0.60 |0.43    |     5 |
|avg / total| 0.87       |        0.81 |0.83    |     42 |

## DecisionTree

This is my code
- critertion : gini and entropy are the ways to check impurity. I just check better way.
- splitter: this is the way to split. default is best. I want to choose better way.
- min_sample_split: It desice volume of each split. If this score is high, algolithm's accuracy can be rough. I want to get approprate ni_sample_split to avoid overfit and nonfit.

``` python
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
```
This is my best estimator: criterion = 'entropy', splitter = 'random', min_samples_split = 1

```
Pipeline(steps=[('pca', RandomizedPCA(copy=True, iterated_power=3, n_components=8, random_state=42,
       whiten=True)), ('tree', DecisionTreeClassifier(class_weight=None, criterion='entropy', max_depth=None,
            max_features=None, max_leaf_nodes=None, min_samples_leaf=1,
            min_samples_split=1, min_weight_fraction_leaf=0.0,
            presort=False, random_state=None, splitter='random'))])
```

DecisionTree score

This score used normal data (scaled data get precision 0.25, recall 0.4)

|| precision   | recall  | f1-score   | support |
|:--|-----------:|------------:|-----------:|------------:|
|nopoi| 0.90       |        0.97 |0.94    |     37 |
|poi| 0.50      |        0.20 |0.29    |     5 |
|avg / total| 0.85       |        0.88 |0.86    |     42 |

## K-neighbors

This is my code
- algorithm: This is the way to compute nearest neighbors. I wanted to choose better ways.
- n_neighbors: It is number of neighbors. If I make this high, It can reduce noise but the accuracy is also low.
- leaf_size: It affect the speed of query. and It can be deside by count of n_point. So I make this parameter a lot to fit n_point.
- I used MinMaxscale in this algolithm to optimized the result.

``` python
knn = KNeighborsClassifier()
algorithm= ['auto', 'ball_tree', 'kd_tree', 'brute']
n_neighbors= [1,3,5,7,9,11,13,15,17,19]
leaf_size= [5,10,20,30,40,50,60]
pipe = Pipeline(steps=[('scaling', MinMaxScaler()),('pca', pca), ('knn', knn)])

clf = GridSearchCV(pipe,
                   dict(knn__algorithm=algorithm,
                        knn__n_neighbors=n_neighbors,
                        knn__leaf_size=leaf_size), scoring = 'recall')

clf = clf.fit(features_train, labels_train)
```
This is my best estimator: algorithm = 'auto', n_neighbors = 1, leaf_size = 5'

```
Pipeline(steps=[('scaling', MinMaxScaler(copy=True, feature_range=(0, 1))), ('pca', RandomizedPCA(copy=True, iterated_power=3, n_components=8, random_state=42,
       whiten=True)), ('knn', KNeighborsClassifier(algorithm='auto', leaf_size=5, metric='minkowski',
           metric_params=None, n_jobs=1, n_neighbors=1, p=2,
           weights='uniform'))])
```
K-neighbors score

This score used scaled data

|| precision   | recall  | f1-score   | support |
|:--|-----------:|------------:|-----------:|------------:|
|nopoi| 0.92       |        0.95 |0.93    |     37 |
|poi| 0.50      |        0.40 |0.44    |     5 |
|avg / total| 0.87       |        0.88 |0.88    |     42 |
`
## GaussianNB

This is my code

``` python
Gaussian = GaussianNB()
clf = GaussianNB()
clf = Pipeline(steps=[('pca', pca), ('gaussian', clf)])
clf = clf.fit(features_train, labels_train)
```

GaussianNB score

|| precision   | recall  | f1-score   | support |
|:--|-----------:|------------:|-----------:|------------:|
|nopoi| 0.94       |        0.92 |0.93    |     37 |
|poi| 0.50      |        0.60 |0.55    |     5 |
|avg / total| 0.89       |        0.88 |0.89    |     42 |

# Conclusion

## Selected algorithm

Goal of this project is identify 'POI's and the data have a lot of non-poi and just a few pois.
So, I concentrate more about 'identifying 'poi' than 'identifying non-poi'.
When I get parameter, which means **gridsearchCV**, I was looking for highest recall model and choose a best algolithm, which have highest recall and approprate precision of 'POI's

I used GaussianNB to identify 'poi' and I think I got a good recall and f1 score.

I select GaussianNB. The score using tester.py is this.

the model

```
Pipeline(steps=[('pca', RandomizedPCA(copy=True, iterated_power=3, n_components=8, random_state=42,
       whiten=True)), ('gaussian', GaussianNB())])
```

| category   | score  |
|:-----------|------------:|
|Accuracy| 0.85350	|
|Precision| 0.48560	|
|Recall| 0.43000	|
|F1| 0.45611	|
|F2| 0.44008|
|Total predictions| 14000	|
|True positives|  860	|
|False positives|  911	|
|False negatives| 1140	|
|True negatives| 11089|

I used tester.py.
The Recall score is 0.43 which is approprate and the precision is also 0.485 which is fitted.
This tester code predict 14,000 data(have 2000 poi) and got 860 True positives which are identify POI nicely. and 911 False positives which are failed to identify POI. and Two negative set(which are identifi non-pois).  

## Review

In this project, I get some information about the data.

First, to identify 'POI', the most features are bonus, and defered_income. Now I knew the information by getting score. However, I will figure the relationship between those features and 'poi' in finantional ways.

Second, my new feature works. It is the funnest thing in this project. To be a good data analyst, I think I have to study other social knowledge broadly.
If I had more knowledge about finance, maybe I can make nice inspire more.
