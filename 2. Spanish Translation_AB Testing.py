import pandas as pd
pd.set_option('display.max_columns', 20)
pd.set_option('display.width', 350)
  
'''Data Import'''
user = pd.read_csv("user_table.csv")
test = pd.read_csv("test_table.csv")

'''Data Quality Check'''
user.head()
user.info(' ')
user.describe()

test.head()
test.info(' ')
test.describe()

# drop duplicates
user = user.drop_duplicates()
test = test.drop_duplicates()

# join the tables
data = test.merge(user, on=['user_id'])

# data format correcting
data.date = pd.to_datetime(data.date)

# country conversion bar chart
country_conversion = pd.DataFrame(data.query('test == 0').groupby('country')['conversion'].mean())
import numpy as np
import matplotlib.pyplot as plt
plt.figure(figsize=(20, 4))
plt.margins(x=0.01)
plt.bar(x = range(len(country_conversion)), height = np.array(country_conversion.conversion),
             data = country_conversion, tick_label=country_conversion.index, width = 0.6)
plt.axhline(y=country_conversion.conversion.mean(), color='r', linestyle='--')
plt.title('Country Conversion')
ax = plt.gca()
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')
plt.show()
# Spain converts much better than LatAm countries

# t-test
# ~0.5MM data and test/control split is ~50/50
from scipy import stats
data_2 = data[data.country != "Spain"]

#t-test of test vs control for target metric 
ttest = stats.ttest_ind(data_2[data_2['test'] == 1]['conversion'], 
                       data_2[data_2['test'] == 0]['conversion'], 
                       equal_var=False)
conversion = pd.DataFrame(data_2.groupby('test')['conversion'].mean())
# control users are converting at 4.8% while test users are at 4.3%
ttest.statistic
ttest.pvalue
# p-value << 0.05, at 95% level, test users conversion is significantly lower 
# than the control group

# compare test and control conversion rate by day
data_test_by_day = data_2.groupby("date")["conversion"].agg({
"test_vs_control": lambda x: x[data_2["test"]==1].mean()/x[data_2["test"]==0].mean()
}).plot()
plt.show()

# group by source and estimate relative frequencies
data_grouped_source = data_2.groupby("source")["test"].agg({
"frequency_test_0": lambda x: len(x[x==0]), 
"frequency_test_1": lambda x: len(x[x==1])
})
source = pd.DataFrame(data_grouped_source/data_grouped_source.sum())

# Check A/B Test Randomization
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
from graphviz import Source
  
# make date as string, so few dates that makes sense to have them as few dummy values  
data['date'] = data['date'].apply(str)
# make dummy vars
data_dummy = pd.get_dummies(data)
# model features, test is the label and conversion is not needed 
train_cols = data_dummy.drop(['test', 'conversion'], axis=1)
# build decision tree model
tree = DecisionTreeClassifier(
    class_weight="balanced",
    min_impurity_decrease = 0.001
    )
tree.fit(train_cols,data_dummy['test'])
  
export_graphviz(tree, out_file="tree_test.dot", feature_names=train_cols.columns, 
                proportion=True, rotate=True)
with open("tree_test.dot") as f:
    dot_graph = f.read()
  
s = Source.from_file("tree_test.dot")
s.view()

data_dummy.groupby("test")[["country_Argentina", "country_Uruguay"]].mean()

# test results using the orginal dataset
original_data = stats.ttest_ind(data_dummy[data['test'] == 1]['conversion'], 
                                data_dummy[data['test'] == 0]['conversion'], 
                                equal_var=False)

# removing Argentina and Uruguay
data_no_AR_UR = stats.ttest_ind(data_dummy[(data['test'] == 1) & 
                                           (data_dummy['country_Argentina'] ==  0) & 
                                           (data_dummy['country_Uruguay'] ==  0)
                                           ]['conversion'], 
                                data_dummy[(data['test'] == 0) & 
                                           (data_dummy['country_Argentina'] ==  0) & 
                                           (data_dummy['country_Uruguay'] ==  0)
                                           ]['conversion'], 
                                equal_var=False)
  

new = pd.DataFrame({"data_type" : ["Full", "Removed_Argentina_Uruguay"], 
                  "p_value" : [original_data.pvalue, data_no_AR_UR.pvalue],
                  "t_statistic" : [original_data.statistic, data_no_AR_UR.statistic]
                 })
    
data_test_country = data.groupby('country')['conversion'].agg({
                  "p_value": lambda x: stats.ttest_ind(x[data["test"]==1], 
                                                        x[data["test"]==0], 
                                                        equal_var=False).pvalue,
                   "conversion_test": lambda x: x[data["test"]==1].mean(),
                   "conversion_control": lambda x: x[data["test"]==0].mean()
                   }).reindex(['p_value','conversion_test','conversion_control'], axis=1)


