################################################
### Exercise 1: Interpolation for missing values
################################################

import pandas as pd
from scipy.interpolate import lagrange
from numpy.distutils.conv_template import header
inputfile = 'E:/Learn_Python/power_theft/data/missing_data.xls'
outputfile = 'E:/Learn_Python/power_theft/data/missing_data_processed.xls'

data = pd.read_excel(inputfile,header=None)

print data.iloc[range(5)] # check read-in data

#Lagrange ploynomial interpolation. s: column vector; n: missing value index; k: data range for the ploynoimal
def ployinnterp_column(s,n,k=5): # k cannot be larger than 9
    y = s[list(range(n-k,n)) + list(range(n+1, n+1+k))]
    y = y[y.notnull()]
    return lagrange(y.index, list(y))(n)

for i in data.columns:
    for j in range(len(data)):
        if (data[i].isnull())[j]:
            data[i][j] = ployinnterp_column(data[i], j)

data.to_excel(outputfile, header=None, index=False) # check the output after interpolation

########################################
### Exercise 2: Build CART decision tree
########################################

def cm_plot(y, yp): # visualization function
    from sklearn.metrics import confusion_matrix 
    cm = confusion_matrix(y, yp) 
    import matplotlib.pyplot as plt
    plt.matshow(cm, cmap=plt.cm.Greens)
    plt.colorbar() # color label
    for x in range(len(cm)): # data label
        for y in range(len(cm)):
            plt.annotate(cm[x,y], xy=(x, y), horizontalalignment='center', verticalalignment='center')
    plt.ylabel('True label') # x-axis label
    plt.xlabel('Predicted label') # y-axis label
    return plt

import pandas as pd
from random import shuffle
inputfile = 'E:/Learn_Python/power_theft/data/model.xls'
data = pd.read_excel(inputfile,header=1) # read-in model data
print data.iloc[range(5)] # check read-in data

data_m = data.as_matrix() # convert to matrix format to shuffle the data
shuffle(data_m)

p = 0.8 # proportion of the training set
train = data_m[:int(len(data_m)*p),:] # training set
test = data_m[int(len(data_m)*p):,:] # validation set

from  sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
from sklearn import tree as t
treefile = 'E:/Learn_Python/power_theft/data/tree.pkl' # output tree

tree = DecisionTreeClassifier() # create tree
tree.fit(train[:,:3],train[:,3]) # train the model
t.export_graphviz(tree, out_file='tree.dot') # graphviz web portal address: http://webgraphviz.com

from sklearn.externals import joblib
joblib.dump(tree, treefile)

y = train[:,3]
yp = tree.predict(train[:,:3])
cm_plot(y, yp).show() # plot the confusion matrix

# draw ROC curve on the validation set
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
y_true = test[:,3]
y_score = tree.predict_proba(test[:,:3])[:,1]
fpr, tpr, thresholds = roc_curve(y_true, y_score, pos_label=1)

plt.plot(fpr, tpr, linewidth=2, label='ROC of CART')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.xlim(0,1.05)
plt.ylim(0,1.05)
plt.legend(loc=4)
plt.show()
