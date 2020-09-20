# You should implement five learning algorithms. They are:
# Decision trees with some form of pruning
# Neural networks
# Boosting
# Support Vector Machines
# k-nearest neighbors
# Each algorithm is described in detail in your textbook, the handouts, and all over the web. In fact, instead of implementing the algorithms yourself, you may (and by may I mean should) use software packages that you find elsewhere; however, if you do so you should provide proper attribution. Also, you will note that you have to do some fiddling to get good results, graphs and such, so even if you use another's package, you may need to be able to modify it in various ways.
# Decision Trees. For the decision tree, you should implement or steal a decision tree algorithm (and by "implement or steal" I mean "steal"). Be sure to use some form of pruning. You are not required to use information gain (for example, there is something called the GINI index that is sometimes used) to split attributes, but you should describe whatever it is that you do use.
# Neural Networks. For the neural network you should implement or steal your favorite kind of network and training algorithm. You may use networks of nodes with as many layers as you like and any activation function you see fit.
# Boosting. Implement or steal a boosted version of your decision trees. As before, you will want to use some form of pruning, but presumably because you're using boosting you can afford to be much more aggressive about your pruning.
# Support Vector Machines. You should implement (for sufficiently loose definitions of implement including "download") SVMs. This should be done in such a way that you can swap out kernel functions. I'd like to see at least two.
# k-Nearest Neighbors. You should "implement" (the quotes mean I don't mean it: steal the code) kNN. Use different values of k.
# Testing. In addition to implementing (wink) the algorithms described above, you should design two interesting classification problems. For the purposes of this assignment, a classification problem is just a set of training examples and a set of test examples. I don't care where you get the data. You can download some, take some from your own research, or make some up on your own. Be careful about the data you choose, though. You'll have to explain why they are interesting, use them in later assignments, and come to really care about them.
# Sources: https://towardsdatascience.com/decision-tree-build-prune-and-visualize-it-using-python-12ceee9af752
#           https://scikit-learn.org/stable/auto_examples/neural_networks/plot_mlp_training_curves.html?fbclid=IwAR239XIIIqYEVLia--s-p91epOrH3TjrXFnzq7h5z2Uw12T3vNCedzNh_FM
#           https://towardsdatascience.com/machine-learning-part-17-boosting-algorithms-adaboost-in-python-d00faac6c464

import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, mean_squared_error, r2_score, plot_roc_curve
from matplotlib.colors import ListedColormap
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.ensemble import  AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
import pickle
import warnings
import skimage.io
import skimage.viewer
from skimage.filters import gaussian
from time import time
from skimage.feature import hog
from skimage.viewer import  ImageViewer
# from skimage.transform import rescale

#---------------------------------------- PRE PROCESSING --------------------------------------------------
def pre_process1(data, noise):
    # https://stackoverflow.c"'
    headers = list(data.columns)
    if not not noise:
        # is not empty.
        noisy_data = data[headers[1:(len(headers) - 1)]]
        noisy_header = headers[1:(len(headers) - 1)]
        noisy_shape = np.shape(noisy_data)
        noisy_variables = int(noisy_shape[0] * noisy_shape[1] * noise)
        noisyx = np.random.randint(0, noisy_shape[0], noisy_variables)
        noisyy = np.random.randint(0, noisy_shape[1], noisy_variables)
        for y in noisyy:
            data.loc[noisyx[noisyy == y], noisy_header[y]] = 'NA'

    vals = data[headers[1:len(headers)]].stack().drop_duplicates().values
    b = [x for x in data[headers[1:len(headers)]].stack().drop_duplicates().rank(method='dense')]
    dictionary = dict(zip(b, vals))  # dictionary for digitization.

    stacked = data[headers[1:len(headers)]].stack()
    data[headers[1:len(headers)]] = pd.Series(stacked.factorize()[0], index=stacked.index).unstack()
    return data, dictionary

def pre_process2(data, noise):
    # https://stackoverflow.c"'
    headers = list(data.columns)
    if not not noise:
        # is not empty.
        noisy_data = data[headers[:(len(headers) - 1)]]
        noisy_header = headers[:(len(headers) - 1)]
        noisy_shape = np.shape(noisy_data)
        noisy_variables = int(noisy_shape[0] * noisy_shape[1] * noise)
        noisyx = np.random.randint(0, noisy_shape[0], noisy_variables)
        noisyy = np.random.randint(0, noisy_shape[1], noisy_variables)
        data.loc[zip(noisyx[noisyy], noisy_header)] = 'NA'
        # for y in noisyy:
        #     data.loc[noisyx[noisyy == y], noisy_header[y]] = 'NA'

    vals = data[headers].stack().drop_duplicates().values
    b = [x for x in data[headers].stack().drop_duplicates().rank(method='dense')]
    dictionary = dict(zip(b, vals))  # dictionary for digitization.

    stacked = data[headers].stack()
    data[headers] = pd.Series(stacked.factorize()[0], index=stacked.index).unstack()
    return data, dictionary

def pre_process3(data_loc, noise):
    labels = os.listdir(data_loc)
    label_data = []
    hog_data = []
    label_count = 0
    for label in labels:
        img_titles = os.listdir(os.path.join(data_loc, label))
        for img_title in img_titles:
            file_path = os.path.join(data_loc, label, img_title)
            if not os.path.isfile(file_path):
                print('problem reading '+ file_path)
            else:
                img = skimage.io.imread(fname=file_path)
                # blurred = [img]
                # blurred = gaussian(img, sigma= (3,3), multichannel=True)
                hog_image = hog(img,feature_vector=True)
                hog_data.append(np.append( hog_image, label_count))
                label_data.append(np.append(img.flatten(), label_count))
        # data.append(label_data)
        label_count += 1

    data = pd.DataFrame(np.asarray(label_data))
    thishog = pd.DataFrame(np.asarray(hog_data))
    return data, labels, thishog

#################################Decision tree classifier########################################################
class decision_tree:
    # This is using the decision_tree from sklearn.
    # https://towardsdatascience.com/decision-tree-build-prune-and-visualize-it-using-python-12ceee9af752
    def __init__(self, data=[], csv_file='diabetes_data_upload.csv', test_size = 0.25, criterion='entropy', max_depth='none', boosting=False, n_estimators = 50, learning_rate = 1, noise=[], preprocess=[]):
        if not csv_file:
            self.data, self.dictionary = data
        else:
            self.data = pd.read_csv(csv_file)
            if not not preprocess:
                self.data, self.dictionary = preprocess(self.data, noise)
        x = self.data.iloc[:,0:-1]
        y = self.data.iloc[:,-1]
        self.xtrain, self.xtest, self.ytrain, self.ytest = train_test_split(x, y, test_size=test_size)
        self.criterion = criterion
        self.max_depth = max_depth # pruning values.

        if max_depth == 'none':
            this_tree = DecisionTreeClassifier(criterion=self.criterion)
        else:
            this_tree = DecisionTreeClassifier(criterion=self.criterion,max_depth=self.max_depth)

        if boosting:
            self.tree = AdaBoostClassifier(this_tree, n_estimators=n_estimators, learning_rate=learning_rate)
        else:
            self.tree = this_tree

    def train(self):

        self.tree.fit(self.xtrain, self.ytrain)

    def predict(self, x_in):
        return self.tree.predict(x_in)

    def accuracy(self):
        train_acc = accuracy_score(self.ytrain, self.predict(self.xtrain))
        test_acc = accuracy_score(self.ytest, self.predict(self.xtest))
        conf_mat = confusion_matrix(self.ytest, self.predict(self.xtest))
        return train_acc, test_acc,  conf_mat

    def roc(self):
        # https://abdalimran.github.io/2019-06-01/Drawing-multiple-ROC-Curves-in-a-single-plot
        return plot_roc_curve(self.tree, self.xtest, self.ytest)

    # Accuracy: The amount of correct classifications / the total amount of classifications.
    # The train accuracy: The accuracy of a model on examples it was constructed on.
    # The test accuracy is the accuracy of a model on examples it hasn't seen.
    # Confusion matrix: A tabulation of the predicted class (usually vertically) against the actual class (thus horizontally).

#################################MLP Neural net Classifier#####################################################
class neural_net:
    def __init__(self, data=[],csv_file='diabetes_data_upload.csv', params=[], test_size=0.25, noise=[], preprocess=[]):
        if not csv_file:
            self.data, self.dictionary = data
        else:
            self.data = pd.read_csv(csv_file)
            if not not preprocess:
                self.data, self.dictionary = preprocess(self.data, noise)
        self.params = params

        x = self.data.iloc[:,0:-1]
        y = self.data.iloc[:,-1]
        self.xtrain, self.xtest, self.ytrain, self.ytest = train_test_split(x, y, test_size=test_size, shuffle=True)
        self.build_nn()


    def build_nn(self):
        params = self.params
        self.nn = MLPClassifier(**params)

    def train(self):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=ConvergenceWarning,module="sklearn")
            self.nn.fit(self.xtrain, self.ytrain)

    def predict(self, x_in):
        return self.nn.predict(x_in)

    def accuracy(self):
        train_acc = accuracy_score(self.ytrain, self.predict(self.xtrain))
        test_acc = accuracy_score(self.ytest, self.predict(self.xtest))
        conf_mat = confusion_matrix(self.ytest, self.predict(self.xtest))
        return train_acc, test_acc,  conf_mat

#################################Super Vector Machine Classifier################################################
class SVM:
    def __init__(self, data =[], csv_file='diabetes_data_upload.csv', params=[], test_size=0.25, noise=[], preprocess=[]):
        if not csv_file:
            self.data, self.dictionary = data
        else:
            self.data = pd.read_csv(csv_file)
            if not not preprocess:
                self.data, self.dictionary = preprocess(self.data, noise)

        self.params = params
        x = self.data.iloc[:,0:-1]
        y = self.data.iloc[:,-1]
        self.xtrain, self.xtest, self.ytrain, self.ytest = train_test_split(x, y, test_size=test_size, shuffle=True)
        self.build_svn()

    def build_svn(self):
        params = self.params
        self.svn = SVC(**params)

    def train(self):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=ConvergenceWarning,module="sklearn")
            self.svn.fit(self.xtrain, self.ytrain)

    def predict(self, x_in):
        return self.svn.predict(x_in)

    def accuracy(self):
        train_acc = accuracy_score(self.ytrain, self.predict(self.xtrain))
        test_acc = accuracy_score(self.ytest, self.predict(self.xtest))
        conf_mat = confusion_matrix(self.ytest, self.predict(self.xtest))
        return train_acc, test_acc,  conf_mat

#################################K Neighbor Classifier################################################
class KNN:
    def __init__(self, data = [], csv_file='diabetes_data_upload.csv', params=[], test_size=0.25, noise=[], preprocess=[]):
        if not csv_file:
            self.data, self.dictionary = data
        else:
            self.data = pd.read_csv(csv_file)
            if not not preprocess:
                self.data, self.dictionary = preprocess(self.data, noise)

        self.params = params
        x = self.data.iloc[:,0:-1]
        y = self.data.iloc[:,-1]
        self.xtrain, self.xtest, self.ytrain, self.ytest = train_test_split(x, y, test_size=test_size, shuffle=True)
        self.build_knn()

    def build_knn(self):
        params = self.params
        self.knn = KNeighborsClassifier(**params)


    def train(self):
        # with warnings.catch_warnings():
        #     warnings.filterwarnings("ignore", category=ConvergenceWarning,module="sklearn")
        self.knn.fit(self.xtrain, self.ytrain)

    def predict(self, x_in):
        return self.knn.predict(x_in)

    def accuracy(self):
        train_acc = accuracy_score(self.ytrain, self.predict(self.xtrain))
        test_acc = accuracy_score(self.ytest, self.predict(self.xtest))
        conf_mat = confusion_matrix(self.ytest, self.predict(self.xtest))
        return train_acc, test_acc,  conf_mat

def decision_tree_testing(data, csv_file, preprocess=pre_process1, nflag=2):
    # Decision Tree Classifier main
    # Testing test_split_size

    labels = ["training", "testing", "training w/25% noise", "testing w/25% noise"]

    plot_args=[{'c': 'red', 'linestyle': '-'},
                     {'c': 'red', 'linestyle': '--'},
                     {'c': 'blue', 'linestyle': '-'},
                     {'c': 'blue', 'linestyle': '--'}]
    ct1 = 0
    ct2 = 0

    plt.close()

    fig1, ax1 = plt.subplots()

    fig2, ax2 = plt.subplots()

    for j in range(nflag):
        noisy = j/4 # % of data loss.
        print(' ----- INTRODUCING MISSING DATA ', noisy*100, '%')
        if noisy == 0:
            noise = []
        else:
            noise = noisy


        split_acc = []
        best_acc = -9999
        best_split = 0.1
        train_acc = -1
        test_acc = -1
        x_axis = range(10, 50, 5)
        for i in x_axis:
            percentage = i/100
            print('new training set on test split ', percentage*100, '% of 50% | best split = ', best_split, ' train_acc = ', train_acc, ' test_acc = ', test_acc , ' best_acc = ', best_acc)
            dt = decision_tree(data= data, csv_file=csv_file, test_size=percentage, noise=noise, preprocess=preprocess)
            # dt.pre_process1()
            dt.train()
            [train_acc, test_acc, conf_mat] = dt.accuracy()
            split_acc.append([train_acc, test_acc, conf_mat])
            if (np.mean([train_acc,test_acc])) >= best_acc:
                best_acc = np.mean([train_acc,test_acc])
                best_split = percentage

        for i in range(2):
            this_data = [d[i] for d in split_acc]
            ax1.plot(x_axis, this_data, label=[labels[ct1]], **plot_args[ct1])
            ct1 += 1

        print('best split = ', best_split)
        best_split = 0.20
        prune_acc = []
        best_acc = -9999
        best_prune = 1
        train_acc = -1
        test_acc = -1
        x_axis2 = range(1, 20)
        for i in x_axis2:
            print('new training set on max_depth ', i, ' of 30 | best_prune = ', best_prune, ' train_acc = ', train_acc, ' test_acc = ', test_acc, ' best_acc = ', best_acc)
            dt = decision_tree(data= data, csv_file=csv_file, test_size=best_split, max_depth=i, noise=noise, preprocess=preprocess)
            dt.train()
            [train_acc, test_acc, conf_mat] = dt.accuracy()
            prune_acc.append([train_acc, test_acc, conf_mat])
            if (np.mean([train_acc,test_acc])) >= best_acc:
                best_acc = np.mean([train_acc,test_acc])
                best_prune = i

        for i in range(2):
            this_data = [d[i] for d in prune_acc]
            ax2.plot(x_axis2, this_data, label=[labels[ct2]], **plot_args[ct2])
            ct2 += 1

    ax1.set_title('exploring data split w/ Decision Trees (0% noise vs 25% noise)')
    ax1.set_xlabel('data split %')
    ax1.set_ylabel('% accurate')
    fig1.legend(ax1.get_lines(), labels, ncol=3, loc="upper center")
    with open('./figures/decision_tree_data_split1.pkl', 'wb') as fid:
        pickle.dump(ax1, fid)
    fig1.savefig('./figures/decision_tree_data_split1.png')# fig1.show()
    ax2.set_title('exploring pruning w/ Decision Trees (0% noise vs 25% noise)')
    ax2.set_xlabel('tree max depth (pruning)')
    ax2.set_ylabel('% accurate')
    fig2.legend(ax2.get_lines(), labels, ncol=3, loc="upper center")
    with open('./figures/decision_tree_prune1.pkl', 'wb') as fid:
        pickle.dump(ax2, fid)
    fig2.savefig('./figures/decision_tree_prune1.png')#fig2.show()
    # plt.show()

def neural_net_testing(data, csv_file, preprocess=pre_process1, nflag=2):

    plot_args = [{'c': 'red', 'linestyle': '-'},
                 {'c': 'red', 'linestyle': '--'},
                 {'c': 'red', 'linestyle': '-.'},
                 {'c': 'orange', 'linestyle': '-'},
                 {'c': 'orange', 'linestyle': '--'},
                 {'c': 'orange', 'linestyle': '-.'},
                 {'c': 'blue', 'linestyle': '-'},
                 {'c': 'blue', 'linestyle': '--'},
                 {'c': 'blue', 'linestyle': '-.'},
                 {'c': 'green', 'linestyle': '--'},
                 {'c': 'green', 'linestyle': '-'},
                 {'c': 'green', 'linestyle': '-.'}]

    # ------------------------------------------------- learning_ rate & activation --------------------------------------------------------------
    params = [{'solver': 'adam', 'learning_rate_init': 0.05, 'activation': 'identity'},
              {'solver': 'adam', 'learning_rate_init': 0.01, 'activation': 'identity'},
              {'solver': 'adam', 'learning_rate_init': 0.005, 'activation': 'identity'},
              {'solver': 'adam', 'learning_rate_init': 0.05, 'activation': 'logistic'},
              {'solver': 'adam', 'learning_rate_init': 0.01, 'activation': 'logistic'},
              {'solver': 'adam', 'learning_rate_init': 0.005, 'activation': 'logistic'},
              {'solver': 'adam', 'learning_rate_init': 0.05, 'activation': 'tanh'},
              {'solver': 'adam', 'learning_rate_init': 0.01, 'activation': 'tanh'},
              {'solver': 'adam', 'learning_rate_init': 0.005, 'activation': 'tanh'},
              {'solver': 'adam', 'learning_rate_init': 0.05, 'activation': 'relu'},
              {'solver': 'adam', 'learning_rate_init': 0.01, 'activation': 'relu'},
              {'solver': 'adam', 'learning_rate_init': 0.005, 'activation': 'relu'}]

    labels = ["adam 0.05 identity", "adam 0.01 identity", "adam 0.005 identity",
              "adam 0.05 logistic", "adam 0.01 logistic", "adam 0.005 logistic",
              "adam 0.05 tanh", "adam 0.01 tanh", "adam 0.005 tanh",
              "adam 0.05 relu", "adam 0.01 relu", "adam 0.005 relu"]

    plt.close()
    fig, ax = plt.subplots()

    split_acc = []
    noise = []
    mlps = []

    for j in range(nflag):
        noisy = j/4
        print(' ----- INTRODUCING MISSING DATA ', noisy*100, '%')
        if noisy > 0:
            noise = noisy

        for label, param in zip(labels, params):
            print('Trying: ', param)
            nn = neural_net(data= data, csv_file=csv_file, noise=noise, params=param,preprocess=preprocess)
            nn.train()
            [train_acc, test_acc, conf_mat] = nn.accuracy()
            print(' train_acc = ', train_acc,' test_acc = ', test_acc, ' hidden layer used: ', nn.nn.hidden_layer_sizes, ' activation used: ', nn.nn.activation)
            split_acc.append([train_acc, test_acc, conf_mat])
            mlps.append(nn)

    for mlp, label, args in zip(mlps, labels, plot_args):

        if mlp.nn.solver == 'lbfgs':
            ax.plot(range(mlp.nn.max_iter), [mlp.nn.loss_]*mlp.nn.max_iter, label=label, **args)
        else:
            ax.plot(mlp.nn.loss_curve_, label=label, **args)

    ax.set_title('neural nets learning rate & activation (loss vs training instance)')
    fig.legend(ax.get_lines(), labels, ncol=3, loc="upper center")
    ax.set_xlabel('iteration')
    ax.set_ylabel('data loss')
    with open('./figures/neural_net_lr_activate_1.pkl', 'wb') as fid:
        pickle.dump(ax, fid)
    fig.savefig('./figures/neural_net_lr_activate_1.png')  #plt.show()

    #-------------------------------------------------solvers --------------------------------------------------------------
    params = [{'solver': 'lbfgs'},
              {'solver': 'sgd', 'learning_rate_init': 0.01},
              {'solver': 'adam', 'learning_rate_init': 0.01}]

    labels = ["lbfgs", "sgd 0.01", "adam 0.01"]

    fig, ax = plt.subplots()

    split_acc = []
    noise = []
    mlps = []

    for j in range(nflag):
        noisy = j/4
        print(' ----- INTRODUCING MISSING DATA ', noisy*100, '%')
        if noisy > 0:
            noise = noisy

        for label, param in zip(labels, params):
            print('Trying: ', param)
            nn = neural_net(data= data, csv_file=csv_file, noise=noise, params=param,preprocess=preprocess)
            nn.train()
            [train_acc, test_acc, conf_mat] = nn.accuracy()
            print(' train_acc = ', train_acc,' test_acc = ', test_acc, ' hidden layer used: ', nn.nn.hidden_layer_sizes, ' activation used: ', nn.nn.activation)
            split_acc.append([train_acc, test_acc, conf_mat])
            mlps.append(nn)

    for mlp, label, args in zip(mlps, labels, plot_args):

        if mlp.nn.solver == 'lbfgs':
            ax.plot(range(mlp.nn.max_iter), [mlp.nn.loss_]*mlp.nn.max_iter, label=label, **args)
        else:
            ax.plot(mlp.nn.loss_curve_, label=label, **args)

    fig.legend(ax.get_lines(), labels, ncol=3, loc="upper center")
    ax.set_title('neural nets solvers (loss vs training instance)')
    ax.set_xlabel('iteration')
    ax.set_ylabel('data loss')
    with open('./figures/neural_net_solver_1.pkl', 'wb') as fid:
        pickle.dump(ax, fid)
    fig.savefig('./figures/neural_net_solver_1.png')  #plt.show()


    params = [{'hidden_layer_sizes': (50, ), 'solver': 'adam', 'learning_rate_init': 0.01},
              {'hidden_layer_sizes': (25, ), 'solver': 'adam', 'learning_rate_init': 0.01},
              {'hidden_layer_sizes': (16, ), 'solver': 'adam', 'learning_rate_init': 0.01},
              {'hidden_layer_sizes': (16, 4), 'solver': 'adam', 'learning_rate_init': 0.01},
              {'hidden_layer_sizes': (100, 10, 3), 'solver': 'adam', 'learning_rate_init': 0.01},
              {'hidden_layer_sizes': (16, 4, 2), 'solver': 'adam', 'learning_rate_init': 0.01}]

    labels = ["hidden layer (50, )", "hidden layer (25, )", "hidden layer (16, )",
              "hidden layer (16, 4)", "hidden layer (100, 10, 3)", "hidden layer (16, 4, 2)"]

    fig, ax = plt.subplots()

    split_acc = []
    noise = []
    mlps = []

    ax.set_title('exploring neural nets (loss vs training instance)')
    for j in range(nflag):
        noisy = j/4
        print(' ----- INTRODUCING MISSING DATA ', noisy*100, '%')
        if noisy > 0:
            noise = noisy

        for label, param in zip(labels, params):
            print('Trying: ', param)
            nn = neural_net(data= data, csv_file=csv_file, noise=noise, params=param,preprocess=preprocess)
            nn.train()
            [train_acc, test_acc, conf_mat] = nn.accuracy()
            print(' train_acc = ', train_acc,' test_acc = ', test_acc, ' hidden layer used: ', nn.nn.hidden_layer_sizes, ' activation used: ', nn.nn.activation)
            split_acc.append([train_acc, test_acc, conf_mat])
            mlps.append(nn)

    for mlp, label, args in zip(mlps, labels, plot_args):

        if mlp.nn.solver == 'lbfgs':
            ax.plot(range(mlp.nn.max_iter), [mlp.nn.loss_]*mlp.nn.max_iter, label=label, **args)
        else:
            ax.plot(mlp.nn.loss_curve_, label=label, **args)

    fig.legend(ax.get_lines(), labels, ncol=3, loc="upper center")
    ax.set_title('neural nets hidden layers (loss vs training instance)')
    ax.set_xlabel('iteration')
    ax.set_ylabel('data loss')
    with open('./figures/neural_net_hidden_layer_1.pkl', 'wb') as fid:
        pickle.dump(ax, fid)
    fig.savefig('./figures/neural_net_hidden_layer_1.png')#plt.show()

def boosting_testing(data, csv_file,preprocess=pre_process1, nflag=2):
    # Decision Tree Classifier main
    # Testing test_split_size

    labels = ["training", "testing", "training w/25% noise", "testing w/25% noise"]

    plot_args=[{'c': 'red', 'linestyle': '-'},
                     {'c': 'red', 'linestyle': '--'},
                     {'c': 'blue', 'linestyle': '-'},
                     {'c': 'blue', 'linestyle': '--'}]
    ct1 = 0
    ct2 = 0
    ct3 = 0

    plt.close()

    fig1, ax1 = plt.subplots()

    fig2, ax2 = plt.subplots()

    fig3, ax3 = plt.subplots()

    for j in range(nflag):
        noisy = j/4 # % of data loss.
        print(' ----- INTRODUCING MISSING DATA ', noisy*100, '%')
        if noisy == 0:
            noise = []
        else:
            noise = noisy
        best_split = 0.2

        lr_acc = []
        best_acc = -9999
        best_lr = 1
        train_acc = -1
        test_acc = -1
        xaxis1 = range(5)
        lr_axis = 1/np.power(10,range(5))
        for i in xaxis1:
            lr = 1/np.power(10,i)
            print('new training set on learning_rate ', lr, ' | best learnrate = ', best_lr, ' train_acc = ', train_acc, ' test_acc = ', test_acc , ' best_acc = ', best_acc)
            dt = decision_tree(data= data, csv_file=csv_file, test_size=best_split, noise=noise, boosting=True, learning_rate=lr,preprocess=preprocess)
            # dt.pre_process1()
            dt.train()
            [train_acc, test_acc, conf_mat] = dt.accuracy()
            lr_acc.append([train_acc, test_acc, conf_mat])
            if (np.mean([train_acc,test_acc])) >= best_acc:
                best_acc = np.mean([train_acc,test_acc])
                best_lr = lr

        for i in range(2):
            this_data = [d[i] for d in lr_acc]
            ax1.plot(xaxis1, this_data, label=[labels[ct1]], **plot_args[ct1])
            ct1 += 1

        n_est_acc=[]
        best_acc = -9999
        best_ne = 1
        train_acc = -1
        test_acc = -1
        xaxis2 = range(20)
        ne_axis = 5*(np.add(xaxis2,1))
        for i in xaxis2:
            n_estimators = np.int(5*(i+1))
            print('new training set on num_estimators ', n_estimators, ' | best num_estimators = ', best_ne, ' train_acc = ', train_acc, ' test_acc = ', test_acc , ' best_acc = ', best_acc)
            dt = decision_tree(data= data, csv_file=csv_file, test_size=best_split, noise=noise, boosting=True, learning_rate=best_lr, n_estimators=n_estimators,preprocess=preprocess)
            # dt.pre_process1()
            dt.train()
            [train_acc, test_acc, conf_mat] = dt.accuracy()
            n_est_acc.append([train_acc, test_acc, conf_mat])
            if (np.mean([train_acc,test_acc])) >= best_acc:
                best_acc = np.mean([train_acc,test_acc])
                best_ne = n_estimators

        for i in range(2):
            this_data = [d[i] for d in n_est_acc]
            ax2.plot(ne_axis, this_data, label=[labels[ct2]], **plot_args[ct2])
            ct2 += 1

        # print('best split = ', best_split)
        # PRUNING
        prune_acc = []
        best_acc = -9999
        best_prune = 1
        train_acc = -1
        test_acc = -1
        xaxis3=range(1, 20)
        for i in xaxis3:
            print('new training set on max_depth ', i, ' of 30 | (lr=', best_lr, 'ne=', best_ne, ') | best_prune = ', best_prune, ' train_acc = ', train_acc, ' test_acc = ', test_acc, ' best_acc = ', best_acc)
            dt = decision_tree(data= data, csv_file=csv_file, test_size=best_split, max_depth=i, noise=noise, boosting=True, learning_rate=best_lr, n_estimators=best_ne,preprocess=preprocess)
            dt.train()
            [train_acc, test_acc, conf_mat] = dt.accuracy()
            prune_acc.append([train_acc, test_acc, conf_mat])
            if (np.mean([train_acc,test_acc])) >= best_acc:
                best_acc = np.mean([train_acc,test_acc])
                best_prune = i

        for i in range(2):
            this_data = [d[i] for d in prune_acc]
            ax3.plot(xaxis3, this_data, label=[labels[ct3]], **plot_args[ct3])
            ct3 += 1


    ax1.set_title('exploring learning rate w/ Boosting (0% noise vs 25% noise)')
    ax1.set_xlabel('learning rate')
    ax1.set_ylabel('% accurate')
    fig1.legend(ax1.get_lines(), labels, ncol=3, loc="upper center")
    with open('./figures/boosting_lr_1.pkl', 'wb') as fid:
        pickle.dump(ax1, fid)
    fig1.savefig('./figures/boosting_lr_1.png')  #fig1.show()

    ax2.set_title('exploring # of estimators w/ Boosting (0% noise vs 25% noise)')
    ax2.set_xlabel('# of estimators')
    ax2.set_ylabel('% accurate')
    fig2.legend(ax2.get_lines(), labels, ncol=3, loc="upper center")
    with open('./figures/boosting_num_est_1.pkl', 'wb') as fid:
        pickle.dump(ax2, fid)
    fig2.savefig('./figures/boosting_num_est_1.png')  #fig2.show()

    ax3.set_title('exploring pruning w/ Boosting (0% noise vs 25% noise)')
    ax3.set_xlabel('tree max depth (pruning)')
    ax3.set_ylabel('% accurate')
    fig3.legend(ax3.get_lines(), labels, ncol=3, loc="upper center")
    with open('./figures/boosting_prune_1.pkl', 'wb') as fid:
        pickle.dump(ax3, fid)
    fig3.savefig('./figures/boosting_prune_1.png')  #fig3.show()
    # plt.show()

def svm_testing(data, csv_file, preprocess=pre_process1, nflag=2):
    plt.close()
    fig, ax = plt.subplots()

    params = [{'kernel': 'poly', 'degree': 2},
              {'kernel': 'poly', 'degree': 5},
              {'kernel': 'poly', 'degree': 8},
              #{'kernel': 'poly', 'degree': 10},
              {'kernel': 'rbf'},
              {'kernel': 'sigmoid'}]

    labels = ["training", "testing", "training w/25% noise", "testing w/25% noise"]

    xlabels = ["poly deg 2", "poly deg 5", "poly deg 8", "sigmoid", "gaussian" ]

    plot_args=[{'c': 'red', 'linestyle': '-'},
                     {'c': 'red', 'linestyle': '--'},
                     {'c': 'blue', 'linestyle': '-'},
                     {'c': 'blue', 'linestyle': '--'}]
    mlps = []
    ct = 0
    for j in range(nflag):
        split_acc = []
        noise = []

        noisy = j/4

        # labels = ["poly deg 2 w/ noise "+str(noisy), "poly deg 5 w/ noise "+str(noisy), "poly deg 8 w/ noise "+str(noisy), "poly deg 10 w/ noise "+str(noisy), "gaussian w/ noise "+str(noisy), "sigmoid w/ noise "+str(noisy)]

        print(' ----- INTRODUCING MISSING DATA ', noisy*100, '%')
        if noisy > 0:
            noise = noisy

        for param in params:
            print('Trying: ', param)
            svm = SVM(data= data, csv_file=csv_file, noise=noise, params=param,preprocess=preprocess)
            svm.train()
            [train_acc, test_acc, conf_mat] = svm.accuracy()
            print(' train_acc = ', train_acc,' test_acc = ', test_acc, ' params used: ', param)
            split_acc.append([train_acc, test_acc, conf_mat])
            mlps.append(svm)

        for i in range(2):
            this_data = [d[i] for d in split_acc]
            this_x = range(len(this_data))
            ax.plot(this_x, this_data, label=[labels[ct]], **plot_args[ct])
            ct += 1


    fig.legend(ax.get_lines(), labels, ncol=3, loc="upper center")
    plt.xticks(this_x, xlabels)
    ax.set_title('exploring kernels w/ SVN (0% noise vs 25% noise)')
    ax.set_xlabel('kernel used')
    ax.set_ylabel('% accurate')
    with open('./figures/svn_kernels_1.pkl', 'wb') as fid:
        pickle.dump(ax, fid)
    fig.savefig('./figures/svn_kernels_1.png')  #plt.show()
    shit = 0

def knn_testing(data, csv_file,preprocess=pre_process1, nflag=2):

    params = [{'n_neighbors': 3},
              {'n_neighbors': 5},
              {'n_neighbors': 7},
              {'n_neighbors': 9},
              {'n_neighbors': 11}]

    labels = ["training", "testing", "training w/25% noise", "testing w/25% noise"]

    xlabels = ["k = 3", "k = 5", "k = 7", "k = 9", "k = 11"]

    plot_args=[{'c': 'red', 'linestyle': '-'},
                     {'c': 'red', 'linestyle': '--'},
                     {'c': 'blue', 'linestyle': '-'},
                     {'c': 'blue', 'linestyle': '--'}]
    mlps = []
    ct = 0

    plt.close()
    fig, ax = plt.subplots()
    for j in range(nflag):
        split_acc = []
        noise = []

        noisy = j/4

        # labels = ["poly deg 2 w/ noise "+str(noisy), "poly deg 5 w/ noise "+str(noisy), "poly deg 8 w/ noise "+str(noisy), "poly deg 10 w/ noise "+str(noisy), "gaussian w/ noise "+str(noisy), "sigmoid w/ noise "+str(noisy)]

        print(' ----- INTRODUCING MISSING DATA ', noisy*100, '%')
        if noisy > 0:
            noise = noisy

        for param in params:
            print('Trying: ', param)
            svm = KNN(data= data, csv_file=csv_file, noise=noise, params=param,preprocess=preprocess)
            svm.train()
            [train_acc, test_acc, conf_mat] = svm.accuracy()
            print(' train_acc = ', train_acc,' test_acc = ', test_acc, ' params used: ', param)
            split_acc.append([train_acc, test_acc, conf_mat])
            mlps.append(svm)

        for i in range(2):
            this_data = [d[i] for d in split_acc]
            this_x = range(len(this_data))
            ax.plot(this_x, this_data, label=[labels[ct]], **plot_args[ct])
            ct += 1


    fig.legend(ax.get_lines(), labels, ncol=3, loc="upper center")
    plt.xticks(this_x, xlabels)
    ax.set_title('exploring KNN # neighbor (loss vs training instance vs noise)')
    ax.set_xlabel('k (num of neighbors ')
    ax.set_ylabel('% accurate')
    with open('./figures/knn_num_neighbors_1.pkl', 'wb') as fid:
        pickle.dump(ax, fid)
    fig.savefig('./figures/knn_num_neighbors_1.png')  #plt.show()
    shit = 0

def reload_figure(filename):
    figx = pickle.load(open(filename, 'rb'))
    plt.show()

    shit = 0

def show(flat_img, imgshape):
    skimage.io.imshow(flat_img._values[:-1].reshape(imgshape))

def show_simple(flat_img, imgshape):
    skimage.io.imshow(flat_img._values.reshape(imgshape))

def decision_tree_testing2(data, csv_file, preprocess=pre_process1, nflag=2):
    # Decision Tree Classifier main
    # Testing test_split_size

    labels = ["training", "testing", "training w/25% noise", "testing w/25% noise"]

    plot_args=[{'c': 'red', 'linestyle': '-'},
                     {'c': 'red', 'linestyle': '--'},
                     {'c': 'blue', 'linestyle': '-'},
                     {'c': 'blue', 'linestyle': '--'}]
    ct1 = 0
    ct2 = 0

    plt.close()

    fig1, ax1 = plt.subplots()

    fig2, ax2 = plt.subplots()

    for j in range(nflag):
        noisy = j/4 # % of data loss.
        print(' ----- INTRODUCING MISSING DATA ', noisy*100, '%')
        if noisy == 0:
            noise = []
        else:
            noise = noisy


        split_acc = []
        best_acc = -9999
        best_split = 0.1
        train_acc = -1
        test_acc = -1
        x_axis = range(10, 50, 5)
        for i in x_axis:
            percentage = i/100
            print('new training set on test split ', percentage*100, '% of 50% | best split = ', best_split, ' train_acc = ', train_acc, ' test_acc = ', test_acc , ' best_acc = ', best_acc)
            dt = decision_tree(data= data, csv_file=csv_file, test_size=percentage, noise=noise, preprocess=preprocess)
            # dt.pre_process1()
            dt.train()
            [train_acc, test_acc, conf_mat] = dt.accuracy()
            split_acc.append([train_acc, test_acc, conf_mat])
            if (np.mean([train_acc,test_acc])) >= best_acc:
                best_acc = np.mean([train_acc,test_acc])
                best_split = percentage

        for i in range(2):
            this_data = [d[i] for d in split_acc]
            ax1.plot(x_axis, this_data, label=[labels[ct1]], **plot_args[ct1])
            ct1 += 1

        print('best split = ', best_split)
        best_split = 0.20
        prune_acc = []
        best_acc = -9999
        best_prune = 1
        train_acc = -1
        test_acc = -1
        x_axis2 = range(1, 20)
        for i in x_axis2:
            print('new training set on max_depth ', i, ' of 30 | best_prune = ', best_prune, ' train_acc = ', train_acc, ' test_acc = ', test_acc, ' best_acc = ', best_acc)
            dt = decision_tree(data= data, csv_file=csv_file, test_size=best_split, max_depth=i, noise=noise, preprocess=preprocess)
            dt.train()
            [train_acc, test_acc, conf_mat] = dt.accuracy()
            prune_acc.append([train_acc, test_acc, conf_mat])
            if (np.mean([train_acc,test_acc])) >= best_acc:
                best_acc = np.mean([train_acc,test_acc])
                best_prune = i

        for i in range(2):
            this_data = [d[i] for d in prune_acc]
            ax2.plot(x_axis2, this_data, label=[labels[ct2]], **plot_args[ct2])
            ct2 += 1

    ax1.set_title('exploring data split w/ Decision Trees (0% noise vs 25% noise)')
    ax1.set_xlabel('data split %')
    ax1.set_ylabel('% accurate')
    fig1.legend(ax1.get_lines(), labels, ncol=3, loc="upper center")
    with open('./figures/decision_tree_data_split1.pkl', 'wb') as fid:
        pickle.dump(ax1, fid)
    fig1.savefig('./figures/decision_tree_data_split1.png')# fig1.show()
    ax2.set_title('exploring pruning w/ Decision Trees (0% noise vs 25% noise)')
    ax2.set_xlabel('tree max depth (pruning)')
    ax2.set_ylabel('% accurate')
    fig2.legend(ax2.get_lines(), labels, ncol=3, loc="upper center")
    with open('./figures/decision_tree_prune1.pkl', 'wb') as fid:
        pickle.dump(ax2, fid)
    fig2.savefig('./figures/decision_tree_prune1.png')#fig2.show()
    # plt.show()

def neural_net_testing2(data, csv_file, preprocess=pre_process1, nflag=2):

    plot_args = [{'c': 'red', 'linestyle': '-'},
                 {'c': 'red', 'linestyle': '--'},
                 {'c': 'red', 'linestyle': '-.'},
                 {'c': 'orange', 'linestyle': '-'},
                 {'c': 'orange', 'linestyle': '--'},
                 {'c': 'orange', 'linestyle': '-.'},
                 {'c': 'blue', 'linestyle': '-'},
                 {'c': 'blue', 'linestyle': '--'},
                 {'c': 'blue', 'linestyle': '-.'},
                 {'c': 'green', 'linestyle': '--'},
                 {'c': 'green', 'linestyle': '-'},
                 {'c': 'green', 'linestyle': '-.'}]

    # ------------------------------------------------- learning_ rate & activation --------------------------------------------------------------
    params = [{'solver': 'adam', 'learning_rate_init': 0.005, 'activation': 'relu', 'hidden_layer_sizes':(100,)},
              {'solver': 'adam', 'learning_rate_init': 0.005, 'activation': 'relu', 'hidden_layer_sizes': (200, )},
              {'solver': 'adam', 'learning_rate_init': 0.005, 'activation': 'relu', 'hidden_layer_sizes': (400, )},
              {'solver': 'adam', 'learning_rate_init': 0.005, 'activation': 'relu', 'hidden_layer_sizes': (800, )},
              {'solver': 'adam', 'learning_rate_init': 0.005, 'activation': 'relu', 'hidden_layer_sizes': (1600, )}]


    plt.close()
    fig, ax = plt.subplots()

    split_acc = []
    noise = []
    mlps = []

    for j in range(nflag):
        noisy = j/4
        print(' ----- INTRODUCING MISSING DATA ', noisy*100, '%')
        if noisy > 0:
            noise = noisy
        all_time = []
        for label, param in zip(labels, params):
            t0 = time()
            print('Trying: ', param)
            nn = neural_net(data= data, csv_file=csv_file, noise=noise, params=param,preprocess=preprocess)
            nn.train()
            [train_acc, test_acc, conf_mat] = nn.accuracy()
            print(' train_acc = ', train_acc,' test_acc = ', test_acc, ' hidden layer used: ', nn.nn.hidden_layer_sizes, ' activation used: ', nn.nn.activation)
            split_acc.append([train_acc, test_acc, conf_mat])
            mlps.append(nn)
            t1 = time()
            all_time.append(t1-t0)

    for mlp, label, args in zip(mlps, labels, plot_args):

        if mlp.nn.solver == 'lbfgs':
            ax.plot(range(mlp.nn.max_iter), [mlp.nn.loss_]*mlp.nn.max_iter, label=label, **args)
        else:
            ax.plot(mlp.nn.loss_curve_, label=label, **args)

    ax.set_title('neural nets learning rate & activation (loss vs training instance)')
    fig.legend(ax.get_lines(), labels, ncol=3, loc="upper center")
    ax.set_xlabel('iteration')
    ax.set_ylabel('data loss')
    with open('./figures/neural_net_lr_activate_1.pkl', 'wb') as fid:
        pickle.dump(ax, fid)
    fig.savefig('./figures/neural_net_lr_activate_1.png')  #plt.show()

    fig2, ax2 = plt.subplots()

    this_x = range(len(all_time))
    ax2.plot(this_x, all_time)
    plt.xticks(this_x, labels)
    ax2.set_title('nnlearning times')
    ax2.set_xlabel('params used')
    ax2.set_ylabel('times (s)')
    with open('./figures/nn_times.pkl', 'wb') as fid:
        pickle.dump(ax2, fid)
    fig2.savefig('./figures/nn_times.png')  #plt.show()

def boosting_testing2(data, csv_file,preprocess=pre_process1, nflag=2):
    # Decision Tree Classifier main
    # Testing test_split_size

    labels = ["training", "testing", "training w/25% noise", "testing w/25% noise"]

    plot_args=[{'c': 'red', 'linestyle': '-'},
                     {'c': 'red', 'linestyle': '--'},
                     {'c': 'blue', 'linestyle': '-'},
                     {'c': 'blue', 'linestyle': '--'}]
    ct1 = 0
    ct2 = 0
    ct3 = 0

    plt.close()

    fig1, ax1 = plt.subplots()

    fig2, ax2 = plt.subplots()

    fig3, ax3 = plt.subplots()

    for j in range(nflag): # NOISE WILL never be on.
        noisy = j/4 # % of data loss.
        print(' ----- INTRODUCING MISSING DATA ', noisy*100, '%')
        if noisy == 0:
            noise = []
        else:
            noise = noisy
        best_split = 0.2

        lr_acc = []
        best_acc = -9999
        best_lr = 1
        train_acc = -1
        test_acc = -1
        xaxis1 = range(5)
        lr_axis = 1/np.power(10,range(5))
        for i in xaxis1:
            lr = 1/np.power(10,i)
            print('new training set on learning_rate ', lr, ' | best learnrate = ', best_lr, ' train_acc = ', train_acc, ' test_acc = ', test_acc , ' best_acc = ', best_acc)
            dt = decision_tree(data= data, csv_file=csv_file, test_size=best_split, noise=noise, boosting=True, learning_rate=lr,preprocess=preprocess)
            # dt.pre_process1()
            dt.train()
            [train_acc, test_acc, conf_mat] = dt.accuracy()
            lr_acc.append([train_acc, test_acc, conf_mat])
            if (np.mean([train_acc,test_acc])) >= best_acc:
                best_acc = np.mean([train_acc,test_acc])
                best_lr = lr

        for i in range(2):
            this_data = [d[i] for d in lr_acc]
            ax1.plot(xaxis1, this_data, label=[labels[ct1]], **plot_args[ct1])
            ct1 += 1

        ax1.set_title('exploring learning rate w/ Boosting')
        ax1.set_xlabel('learning rate')
        ax1.set_ylabel('% accurate')
        fig1.legend(ax1.get_lines(), labels, ncol=3, loc="upper center")
        with open('./figures/boosting_lr_1.pkl', 'wb') as fid:
            pickle.dump(ax1, fid)
        fig1.savefig('./figures/boosting_lr_1.png')  # fig1.show()

        n_est_acc=[]
        best_acc = -9999
        best_ne = 1
        train_acc = -1
        test_acc = -1
        xaxis2 = range(5)
        ne_axis = 5*(np.add(xaxis2,1))
        for i in xaxis2:
            n_estimators = np.int(5*(i+1))
            print('new training set on num_estimators ', n_estimators, ' | best num_estimators = ', best_ne, ' train_acc = ', train_acc, ' test_acc = ', test_acc , ' best_acc = ', best_acc)
            dt = decision_tree(data= data, csv_file=csv_file, test_size=best_split, noise=noise, boosting=True, learning_rate=best_lr, n_estimators=n_estimators,preprocess=preprocess)
            # dt.pre_process1()
            dt.train()
            [train_acc, test_acc, conf_mat] = dt.accuracy()
            n_est_acc.append([train_acc, test_acc, conf_mat])
            if (np.mean([train_acc,test_acc])) >= best_acc:
                best_acc = np.mean([train_acc,test_acc])
                best_ne = n_estimators

        for i in range(2):
            this_data = [d[i] for d in n_est_acc]
            ax2.plot(ne_axis, this_data, label=[labels[ct2]], **plot_args[ct2])
            ct2 += 1

        ax2.set_title('exploring # of estimators w/ Boosting (0% noise vs 25% noise)')
        ax2.set_xlabel('# of estimators')
        ax2.set_ylabel('% accurate')
        fig2.legend(ax2.get_lines(), labels, ncol=3, loc="upper center")
        with open('./figures/boosting_num_est_1.pkl', 'wb') as fid:
            pickle.dump(ax2, fid)
        fig2.savefig('./figures/boosting_num_est_1.png')  # fig2.show()

        # print('best split = ', best_split)
        # PRUNING
        prune_acc = []
        best_acc = -9999
        best_prune = 1
        train_acc = -1
        test_acc = -1
        xaxis3=range(1, 10)
        for i in xaxis3:
            print('new training set on max_depth ', i, ' of 30 | (lr=', best_lr, 'ne=', best_ne, ') | best_prune = ', best_prune, ' train_acc = ', train_acc, ' test_acc = ', test_acc, ' best_acc = ', best_acc)
            dt = decision_tree(data= data, csv_file=csv_file, test_size=best_split, max_depth=i, noise=noise, boosting=True, learning_rate=best_lr, n_estimators=best_ne,preprocess=preprocess)
            dt.train()
            [train_acc, test_acc, conf_mat] = dt.accuracy()
            prune_acc.append([train_acc, test_acc, conf_mat])
            if (np.mean([train_acc,test_acc])) >= best_acc:
                best_acc = np.mean([train_acc,test_acc])
                best_prune = i

        for i in range(2):
            this_data = [d[i] for d in prune_acc]
            ax3.plot(xaxis3, this_data, label=[labels[ct3]], **plot_args[ct3])
            ct3 += 1

        ax3.set_title('exploring pruning w/ Boosting (0% noise vs 25% noise)')
        ax3.set_xlabel('tree max depth (pruning)')
        ax3.set_ylabel('% accurate')
        fig3.legend(ax3.get_lines(), labels, ncol=3, loc="upper center")
        with open('./figures/boosting_prune_1.pkl', 'wb') as fid:
            pickle.dump(ax3, fid)
        fig3.savefig('./figures/boosting_prune_1.png')  #fig3.show()
        # plt.show()

def svm_testing2(data, csv_file, preprocess=pre_process1, nflag=2):
    plt.close()
    fig, ax = plt.subplots()

    params = [{'kernel': 'poly', 'degree': 2},
              {'kernel': 'poly', 'degree': 5},
              {'kernel': 'poly', 'degree': 8},
              #{'kernel': 'poly', 'degree': 10},
              {'kernel': 'rbf'},
              {'kernel': 'sigmoid'}]

    labels = ["training", "testing", "training w/25% noise", "testing w/25% noise"]

    xlabels = ["poly deg 2", "poly deg 5", "poly deg 8", "sigmoid", "gaussian" ]

    plot_args=[{'c': 'red', 'linestyle': '-'},
                     {'c': 'red', 'linestyle': '--'},
                     {'c': 'blue', 'linestyle': '-'},
                     {'c': 'blue', 'linestyle': '--'}]
    mlps = []
    ct = 0
    for j in range(nflag):
        split_acc = []
        noise = []

        noisy = j/4

        # labels = ["poly deg 2 w/ noise "+str(noisy), "poly deg 5 w/ noise "+str(noisy), "poly deg 8 w/ noise "+str(noisy), "poly deg 10 w/ noise "+str(noisy), "gaussian w/ noise "+str(noisy), "sigmoid w/ noise "+str(noisy)]

        print(' ----- INTRODUCING MISSING DATA ', noisy*100, '%')
        if noisy > 0:
            noise = noisy

        for param in params:
            print('Trying: ', param)
            svm = SVM(data= data, csv_file=csv_file, noise=noise, params=param,preprocess=preprocess)
            svm.train()
            [train_acc, test_acc, conf_mat] = svm.accuracy()
            print(' train_acc = ', train_acc,' test_acc = ', test_acc, ' params used: ', param)
            split_acc.append([train_acc, test_acc, conf_mat])
            mlps.append(svm)

        for i in range(2):
            this_data = [d[i] for d in split_acc]
            this_x = range(len(this_data))
            ax.plot(this_x, this_data, label=[labels[ct]], **plot_args[ct])
            ct += 1


    fig.legend(ax.get_lines(), labels, ncol=3, loc="upper center")
    plt.xticks(this_x, xlabels)
    ax.set_title('exploring kernels w/ SVN (0% noise vs 25% noise)')
    ax.set_xlabel('kernel used')
    ax.set_ylabel('% accurate')
    with open('./figures/svn_kernels_1.pkl', 'wb') as fid:
        pickle.dump(ax, fid)
    fig.savefig('./figures/svn_kernels_1.png')  #plt.show()
    shit = 0

def knn_testing2(data, csv_file,preprocess=pre_process1, nflag=2):

    params = [{'n_neighbors': 3},
              {'n_neighbors': 5},
              {'n_neighbors': 7},
              {'n_neighbors': 9},
              {'n_neighbors': 11}]

    labels = ["training", "testing", "training w/25% noise", "testing w/25% noise"]

    xlabels = ["k = 3", "k = 5", "k = 7", "k = 9", "k = 11"]

    plot_args=[{'c': 'red', 'linestyle': '-'},
                     {'c': 'red', 'linestyle': '--'},
                     {'c': 'blue', 'linestyle': '-'},
                     {'c': 'blue', 'linestyle': '--'}]
    mlps = []
    ct = 0

    plt.close()
    fig, ax = plt.subplots()
    for j in range(nflag):
        split_acc = []
        noise = []

        noisy = j/4

        # labels = ["poly deg 2 w/ noise "+str(noisy), "poly deg 5 w/ noise "+str(noisy), "poly deg 8 w/ noise "+str(noisy), "poly deg 10 w/ noise "+str(noisy), "gaussian w/ noise "+str(noisy), "sigmoid w/ noise "+str(noisy)]

        print(' ----- INTRODUCING MISSING DATA ', noisy*100, '%')
        if noisy > 0:
            noise = noisy

        for param in params:
            print('Trying: ', param)
            svm = KNN(data= data, csv_file=csv_file, noise=noise, params=param,preprocess=preprocess)
            svm.train()
            [train_acc, test_acc, conf_mat] = svm.accuracy()
            print(' train_acc = ', train_acc,' test_acc = ', test_acc, ' params used: ', param)
            split_acc.append([train_acc, test_acc, conf_mat])
            mlps.append(svm)

        for i in range(2):
            this_data = [d[i] for d in split_acc]
            this_x = range(len(this_data))
            ax.plot(this_x, this_data, label=[labels[ct]], **plot_args[ct])
            ct += 1


    fig.legend(ax.get_lines(), labels, ncol=3, loc="upper center")
    plt.xticks(this_x, xlabels)
    ax.set_title('exploring KNN # neighbor (loss vs training instance vs noise)')
    ax.set_xlabel('k (num of neighbors ')
    ax.set_ylabel('% accurate')
    with open('./figures/knn_num_neighbors_1.pkl', 'wb') as fid:
        pickle.dump(ax, fid)
    fig.savefig('./figures/knn_num_neighbors_1.png')  #plt.show()
    shit = 0

def main():

    fix=False
    if fix:
        # filename = './figures/diabetes/neural_net_lr_activate_1.pkl'
        # reload_figure(filename)

        filename = './figures/diabetes/neural_net_solver_1.pkl'
        reload_figure(filename)

        filename = './figures/diabetes/neural_net_hidden_layer_1.pkl'
        reload_figure(filename)

    reload = True
    d1 = False
    d2 = False
    d3 = False
    d4 = True
    if d1:
        dataset1 = "diabetes_data_upload.csv" # https://archive.ics.uci.edu/ml/datasets/Early+stage+diabetes+risk+prediction+dataset
        print(' ----------------------------------- Decision Tree Data set 1 --------------------------------------')
        decision_tree_testing(data= [], csv_file=dataset1, preprocess= pre_process1, nflag=2)
        print(' ----------------------------------- Neural Net Data set 1 --------------------------------------')
        neural_net_testing(data= [], csv_file=dataset1, preprocess= pre_process1, nflag=2)
        print(' ----------------------------------- BOOSTING Data set 1 --------------------------------------')
        boosting_testing(data= [], csv_file=dataset1, preprocess= pre_process1, nflag=2)
        print(' ----------------------------------- Super Vector Machine Data set 1 --------------------------------------')
        svm_testing(data= [], csv_file=dataset1, preprocess= pre_process1, nflag=2)
        print(' ----------------------------------- K Neighbors Data set 1 --------------------------------------')
        knn_testing(data= [], csv_file=dataset1, preprocess= pre_process1, nflag=2)
    if d2:
        dataset2 = "mushrooms.csv"  # https://www.kaggle.com/uciml/mushroom-classification#
        print(' ----------------------------------- Decision Tree Data set 1 --------------------------------------')
        decision_tree_testing(data= [], csv_file=dataset2, preprocess= pre_process2, nflag=2)
        print(' ----------------------------------- Neural Net Data set 1 --------------------------------------')
        neural_net_testing(data= [], csv_file=dataset2, preprocess= pre_process2, nflag=2)
        print(' ----------------------------------- BOOSTING Data set 1 --------------------------------------')
        boosting_testing(data= [], csv_file=dataset2, preprocess= pre_process2, nflag=2)
        print(' ----------------------------------- Super Vector Machine Data set 1 --------------------------------------')
        svm_testing(data= [], csv_file=dataset2, preprocess= pre_process2, nflag=2)
        print(' ----------------------------------- K Neighbors Data set 1 --------------------------------------')
        knn_testing(data= [], csv_file=dataset2, preprocess= pre_process2, nflag=2)
    if d3:
        data_loc = "./REEF_DATASET/EILAT2/"
        # data_loc = "./REEF_DATASET/RSMAS/"
        if os.path.isfile('eilat.pkl'):
            with open('eilat.pk1', 'rb') as f:
                [data, label, hog_d]= pickle.load(f)
        else:
            data,label,hog_d = pre_process3(data_loc, [])
            with open('eilat.pk1', 'wb') as f:
                pickle.dump([data, label, hog_d], f)

        print(' ----------------------------------- Decision Tree Data set 1 --------------------------------------')
        decision_tree_testing2(data= (data, label), csv_file=[], preprocess= [], nflag=1)
        print(' ----------------------------------- Neural Net Data set 1 --------------------------------------')
        neural_net_testing2(data= (data, label), csv_file=[], preprocess= [], nflag=1)
        print(' ----------------------------------- BOOSTING Data set 1 --------------------------------------')
        boosting_testing2(data= (data, label), csv_file=[], preprocess= [], nflag=1)
        print(' ----------------------------------- Super Vector Machine Data set 1 --------------------------------------')
        svm_testing2(data= (data, label), csv_file=[], preprocess= [], nflag=1)
        print(' ----------------------------------- K Neighbors Data set 1 --------------------------------------')
        knn_testing2(data= (data, label), csv_file=[], preprocess= [], nflag=1)

        print('-----------------------------------------USING HOG -------------------------------------------')

        print(' ----------------------------------- Decision Tree Data set 1 --------------------------------------')
        decision_tree_testing2(data= (hog_d, label), csv_file=[], preprocess= [], nflag=1)
        print(' ----------------------------------- Neural Net Data set 1 --------------------------------------')
        neural_net_testing2(data= (hog_d, label), csv_file=[], preprocess= [], nflag=1)
        print(' ----------------------------------- BOOSTING Data set 1 --------------------------------------')
        boosting_testing2(data= (hog_d, label), csv_file=[], preprocess= [], nflag=1)
        print(' ----------------------------------- Super Vector Machine Data set 1 --------------------------------------')
        svm_testing2(data= (hog_d, label), csv_file=[], preprocess= [], nflag=1)
        print(' ----------------------------------- K Neighbors Data set 1 --------------------------------------')
        knn_testing2(data= (hog_d, label), csv_file=[], preprocess= [], nflag=1)
    if d4:
        normal = True
        uhog = False
        data_loc = "./REEF_DATASET/EILAT2/"
        # data_loc = "./REEF_DATASET/RSMAS/"
        if os.path.isfile('./eilat.pkl') and not reload:
            with open('eilat.pk1', 'rb') as f:
                [data, label, hog_d]= pickle.load(f)
        else:
            data,label,hog_d = pre_process3(data_loc, [])
            with open('eilat.pkl', 'wb') as f:
                pickle.dump([data, label, hog_d], f)
        if normal:
            print(' ----------------------------------- Decision Tree Data set 1 --------------------------------------')
            # decision_tree_testing(data= (data,label), csv_file=[], preprocess= [], nflag=1)
            print(' ----------------------------------- Neural Net Data set 1 --------------------------------------')
            neural_net_testing2(data= (data,label), csv_file=[], preprocess= [], nflag=1)
            print(' ----------------------------------- BOOSTING Data set 1 --------------------------------------')
            boosting_testing(data= (data,label), csv_file=[], preprocess= [], nflag=1)
            print(' ----------------------------------- Super Vector Machine Data set 1 --------------------------------------')
            svm_testing(data= (data,label), csv_file=[], preprocess= [], nflag=1)
            print(' ----------------------------------- K Neighbors Data set 1 --------------------------------------')
            knn_testing(data= (data,label), csv_file=[], preprocess= [], nflag=1)

        if uhog:
            print('-----------------------------------------USING HOG -------------------------------------------')

            print(' ----------------------------------- Decision Tree Data set 1 --------------------------------------')
            decision_tree_testing(data= (hog_d,label), csv_file=[], preprocess= [], nflag=1)
            print(' ----------------------------------- Neural Net Data set 1 --------------------------------------')
            neural_net_testing(data= (hog_d,label), csv_file=[], preprocess= [], nflag=1)
            print(' ----------------------------------- BOOSTING Data set 1 --------------------------------------')
            boosting_testing(data= (hog_d,label), csv_file=[], preprocess= [], nflag=1)
            print(' ----------------------------------- Super Vector Machine Data set 1 --------------------------------------')
            svm_testing(data= (hog_d,label), csv_file=[], preprocess= [], nflag=1)
            print(' ----------------------------------- K Neighbors Data set 1 --------------------------------------')
            knn_testing(data= (hog_d,label), csv_file=[], preprocess= [], nflag=1)



if __name__ == "__main__":
    main()



