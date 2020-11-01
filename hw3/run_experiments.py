# Script to run experiments

import os
import pickle
import numpy as np
import pandas as pd

from skimage.color import rgb2gray
from skimage import filters
import skimage.io
import skimage.viewer
from time import time
from skimage.feature import hog, blob_dog, canny

from clustering import KMeansClustering, MixtureOfGaussians
from dimensionality_reduction import IndependentComponents, KernelPrincipalComponents, \
                                     PrincipalComponents, RandomProjections
from neural_networks import NeuralNetwork

from sklearn.datasets import load_breast_cancer, fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

IMAGE_DIR = 'images/'  # output images directory

def pre_process1(csv_file, noise):
    # https://stackoverflow.c"'
    # x, y, labels, features = data.data, data.target, data.target_names, data.feature_names
    data = pd.read_csv(csv_file)
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

    vals = data[headers[:-1]].stack().drop_duplicates().values
    b = [x for x in data[headers[:-1]].stack().drop_duplicates().rank(method='dense')]
    dictionary = dict(zip(b, vals))  # dictionary for digitization.

    stacked = data[headers[:-1]].stack()
    data[headers[:-1]] = pd.Series(stacked.factorize()[0], index=stacked.index).unstack()

    # class_vals = data[headers[-1]].drop_duplicates().values
    # b = data[headers[-1]].drop_duplicates().rank(method='dense')
    data[headers[-1]] = pd.Series(data[headers[-1]].factorize()[0])

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
    sobel_data = []
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
                sobel_img = filters.sobel(rgb2gray(img))
                shit = sobel_img
                shit[sobel_img >= np.mean(sobel_img)] = 1.0
                shit[sobel_img < np.mean(sobel_img)] = 0.0
                hog_data.append(np.append( hog_image, label_count))
                sobel_data.append(np.append(shit.flatten(), label_count))
                label_data.append(np.append(img.flatten(), label_count))
        # data.append(label_data)
        label_count += 1

    data = pd.DataFrame(np.asarray(label_data))
    thishog = pd.DataFrame(np.asarray(hog_data))
    thissobel = pd.DataFrame(np.asarray(sobel_data))
    return data, labels, thishog, thissobel

def load_dataset(split_percentage=0.2, dataset1=[], data=load_breast_cancer()):
    """Load WDBC dataset.

        Args:
           split_percentage (float): validation split.

        Returns:
           x_train (ndarray): training data.
           x_test (ndarray): test data.
           y_train (ndarray): training labels.
           y_test (ndarray): test labels.
        """

    # Loadt dataset and split in training and validation sets, preserving classes representation

    if not dataset1:
        x, y, labels, features = data.data, data.target, data.target_names, data.feature_names
    else:
        if os.path.isdir(dataset1):
            if os.path.isfile('eilat.pkl'):
                with open('eilat.pk1', 'rb') as f:
                    [data, label, hog_d,sobel_d]= pickle.load(f)
            else:
                data,label,hog_d,sobel_d = pre_process3(dataset1, [])
                with open('eilat.pk1', 'wb') as f:
                    pickle.dump([data, label, hog_d,sobel_d], f)
        else:
            # dataset1 = "diabetes_data_upload.csv"
            data, dictionary = pre_process1(dataset1, noise=False)
        x = data.iloc[:, 0:-1]
        y = np.asarray(data.iloc[:, -1])

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=split_percentage, shuffle=True, random_state=28, stratify=y)

    # Normalize feature data
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    print('\nTotal dataset size:')
    print('Number of instances: {}'.format(x.shape[0]))
    print('Number of features: {}'.format(x.shape[1]))
    # print('Number of classes: {}'.format(len(labels)))
    print('Training Set : {}'.format(x_train.shape))
    print('Testing Set : {}'.format(x_test.shape))

    return x_train, x_test, y_train, y_test

def clustering(x_train, x_test, y_train, **kwargs):
    """Perform clustering experiment.

        Args:
           x_train (ndarray): training data.
           x_test (ndarray): test data.
           y_train (ndarray): training labels.
           kwargs (dict): additional arguments to pass:
                    - dataset (string): dataset, WDBC or MNIST.
                    - perform_model_complexity (bool): True if we want to perform model complexity.
                    - kmeans_n_clusters (int): number of clusters for k-Means.
                    - gmm_n_clusters (int): number of clusters for Gaussians Mixture Models.
                    - gmm_covariance (string): covariance type for Gaussians Mixture Models.

        Returns:
           kmeans_clusters (ndarray): k-Means cluster on training and test data.
           gmm_clusters (ndarray): Gaussian Mixture Models cluster on training and test data.
        """

    print('\n--------------------------')
    print('kMeans')

    # Declare kMeans, perform experiments and get clusters on training data
    kmeans = KMeansClustering(n_clusters=kwargs['kmeans_n_clusters'], max_n_clusters=10)
    kmeans_clusters = kmeans.experiment(x_train, x_test, y_train,
                                        dataset=kwargs['dataset'],
                                        perform_model_complexity=kwargs['perform_model_complexity'])
    # kmeans.plot_model_complexity(x_train, kwargs['dataset'])
    print('\n--------------------------')
    print('GMM')

    # Declare Gaussian Mixtures Models, perform experiments and get clusters on training data
    gmm = MixtureOfGaussians(n_clusters=kwargs['gmm_n_clusters'], covariance=kwargs['gmm_covariance'], max_n_clusters=10)
    gmm_clusters = gmm.experiment(x_train, x_test, y_train,
                                  dataset=kwargs['dataset'],
                                  perform_model_complexity=kwargs['perform_model_complexity'])
    # gmm.plot_model_complexity(x_train,kwargs['dataset'])
    return kmeans_clusters, gmm_clusters


def dimensionality_reduction(x_train, x_test, y_train, **kwargs):
    """Perform dimensionality reduction experiment.

        Args:
           x_train (ndarray): training data.
           x_test (ndarray): test data.
           y_train (ndarray): training labels.
           kwargs (dict): additional arguments to pass:
                    - dataset (string): dataset, WDBC or MNIST.
                    - perform_model_complexity (bool): True if we want to perform model complexity.
                    - pca_n_components (int): number of components for PCA.
                    - pca_kmeans_n_clusters (int): number of clusters for PCA + k-Means.
                    - pca_gmm_n_clusters (int): number of clusters for PCA + Gaussians Mixtures Models.
                    - pca_gmm_covariance (string): covariance type for PCA + Gaussians Mixture Models.
                    - ica_n_components (int): number of components for ICA.
                    - ica_kmeans_n_clusters (int): number of clusters for ICA + k-Means.
                    - ica_gmm_n_clusters (int): number of clusters for ICA + Gaussians Mixtures Models.
                    - ica_gmm_covariance (string): covariance type for ICA + Gaussians Mixture Models.
                    - kpca_n_components (int): number of components for KPCA.
                    - kpca_kmeans_n_clusters (int): number of clusters for KPCA + k-Means.
                    - kpca_gmm_n_clusters (int): number of clusters for KPCA + Gaussians Mixtures Models.
                    - kpca_gmm_covariance (string): covariance type for KPCA + Gaussians Mixture Models.
                    - rp_n_components (int): number of components for RP.
                    - rp_kmeans_n_clusters (int): number of clusters for RP + k-Means.
                    - rp_gmm_n_clusters (int): number of clusters for RP + Gaussians Mixtures Models.
                    - rp_gmm_covariance (string): covariance type for RP + Gaussians Mixture Models.

        Returns:
           x_pca (ndarray): reduced dataset by PCA.
           x_ica (ndarray): reduced dataset by ICA.
           x_kpca (ndarray): reduced dataset by KPCA.
           x_rp (ndarray): reduced dataset by RP.
        """

    print('\n--------------------------')
    print('PCA')
    print('--------------------------')

    # Declare PCA, perform experiments by reducing the dataset and perform clustering experiments on it
    thisname = kwargs['dataset'] + '_pca_reduced'
    pca = PrincipalComponents(n_components=kwargs['pca_n_components'])
    x_pca = pca.experiment(x_train, x_test, y_train,
                           dataset=thisname,
                           perform_model_complexity=kwargs['perform_model_complexity'])
    # pca.plot_model_complexity(x_test,kwargs['dataset'])
    pca_kmeans, pca_gmm = clustering(x_pca[0],  x_pca[1], y_train,
                               dataset=thisname,
                               kmeans_n_clusters=kwargs['pca_kmeans_n_clusters'],
                               gmm_n_clusters=kwargs['pca_gmm_n_clusters'], gmm_covariance=kwargs['pca_gmm_covariance'],
                               perform_model_complexity=kwargs['perform_model_complexity'])

    print('\n--------------------------')
    print('ICA')
    print('--------------------------')

    # Declare ICA, perform experiments by reducing the dataset and perform clustering experiments on it
    thisname = kwargs['dataset'] + '_ica_reduced'
    ica = IndependentComponents(n_components=kwargs['ica_n_components'])
    x_ica = ica.experiment(x_train, x_test, y_train,
                           dataset=thisname,
                           perform_model_complexity=kwargs['perform_model_complexity'])
    # ica.plot_model_complexity(x_train, thisname)
    ica_kmeans, ica_gmm = clustering(x_ica[0],  x_ica[1], y_train,
                               dataset=thisname,
                               kmeans_n_clusters=kwargs['ica_kmeans_n_clusters'],
                               gmm_n_clusters=kwargs['ica_gmm_n_clusters'], gmm_covariance=kwargs['ica_gmm_covariance'],
                               perform_model_complexity=kwargs['perform_model_complexity'])

    print('\n--------------------------')
    print('KPCA')
    print('--------------------------')

    # Declare KPCA, perform experiments by reducing the dataset and perform clustering experiments on it
    thisname = kwargs['dataset'] + '_kpca_reduced'
    kpca = KernelPrincipalComponents(n_components=kwargs['kpca_n_components'], kernel=kwargs['kpca_kernel'])
    x_kpca = kpca.experiment(x_train, x_test, y_train,
                             dataset=thisname,
                             perform_model_complexity=kwargs['perform_model_complexity'])
    # kpca.plot_model_complexity(x_train, thisname)
    kpca_kmeans, kpca_gmm = clustering(x_kpca[0], x_kpca[1], y_train,
                               dataset=thisname,
                               kmeans_n_clusters=kwargs['kpca_kmeans_n_clusters'],
                               gmm_n_clusters=kwargs['kpca_gmm_n_clusters'], gmm_covariance=kwargs['kpca_gmm_covariance'],
                               perform_model_complexity=kwargs['perform_model_complexity'])

    print('\n--------------------------')
    print('RP')
    print('--------------------------')

    # Declare RP, perform experiments by reducing the dataset and perform clustering experiments on it
    thisname = kwargs['dataset'] + '_rp_reduced'
    rp = RandomProjections(n_components=kwargs['rp_n_components'])
    x_rp = rp.experiment(x_train, x_test, y_train,
                         dataset=thisname,
                         perform_model_complexity=kwargs['perform_model_complexity'])
    # rp.plot_model_complexity(x_train,thisname)
    rp_kmeans, rp_gmm = clustering(x_rp[0], x_rp[1], y_train,
                           dataset=thisname,
                           kmeans_n_clusters=kwargs['rp_kmeans_n_clusters'],
                           gmm_n_clusters=kwargs['rp_gmm_n_clusters'], gmm_covariance=kwargs['rp_gmm_covariance'],
                           perform_model_complexity=kwargs['perform_model_complexity'])

    return x_pca, x_ica, x_kpca, x_rp, pca_kmeans, pca_gmm, ica_kmeans, ica_gmm, kpca_kmeans, kpca_gmm, rp_kmeans, rp_gmm


def neural_network(x_train, x_test, y_train, y_test,
                   x_pca, x_ica, x_kpca, x_rp,
                   x_kmeans, x_gmm, pca_kmeans, pca_gmm, ica_kmeans, ica_gmm, kpca_kmeans, kpca_gmm, rp_kmeans, rp_gmm, **kwargs):
    """Perform neural network experiment.

        Args:
           x_train (ndarray): training data.
           x_test (ndarray): test data.
           y_train (ndarray): training labels.
           y_test (ndarray): test labels.
           x_pca (ndarray): reduced dataset by PCA.
           x_ica (ndarray): reduced dataset by ICA.
           x_kpca (ndarray): reduced dataset by KPCA.
           x_rp (ndarray): reduced dataset by RP.
           x_kmeans (ndarray): clusters produced by k-Means.
           x_gmm (ndarray): clusters produced by Gaussian Mixture Models.
           kwargs (dict): additional arguments to pass:
                    - layer1_nodes (int): number of neurons in first layer.
                    - layer2_nodes (int): number of neurons in second layer.
                    - learning_rate (float): learning rate.

        Returns:
           None.
        """

    print('\n--------------------------')
    print('NN')
    print('--------------------------')

    # Declare Neural Network and perform experiments on the original dataset
    nn = NeuralNetwork(layer1_nodes=kwargs['layer1_nodes'],
                       layer2_nodes=kwargs['layer2_nodes'],
                       learning_rate=kwargs['learning_rate'])
    nn.experiment(x_train, x_test, y_train, y_test)

    print('\n--------------------------')
    print('PCA + NN')
    print('--------------------------')

    # Declare Neural Network and perform experiments on the reduced dataset by PCA
    nn = NeuralNetwork(layer1_nodes=kwargs['layer1_nodes'],
                       layer2_nodes=kwargs['layer2_nodes'],
                       learning_rate=kwargs['learning_rate'])
    nn.experiment(x_pca[0], x_pca[1], y_train, y_test)

    print('\n--------------------------')
    print('ICA + NN')
    print('--------------------------')

    # Declare Neural Network and perform experiments on the reduced dataset by ICA
    nn = NeuralNetwork(layer1_nodes=kwargs['layer1_nodes'],
                       layer2_nodes=kwargs['layer2_nodes'],
                       learning_rate=kwargs['learning_rate'])
    nn.experiment(x_ica[0], x_ica[1], y_train, y_test)

    print('\n--------------------------')
    print('KPCA + NN')
    print('--------------------------')

    # Declare Neural Network and perform experiments on the reduced dataset by KPCA
    nn = NeuralNetwork(layer1_nodes=kwargs['layer1_nodes'],
                       layer2_nodes=kwargs['layer2_nodes'],
                       learning_rate=kwargs['learning_rate'])
    nn.experiment(x_kpca[0], x_kpca[1], y_train, y_test)

    print('\n--------------------------')
    print('RP+ NN')
    print('--------------------------')

    # Declare Neural Network and perform experiments on the reduced dataset by RP
    nn = NeuralNetwork(layer1_nodes=kwargs['layer1_nodes'],
                       layer2_nodes=kwargs['layer2_nodes'],
                       learning_rate=kwargs['learning_rate'])
    nn.experiment(x_rp[0], x_rp[1], y_train, y_test)

    print('\n--------------------------')
    print('KMEANS+ NN')
    print('--------------------------')

    # Declare Neural Network
    nn = NeuralNetwork(layer1_nodes=kwargs['layer1_nodes'],
                       layer2_nodes=kwargs['layer2_nodes'],
                       learning_rate=kwargs['learning_rate'])

    # Augment the original dataset by adding clusters produced by k-Means as features
    x_kmeans_normalized = (x_kmeans[0] - np.mean(x_kmeans[0])) / np.std(x_kmeans[0])
    x_kmeans_normalized = np.expand_dims(x_kmeans_normalized, axis=1)
    x_train_new = np.append(x_train, x_kmeans_normalized, axis=1)
    x_kmeans_normalized = (x_kmeans[1] - np.mean(x_kmeans[1])) / np.std(x_kmeans[1])
    x_kmeans_normalized = np.expand_dims(x_kmeans_normalized, axis=1)
    x_test_new = np.append(x_test, x_kmeans_normalized, axis=1)

    # Perform experiments on it
    nn.experiment(x_train_new, x_test_new, y_train, y_test)

    print('\n--------------------------')
    print('GMM+ NN')
    print('--------------------------')

    # Declare Neural Network
    nn = NeuralNetwork(layer1_nodes=kwargs['layer1_nodes'],
                       layer2_nodes=kwargs['layer2_nodes'],
                       learning_rate=kwargs['learning_rate'])

    # Augment the original dataset by adding clusters produced by Gaussian Mixture Models as features
    x_gmm_normalized = (x_gmm[0] - np.mean(x_gmm[0])) / np.std(x_gmm[0])
    x_gmm_normalized = np.expand_dims(x_gmm_normalized, axis=1)
    x_train_new = np.append(x_train, x_gmm_normalized, axis=1)
    x_gmm_normalized = (x_gmm[1] - np.mean(x_gmm[1])) / np.std(x_gmm[1])
    x_gmm_normalized = np.expand_dims(x_gmm_normalized, axis=1)
    x_test_new = np.append(x_test, x_gmm_normalized, axis=1)

    # Perform experiments on it
    nn.experiment(x_train_new, x_test_new, y_train, y_test)

    print('\n--------------------------')
    print('PCA + KMEANS+ NN')
    print('--------------------------')

    # Declare Neural Network
    nn = NeuralNetwork(layer1_nodes=kwargs['layer1_nodes'],
                       layer2_nodes=kwargs['layer2_nodes'],
                       learning_rate=kwargs['learning_rate'])

    # Augment the original dataset by adding clusters produced by k-Means as features
    pca_kmeans_normalized = (pca_kmeans[0] - np.mean(pca_kmeans[0])) / np.std(pca_kmeans[0])
    pca_kmeans_normalized = np.expand_dims(pca_kmeans_normalized, axis=1)
    pca_kmeans_train_new = np.append(x_train, pca_kmeans_normalized, axis=1)
    pca_kmeans_normalized = (pca_kmeans[1] - np.mean(pca_kmeans[1])) / np.std(pca_kmeans[1])
    pca_kmeans_normalized = np.expand_dims(pca_kmeans_normalized, axis=1)
    pca_kmeans_test_new = np.append(x_test, pca_kmeans_normalized, axis=1)

    # Perform experiments on it
    nn.experiment(pca_kmeans_train_new, pca_kmeans_test_new, y_train, y_test)

    print('\n--------------------------')
    print('ICA + KMEANS+ NN')
    print('--------------------------')

    # Declare Neural Network
    nn = NeuralNetwork(layer1_nodes=kwargs['layer1_nodes'],
                       layer2_nodes=kwargs['layer2_nodes'],
                       learning_rate=kwargs['learning_rate'])

    # Augment the original dataset by adding clusters produced by k-Means as features
    ica_kmeans_normalized = (ica_kmeans[0] - np.mean(ica_kmeans[0])) / np.std(ica_kmeans[0])
    ica_kmeans_normalized = np.expand_dims(ica_kmeans_normalized, axis=1)
    ica_kmeans_train_new = np.append(x_train, ica_kmeans_normalized, axis=1)
    ica_kmeans_normalized = (ica_kmeans[1] - np.mean(ica_kmeans[1])) / np.std(ica_kmeans[1])
    ica_kmeans_normalized = np.expand_dims(ica_kmeans_normalized, axis=1)
    ica_kmeans_test_new = np.append(x_test, ica_kmeans_normalized, axis=1)

    # Perform experiments on it
    nn.experiment(ica_kmeans_train_new, ica_kmeans_test_new, y_train, y_test)

    print('\n--------------------------')
    print('KPCA + KMEANS+ NN')
    print('--------------------------')

    # Declare Neural Network
    nn = NeuralNetwork(layer1_nodes=kwargs['layer1_nodes'],
                       layer2_nodes=kwargs['layer2_nodes'],
                       learning_rate=kwargs['learning_rate'])

    # Augment the original dataset by adding clusters produced by k-Means as features
    kpca_kmeans_normalized = (kpca_kmeans[0] - np.mean(kpca_kmeans[0])) / np.std(kpca_kmeans[0])
    kpca_kmeans_normalized = np.expand_dims(kpca_kmeans_normalized, axis=1)
    kpca_kmeans_train_new = np.append(x_train, kpca_kmeans_normalized, axis=1)
    kpca_kmeans_normalized = (kpca_kmeans[1] - np.mean(kpca_kmeans[1])) / np.std(kpca_kmeans[1])
    kpca_kmeans_normalized = np.expand_dims(kpca_kmeans_normalized, axis=1)
    kpca_kmeans_test_new = np.append(x_test, kpca_kmeans_normalized, axis=1)

    # Perform experiments on it
    nn.experiment(kpca_kmeans_train_new, kpca_kmeans_test_new, y_train, y_test)

    print('\n--------------------------')
    print('RP + KMEANS+ NN')
    print('--------------------------')

    # Declare Neural Network
    nn = NeuralNetwork(layer1_nodes=kwargs['layer1_nodes'],
                       layer2_nodes=kwargs['layer2_nodes'],
                       learning_rate=kwargs['learning_rate'])

    # Augment the original dataset by adding clusters produced by k-Means as features
    rp_kmeans_normalized = (rp_kmeans[0] - np.mean(rp_kmeans[0])) / np.std(rp_kmeans[0])
    rp_kmeans_normalized = np.expand_dims(rp_kmeans_normalized, axis=1)
    rp_kmeans_train_new = np.append(x_train, rp_kmeans_normalized, axis=1)
    rp_kmeans_normalized = (rp_kmeans[1] - np.mean(rp_kmeans[1])) / np.std(rp_kmeans[1])
    rp_kmeans_normalized = np.expand_dims(rp_kmeans_normalized, axis=1)
    rp_kmeans_test_new = np.append(x_test, rp_kmeans_normalized, axis=1)

    # Perform experiments on it
    nn.experiment(rp_kmeans_train_new, rp_kmeans_test_new, y_train, y_test)

    print('\n--------------------------')
    print('PCA + GMM + NN')
    print('--------------------------')

    # Declare Neural Network
    nn = NeuralNetwork(layer1_nodes=kwargs['layer1_nodes'],
                       layer2_nodes=kwargs['layer2_nodes'],
                       learning_rate=kwargs['learning_rate'])

    # Augment the original dataset by adding clusters produced by k-Means as features
    pca_gmm_normalized = (pca_gmm[0] - np.mean(pca_gmm[0])) / np.std(pca_gmm[0])
    pca_gmm_normalized = np.expand_dims(pca_gmm_normalized, axis=1)
    pca_gmm_train_new = np.append(x_train, pca_gmm_normalized, axis=1)
    pca_gmm_normalized = (pca_gmm[1] - np.mean(pca_gmm[1])) / np.std(pca_gmm[1])
    pca_gmm_normalized = np.expand_dims(pca_gmm_normalized, axis=1)
    pca_gmm_test_new = np.append(x_test, pca_gmm_normalized, axis=1)

    # Perform experiments on it
    nn.experiment(pca_gmm_train_new, pca_gmm_test_new, y_train, y_test)

    print('\n--------------------------')
    print('ICA + GMM + NN')
    print('--------------------------')

    # Declare Neural Network
    nn = NeuralNetwork(layer1_nodes=kwargs['layer1_nodes'],
                       layer2_nodes=kwargs['layer2_nodes'],
                       learning_rate=kwargs['learning_rate'])

    # Augment the original dataset by adding clusters produced by k-Means as features
    ica_gmm_normalized = (ica_gmm[0] - np.mean(ica_gmm[0])) / np.std(ica_gmm[0])
    ica_gmm_normalized = np.expand_dims(ica_gmm_normalized, axis=1)
    ica_gmm_train_new = np.append(x_train, ica_gmm_normalized, axis=1)
    ica_gmm_normalized = (ica_gmm[1] - np.mean(ica_gmm[1])) / np.std(ica_gmm[1])
    ica_gmm_normalized = np.expand_dims(ica_gmm_normalized, axis=1)
    ica_gmm_test_new = np.append(x_test, ica_gmm_normalized, axis=1)

    # Perform experiments on it
    nn.experiment(ica_gmm_train_new, ica_gmm_test_new, y_train, y_test)

    print('\n--------------------------')
    print('KPCA + GMM + NN')
    print('--------------------------')

    # Declare Neural Network
    nn = NeuralNetwork(layer1_nodes=kwargs['layer1_nodes'],
                       layer2_nodes=kwargs['layer2_nodes'],
                       learning_rate=kwargs['learning_rate'])

    # Augment the original dataset by adding clusters produced by k-Means as features
    kpca_gmm_normalized = (kpca_gmm[0] - np.mean(kpca_gmm[0])) / np.std(kpca_gmm[0])
    kpca_gmm_normalized = np.expand_dims(kpca_gmm_normalized, axis=1)
    kpca_gmm_train_new = np.append(x_train, kpca_gmm_normalized, axis=1)
    kpca_gmm_normalized = (kpca_gmm[1] - np.mean(kpca_gmm[1])) / np.std(kpca_gmm[1])
    kpca_gmm_normalized = np.expand_dims(kpca_gmm_normalized, axis=1)
    kpca_gmm_test_new = np.append(x_test, kpca_gmm_normalized, axis=1)

    # Perform experiments on it
    nn.experiment(kpca_gmm_train_new, kpca_gmm_test_new, y_train, y_test)

    print('\n--------------------------')
    print('RP + GMM + NN')
    print('--------------------------')

    # Declare Neural Network
    nn = NeuralNetwork(layer1_nodes=kwargs['layer1_nodes'],
                       layer2_nodes=kwargs['layer2_nodes'],
                       learning_rate=kwargs['learning_rate'])

    # Augment the original dataset by adding clusters produced by k-Means as features
    rp_gmm_normalized = (rp_gmm[0] - np.mean(rp_gmm[0])) / np.std(rp_gmm[0])
    rp_gmm_normalized = np.expand_dims(rp_gmm_normalized, axis=1)
    rp_gmm_train_new = np.append(x_train, rp_gmm_normalized, axis=1)
    rp_gmm_normalized = (rp_gmm[1] - np.mean(rp_gmm[1])) / np.std(rp_gmm[1])
    rp_gmm_normalized = np.expand_dims(rp_gmm_normalized, axis=1)
    rp_gmm_test_new = np.append(x_test, rp_gmm_normalized, axis=1)

    # Perform experiments on it
    nn.experiment(rp_gmm_train_new, rp_gmm_test_new, y_train, y_test)



if __name__ == '__main__':
    diab_flag = True
    reef_flag = False
    cancer_flag = True
    reload_flag = False
    # Run experiment 1 on Diabetes
    if diab_flag:
        print('\n--------------------------')
        dataset = "diabetes_data_upload.csv"
        mytitle = "diabetes_data_upload"
        perform_model_complexity = False
        x_train, x_test, y_train, y_test = load_dataset(dataset1=dataset)

        if reload_flag or ~os.path.isfile('diabetes_dim_red.pkl'):
            # Clustering experiments
            kmeans_clusters, gmm_clusters = clustering(x_train, x_test, y_train,
                                                       dataset=mytitle,
                                                       kmeans_n_clusters=2,
                                                       gmm_n_clusters=2, gmm_covariance='diag',
                                                       perform_model_complexity=perform_model_complexity)

            # Dimensionality reduction experiments
            x_pca, x_ica, x_kpca, x_rp, pca_kmeans, pca_gmm, ica_kmeans, ica_gmm, kpca_kmeans, kpca_gmm, rp_kmeans, rp_gmm = dimensionality_reduction(x_train, x_test, y_train,
                                                                                                                                                      dataset=mytitle,
                                                                                                                                                      pca_n_components=16, pca_kmeans_n_clusters=2,
                                                                                                                                                      pca_gmm_n_clusters=2, pca_gmm_covariance='full',
                                                                                                                                                      ica_n_components=11, ica_kmeans_n_clusters=2,
                                                                                                                                                      ica_gmm_n_clusters=2, ica_gmm_covariance='full',
                                                                                                                                                      kpca_n_components=16, kpca_kernel='sigmoid',
                                                                                                                                                      kpca_kmeans_n_clusters=2,
                                                                                                                                                      kpca_gmm_n_clusters=2, kpca_gmm_covariance='full',
                                                                                                                                                      rp_n_components=16, rp_kmeans_n_clusters=2,
                                                                                                                                                      rp_gmm_n_clusters=2, rp_gmm_covariance='full',
                                                                                                                                                      perform_model_complexity=perform_model_complexity)
            with open('diabetes_dim_red.pkl', 'wb') as f:
                pickle.dump([x_pca, x_ica, x_kpca, x_rp, pca_kmeans, pca_gmm, ica_kmeans, ica_gmm, kpca_kmeans, kpca_gmm, rp_kmeans, rp_gmm, kmeans_clusters, gmm_clusters], f)

        else:
            with open('diabetes_dim_red.pkl', 'rb') as f:
                [x_pca, x_ica, x_kpca, x_rp, pca_kmeans, pca_gmm, ica_kmeans, ica_gmm, kpca_kmeans, kpca_gmm, rp_kmeans, rp_gmm, kmeans_clusters, gmm_clusters] = pickle.load(f)
        # Neural Network experiments
        neural_network(x_train, x_test, y_train, y_test,
                       x_pca, x_ica, x_kpca, x_rp, pca_kmeans, pca_gmm, ica_kmeans, ica_gmm, kpca_kmeans, kpca_gmm, rp_kmeans, rp_gmm ,
                       kmeans_clusters, gmm_clusters,
                       layer1_nodes=50, layer2_nodes=30, learning_rate=0.4)

    if cancer_flag:
        # Run experiment 2 on Breast Cancer
        print('\n--------------------------')
        perform_model_complexity = False
        dataset1 = load_breast_cancer()
        mytitle = 'breast_cancer'
        x_train, x_test, y_train, y_test = load_dataset(data=dataset1)

        if reload_flag or ~os.path.isfile('cancer_dim_red.pkl'):
            # Clustering experiments
            kmeans_clusters, gmm_clusters = clustering(x_train, x_test, y_train,
                                                       dataset=mytitle,
                                                       kmeans_n_clusters=2,
                                                       gmm_n_clusters=2, gmm_covariance='full',
                                                       perform_model_complexity=perform_model_complexity)

            # Dimensionality reduction experiments
            x_pca, x_ica, x_kpca, x_rp, pca_kmeans, pca_gmm, ica_kmeans, ica_gmm, kpca_kmeans, kpca_gmm, rp_kmeans, rp_gmm = dimensionality_reduction(x_train, x_test, y_train,
                                                                                                                                                      dataset=mytitle,
                                                                                                                                                      pca_n_components=21, pca_kmeans_n_clusters=2,
                                                                                                                                                      pca_gmm_n_clusters=2, pca_gmm_covariance='diag',
                                                                                                                                                      ica_n_components=11, ica_kmeans_n_clusters=2,
                                                                                                                                                      ica_gmm_n_clusters=2, ica_gmm_covariance='full',
                                                                                                                                                      kpca_n_components=21, kpca_kernel='cosine',
                                                                                                                                                      kpca_kmeans_n_clusters=2,
                                                                                                                                                      kpca_gmm_n_clusters=2, kpca_gmm_covariance='full',
                                                                                                                                                      rp_n_components=30, rp_kmeans_n_clusters=2,
                                                                                                                                                      rp_gmm_n_clusters=2, rp_gmm_covariance='diag',
                                                                                                                                                      perform_model_complexity=perform_model_complexity)
            with open('cancer_dim_red.pkl', 'wb') as f:
                pickle.dump([x_pca, x_ica, x_kpca, x_rp, pca_kmeans, pca_gmm, ica_kmeans, ica_gmm, kpca_kmeans, kpca_gmm, rp_kmeans, rp_gmm, kmeans_clusters, gmm_clusters], f)

        else:
            with open('cancer_dim_red.pkl', 'rb') as f:
                [x_pca, x_ica, x_kpca, x_rp, pca_kmeans, pca_gmm, ica_kmeans, ica_gmm, kpca_kmeans, kpca_gmm, rp_kmeans, rp_gmm, kmeans_clusters, gmm_clusters] = pickle.load(f)

        # Neural Network experiments
        neural_network(x_train, x_test, y_train, y_test,
                       x_pca, x_ica, x_kpca, x_rp,
                       kmeans_clusters, gmm_clusters, pca_kmeans, pca_gmm, ica_kmeans, ica_gmm, kpca_kmeans, kpca_gmm, rp_kmeans, rp_gmm,
                       layer1_nodes=150, layer2_nodes=100, learning_rate=0.06)
    if reef_flag:
        # Run experiment 3 on Coral Reef
        print('\n--------------------------')
        dataset = "./REEF_DATASET/EILAT2/"
        mytitle = 'coral_reef'
        perform_model_complexity = False
        x_train, x_test, y_train, y_test = load_dataset(dataset1=dataset)

        # Clustering experiments
        kmeans_clusters, gmm_clusters = clustering(x_train, x_test, y_train,
                                                   dataset=mytitle,
                                                   kmeans_n_clusters=5,
                                                   gmm_n_clusters=5, gmm_covariance='full',
                                                   perform_model_complexity=perform_model_complexity)

        # Dimensionality reduction experiments
        x_pca, x_ica, x_kpca, x_rp = dimensionality_reduction(x_train, x_test, y_train,
                                                              dataset=mytitle,
                                                              pca_n_components=10, pca_kmeans_n_clusters=2,
                                                              pca_gmm_n_clusters=4, pca_gmm_covariance='diag',
                                                              ica_n_components=12, ica_kmeans_n_clusters=2,
                                                              ica_gmm_n_clusters=5, ica_gmm_covariance='diag',
                                                              kpca_n_components=10, kpca_kernel='cosine',
                                                              kpca_kmeans_n_clusters=2,
                                                              kpca_gmm_n_clusters=4, kpca_gmm_covariance='diag',
                                                              rp_n_components=20, rp_kmeans_n_clusters=2,
                                                              rp_gmm_n_clusters=3, rp_gmm_covariance='full',
                                                              perform_model_complexity=perform_model_complexity)

        # Neural Network experiments
        neural_network(x_train, x_test, y_train, y_test,
                       x_pca, x_ica, x_kpca, x_rp,
                       kmeans_clusters, gmm_clusters,
                       layer1_nodes=50, layer2_nodes=30, learning_rate=0.4)