# -*- coding: utf-8 -*-
"""
Created on Tue Apr 04 23:17:09 2017

@author: Miki
"""

################imports#################
####importing Env packages#####
from os import getcwd ,listdir
from input_collector_latest import MovmentElement
import time
from time import sleep
import re
import pickle
####importing data/ploting packages####
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.axes as axes
from math import sqrt
from scipy.spatial import distance
from matplotlib import pyplot
import matplotlib.pyplot as plt
import pylab
from mpl_toolkits.mplot3d import Axes3D
from random import randint,shuffle
####importing sklern####
from sklearn import svm,linear_model
from sklearn.preprocessing import scale,MinMaxScaler
from sklearn import tree
from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import cross_val_score
from sklearn.grid_search import GridSearchCV
from sklearn.decomposition import PCA
from sklearn.metrics import recall_score,precision_score,f1_score,confusion_matrix
from sklearn.covariance import EllipticEnvelope
from sklearn.neural_network import MLPClassifier
###############################




##This function merges db_\d.pickle database files to one file.
def merge_files():
    files = listdir(getcwd())
    data=[]
    files_to_read =[]
    data_to_merged_file=[]
    for f in files:
        if re.match(r'db_n_\d.pickle',f):
            files_to_read.append(f)
    print files_to_read
    for filename in files_to_read:
        with open(filename,'rb') as handle:
            data = pickle.load(handle)
        data_to_merged_file +=data
    print "merged file len",len(data_to_merged_file)
    with open('db_merged_n.pickle','wb') as handle_write:
        pickle.dump(data_to_merged_file,handle_write)




##This class holds the data after downsapmple function ran.
## adding feture_mode which gets a df made of fetures not coords
class dataset_class:
    def __init__(self,db_filename,class_symbol,best_fetures,size=0):
        self.class_symbol_ = class_symbol
        self.db_filename_=db_filename
        self.df_path_,self.df_fetures_ = self.create_fetures_and_path_df(self.db_filename_,size)
        self.df_size_ = len(self.df_path_)
        self.best_fetures_ = best_fetures
        self.gen_fetures()
    ## creates two data_farmes from db file: one holds the path coords and the second holds the fetures values
    ## it also removes duplicates paths from the db     
    def create_fetures_and_path_df(self,db_filename,size=0):
        data=[]
        Path_db = []
        Fetures_db=[]
        prev_len = 0
        with open(db_filename,'rb') as handle:
            print "Creating db from file:%s" %(db_filename)
            data = pickle.load(handle)
        for item in data:
            if item.relative_path == prev_len:
                continue
            #print "len is: %d" %len(item.relative_path)
            Path_db.append(item.relative_path)
            Fetures_db.append(item.duration)
            prev_len = item.relative_path
        print "Original db size:%d\nNum of Uniq Pathes\Fetures in db is:%d\%d" %(len(data),len(Path_db),len(Fetures_db))
        if size==0:
            size = len(Path_db)
        Path_db = Path_db[0:size]
        Df_path = pd.DataFrame(Path_db)
        Df_path = Df_path.fillna(0)
        Fetures_db = Fetures_db[0:size]
        Df_Fetures = pd.DataFrame(Fetures_db)
        Df_Fetures.rename(columns={0:"duration"},inplace=True)
        return Df_path,Df_Fetures
    ## Generates fetures out of the raw path data
    def gen_fetures(self):
        index_list=[]
        len_list =[]
        distance_list =[]
        speed_list =[]
        start_x_list =[]
        start_y_list =[]
        end_x_list =[]
        end_y_list =[]
        passed_distance_list =[]
        avg_x_coords_list=[]
        avg_y_coords_list =[]
        for ii in range(len(self.df_path_)):
            index_list.append(ii)
            len_list.append(sum(self.df_path_.iloc[ii]!=0))
            first_coord = self.df_path_.iloc[ii][0]
            last_coord = self.df_path_.iloc[ii][len_list[ii]-1]
            distance_list.append(sqrt(((first_coord[0] - last_coord[0]) ** 2) + ((first_coord[1] - last_coord[1]) ** 2)))
            speed_list.append(distance_list[ii]/self.df_fetures_['duration'][ii])
            start_x_list.append(first_coord[0])
            start_y_list.append(first_coord[1])
            end_x_list.append(last_coord[0])
            end_y_list.append(last_coord[1])
            if not self.best_fetures_:
                passed_distance,avg_x_coords,avg_y_coords = self.calc_passed_distance(ii)
                passed_distance_list.append(passed_distance)
                avg_x_coords_list.append(avg_x_coords)
                avg_y_coords_list.append(avg_y_coords)
        self.df_fetures_.insert(0,'idx',pd.Series(index_list))
        self.df_fetures_['length'] = pd.Series(len_list)
        self.df_fetures_['distance'] = pd.Series(distance_list)
        self.df_fetures_['speed'] = pd.Series(speed_list)    
        self.df_fetures_['start_x'] =pd.Series(start_x_list)
        self.df_fetures_['start_y'] =pd.Series(start_y_list)
        self.df_fetures_['end_x'] =pd.Series(end_x_list)
        self.df_fetures_['end_y'] =pd.Series(end_y_list)
        if  not self.best_fetures_:
            self.df_fetures_['passed_distance'] = pd.Series(passed_distance_list)
            self.df_fetures_['avg_x_coords'] = pd.Series(avg_x_coords_list)
            self.df_fetures_['avg_y_coords'] = pd.Series(avg_y_coords_list)
    ## calculating the paassed distacne for the all path
    def calc_passed_distance(self,idx):
        passed_distance=0
        avg_x_coords=0
        avg_y_coords=0
        path_len=0
        for ii in range(1,len(self.df_path_.iloc[idx,:])):
            curr_tup = self.df_path_.iloc[idx,:][ii]
            prev_tup = self.df_path_.iloc[idx,:][ii-1]
            if ii == 1:
                avg_x_coords +=prev_tup[0]
                avg_y_coords +=prev_tup[1]
            if type(curr_tup) == tuple:
                path_len += 1
                avg_x_coords +=curr_tup[0]
                avg_y_coords +=curr_tup[1]
                passed_distance += sqrt(((curr_tup[0] - prev_tup[0]) ** 2) + ((curr_tup[1] - prev_tup[1]) ** 2))
            else:
                break
        avg_x_coords = avg_x_coords/float(path_len)
        avg_y_coords = avg_y_coords/float(path_len)
        return passed_distance,avg_x_coords,avg_y_coords
    ## filter_db:removes samples that are larger then value 
    def filter_db(self,value,feture_name):
        self.df_path_ = self.df_path_[self.df_fetures_[feture_name]>value]
        self.df_fetures_ = self.df_fetures_[self.df_fetures_[feture_name]>value]
        self.df_size_ = len(self.df_fetures_)
    ## print the path we generated 
    def print_path(self,num_of_path,path_index):
        coordX = [x[0] for x in self.df_path_.loc[path_index] if type(x)==tuple]
        coordY = [y[1] for y in self.df_path_.loc[path_index] if type(y)==tuple] 
        colors = range(len(coordX))
        plt.xlim(0,1920)
        title = "Path :",path_index ,"Length is:",len(coordX)
        plt.ylim(1080,0)
        plt.suptitle(title)
        plt.scatter(coordX,coordY,c = colors)
        plt.show() ##blue dot starts the path
     ##print_hist: plot an historgram of all the number of length in the db
    def print_hist(self):
        bins = []
        for rng in range(10,max(self.df_fetures_["length"]),5):
            bins.append(sum(self.df_fetures_["length"] > rng))
        hist = pd.Series(bins,index=range(10,max(self.df_fetures_["length"]),5))
        hist.plot()
    ## dataframe2list: convert the df_path_ or df_fetures_ to list-
    def dataframe2list(self,feture_mode):
        flatten_df =[]
        line=[]
        if feture_mode:
            for i in range(len(self.df_fetures_)):
                flatten_df.append(list(self.df_fetures_.iloc[i]))
        else:
            for k,row in self.df_path_.iterrows():
                for tup in row:
                    if type(tup) == tuple:
                        for val in tup:
                            line.append(val)
                    else:
                        break
                flatten_df.append(line)
                line=[]
        return flatten_df
    ## getting a class symbol array (size==num of samples)
    def get_output_array(self):
        return [self.class_symbol_ for ii in range(self.df_size_)] 
    ## scaling the fetures dataframe
    def scale_fetures(self):
        zero_mean_scaler = scale
        self.df_fetures_['length'] = zero_mean_scaler(self.df_fetures_['length'])
        self.df_fetures_['duration'] = zero_mean_scaler(self.df_fetures_['duration'])
        self.df_fetures_['distance'] = zero_mean_scaler(self.df_fetures_['distance'])
        self.df_fetures_['speed'] = zero_mean_scaler(self.df_fetures_['speed'])
        self.df_fetures_['start_x'] = zero_mean_scaler(self.df_fetures_['start_x'])
        self.df_fetures_['start_y'] = zero_mean_scaler(self.df_fetures_['start_y'])
        self.df_fetures_['end_x'] = zero_mean_scaler(self.df_fetures_['end_x'])
        self.df_fetures_['end_y'] = zero_mean_scaler(self.df_fetures_['end_y'])
        if not self.best_fetures_:
            self.df_fetures_['passed_distance'] = zero_mean_scaler(self.df_fetures_['passed_distance'])
            self.df_fetures_['avg_x_coords'] = zero_mean_scaler(self.df_fetures_['avg_x_coords'])
            self.df_fetures_['avg_y_coords'] = zero_mean_scaler(self.df_fetures_['avg_y_coords'])
    def scale_fetures_minmax(self):
        zero_mean_scaler =  MinMaxScaler(feature_range=(0,10))
        self.df_fetures_['length'] = zero_mean_scaler.fit_transform(self.df_fetures_['length'])
        self.df_fetures_['duration'] = zero_mean_scaler.fit_transform(self.df_fetures_['duration'])
        self.df_fetures_['distance'] = zero_mean_scaler.fit_transform(self.df_fetures_['distance'])
        self.df_fetures_['speed'] = zero_mean_scaler.fit_transform(self.df_fetures_['speed'])
        self.df_fetures_['start_x'] = zero_mean_scaler.fit_transform(self.df_fetures_['start_x'])
        self.df_fetures_['start_y'] = zero_mean_scaler.fit_transform(self.df_fetures_['start_y'])
        self.df_fetures_['end_x'] = zero_mean_scaler.fit_transform(self.df_fetures_['end_x'])
        self.df_fetures_['end_y'] = zero_mean_scaler.fit_transform(self.df_fetures_['end_y'])
        if not self.best_fetures_:
            self.df_fetures_['passed_distance'] = zero_mean_scaler.fit_transform(self.df_fetures_['passed_distance'])
            self.df_fetures_['avg_x_coords'] = zero_mean_scaler.fit_transform(self.df_fetures_['avg_x_coords'])
            self.df_fetures_['avg_y_coords'] = zero_mean_scaler.fit_transform(self.df_fetures_['avg_y_coords'])

def set_fetures_env(test_size=0.2,same_size=1):
    global X_train,X_test,Y_train,Y_test,ds_fetures_m,train_data,Y_train,ds_m,ds_n,train_output,X_train_modfied,X_test_modfied
    ds_n = dataset_class('db_merged_n.pickle',0,1,0)
    fetch_size = ds_n.df_fetures_shape[1] if same_size  else 0
    ds_m = dataset_class('db_merged.pickle',1,1,fetch_size)
    ds_n.filter_db(23,'length')
    ds_m.filter_db(23,'length')
    ds_n.scale_fetures()
    ds_m.scale_fetures()
    
   ## df_fetures_m = filter_eliptic(df_fetures_m)
   ## df_fetures_n = filter_eliptic(df_fetures_n)
    ##dropping highly correlated fetures
    #ds_m.df_fetures_.drop(['avg_y_coords','distance','passed_distance'],1,inplace=True)
    #ds_n.df_fetures_.drop(['avg_y_coords','distance','passed_distance'],1,inplace=True)
    print "Setting Fetures Env: df_fetures_m/n Sizes: %d/%d\n"%(ds_m.df_fetures_.shape[0],ds_n.df_fetures_.shape[0])
    #setting global variables
    train_data = ds_m.dataframe2list(1) + ds_n.dataframe2list(1)
    train_output = ds_m.get_output_array() + ds_n.get_output_array()
    X_train,X_test,Y_train,Y_test = train_test_split(train_data,train_output,test_size=test_size,random_state=0)
    print "Referance Score:%f" %(sum(Y_train)/float(len(Y_train)))
    X_train_modfied = remove_idx(X_train)
    X_test_modfied = remove_idx(X_test)
    

def eval_svm_params(X_fit,Y_fit,use_best):
    t0 = time.time()
    global cls_svm
    best_param = [{'kernel':['rbf'],'gamma':[0.01],'C':[10e6]}]
    tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-1,1e-2,1e-3, 1e-4,1e-5],
                     'C': [1e2, 1e3,1e4,1e5,1e6,1e7,1e8]}]
    tuned_parameters = best_param if use_best else tuned_parameters
    cls_svm = GridSearchCV(svm.SVC(),tuned_parameters)
    cls_svm.fit(X_fit,Y_fit)
    print "Best Score SVM: %f" %cls_svm.best_score_
    print "Running Time:%f"  %(time.time()-t0)


def calc_score(cls,X,Y_actual):
    Y_pred = cls.predict(X)
    rc_score = recall_score(Y_actual,Y_pred)
    pr_score = precision_score(Y_actual,Y_pred)
    fo_score = f1_score(Y_actual,Y_pred)
    print "recall: %f\nprecision:%f\nf1_score:%f\n" %(rc_score,pr_score,fo_score)
    print "Confusion Matrix:\n", confusion_matrix(Y_actual,Y_pred)  
    
    
def eval_nn_params(X_fit,Y_fit):
    t0 = time.time()
    global cls_nn
    tuned_parameters = [{'activation': ['identity', 'logistic', 'tanh', 'relu'], 'alpha': [1e2,1e-1,1e-2,1e-3,1e-4,1e-5],
                     'hidden_layer_sizes': [(8,8,8),(16,16,16)]}]
    cls_nn = GridSearchCV(MLPClassifier(),tuned_parameters)
    cls_nn.fit(X_fit,Y_fit)
    print "Best Score NN: %f" %cls_nn.best_score_
    print "Running Time:%f"  %(time.time()-t0)
    
def remove_idx(X):
    X_modfied =[]
    for ll in X:
        X_modfied.append(ll[1:])
    return X_modfied

def analyze_errors(cls,X_te_modfied,X_te_orig,Y_true,max_size,mode='fp'):
    Y_prediction = cls.predict(X_te_modfied)
    err = Y_prediction - Y_true
    cnt = 0
    for i in range(len(err)):
        if err[i] != 0:
            idx = X_te_orig[i][0]
            if (err[i] == 1 and mode=='fp'):
                ##err[i]==1  pred ==1 true=0 =>false_postive
                cnt +=1
                print "Error index #%d\nFalse Postive-Y_prediction is 1 Y_true is 0" %(i)
                print idx
                ds_n.print_path(1,int(idx))
            elif (err[i] == -1 and mode=='fn'):
                ##err[i]==-1 pred ==0 true =1 =>false_negtive
                cnt +=1
                print "Error index #%d\nFalse Negtive-Y_prediction is 0 Y_true is 1" %(i)     
                ds_m.print_path(1,int(idx))
        if cnt==max_size:
            break
            
            
def evaluate_pca(dim_vec=[7,8,9]):
    pca_array = {}
    global X_train_pca,cls_pca
    X_train_pca ={}
    cls_pca ={}
    for dim in dim_vec:
        pca_name = "pca_" + str(dim)
        print "Dim reduction: %s"%dim
        pca_array[pca_name] = PCA(dim)
        X_train_pca[pca_name] =  pca_array[pca_name].fit(X_train_modfied).transform(X_train_modfied)
        eval_svm_params(X_train_pca[pca_name],Y_train,1)
        cls_pca[pca_name] = cls_svm            
    
        