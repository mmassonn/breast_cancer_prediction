#BreastCancer


#Library
    import numpy as np # linear algebra
    import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
    # Input data files are available in the "../input/" directory.
    import os
    print(os.listdir("F:/Fichier/IA/Pensée artificielle/Article/Article 3"))
    # Any results you write to the current directory are saved as output.
    #sklearn library (Machine learning project)
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC
    from sklearn.model_selection import GridSearchCV
    from sklearn.utils import class_weight
    from sklearn.metrics import roc_curve, auc, roc_auc_score, f1_score
    from sklearn.model_selection import train_test_split
    from sklearn.calibration import CalibratedClassifierCV
    import pickle
    #keras libraty (Deep learning project)
    from keras.layers import Dense, Activation, Dropout, BatchNormalization, Input
    from keras.models import Sequential, Model
    from keras import optimizers, regularizers, initializers
    from keras.callbacks import ModelCheckpoint, Callback
    from keras import backend as K
    from keras.optimizers import Adam
    #tensorflow library
    import tensorflow as tf
    #Classifier library
    from xgboost import XGBClassifier
    #design library
    import matplotlib.pyplot as plt
    import seaborn as sns
    #Warning library
    import warnings
    warnings.filterwarnings("ignore")
    
#DataExploration 
    
    #LoadDataset
        
        BC_df= pd.read_csv('F:/Fichier/IA/Pensée artificielle/Article/Article 3/data/data.csv')#Read a comma-separated values (csv) file into DataFrame.
            
        #ExploringDataset
            
            print (BC_df.info())#info_get a concise summary of the dataframe.
            print(BC_df.shape)#Shape_Return the shape of an array.(569, 33)
            BC_head = BC_df.head()#This function returns the first n rows for the object based on position_The first 5 raws
            BC_df.describe
       
        # Map : 
        
            #Benign and malignant number (357 benign, 212 malignant)
                BC_df['diagnosis'].value_counts()
                
                sns.set(font_scale=1.4)
                BC_df['diagnosis'].value_counts().plot(kind='bar', figsize=(7, 6), rot=0, color='r') #Return a Series containing counts of unique values._The resulting object will be in descending order so that the first element is the most frequently-occurring element.
                plt.xlabel("diagnosis", labelpad=14)
                plt.ylabel("Nombre de patient", labelpad=14)
                plt.title('Nombre de cancers malin et benin diagnostiqué ', y=1.02)
                
                
        #scatterplot
            BC_df.columns#Columns name
            
            sns.set()
            cols = ['radius_mean', 'texture_mean', 'perimeter_mean',
       'area_mean', 'smoothness_mean', 'compactness_mean', 'concavity_mean',
       'concave points_mean', 'symmetry_mean', 'fractal_dimension_mean',
       'radius_se', 'texture_se', 'perimeter_se', 'area_se', 'smoothness_se',
       'compactness_se', 'concavity_se', 'concave points_se', 'symmetry_se',
       'fractal_dimension_se', 'radius_worst', 'texture_worst',
       'perimeter_worst', 'area_worst', 'smoothness_worst',
       'compactness_worst', 'concavity_worst', 'concave points_worst',
       'symmetry_worst', 'fractal_dimension_worst']
            sns.pairplot(BC_df[cols], size = 2.5)
            plt.show()
            
        #drop some column
        BCx_df = BC_df.drop(['id', 'diagnosis','Unnamed: 32'], axis=1)
        BCy_df = BC_df.drop(['id','radius_mean', 'texture_mean', 'perimeter_mean',
       'area_mean', 'smoothness_mean', 'compactness_mean', 'concavity_mean',
       'concave points_mean', 'symmetry_mean', 'fractal_dimension_mean',
       'radius_se', 'texture_se', 'perimeter_se', 'area_se', 'smoothness_se',
       'compactness_se', 'concavity_se', 'concave points_se', 'symmetry_se',
       'fractal_dimension_se', 'radius_worst', 'texture_worst',
       'perimeter_worst', 'area_worst', 'smoothness_worst',
       'compactness_worst', 'concavity_worst', 'concave points_worst',
       'symmetry_worst', 'fractal_dimension_worst','Unnamed: 32'], axis=1)
            
            
        #Change M = 1 et B = 0
        BCy_df = BCy_df.replace('M', 1)
        BCy_df = BCy_df.replace('B', 0)
        
#Scaling (Normalized before Feature extraction)
    BCx_scaler1 = StandardScaler()#Standardize features by removing the mean and scaling to unit variance
    BCx_scaler1.fit(BCx_df.values)#Fit with BEE_descriptors_df.values
    BCx_df = pd.DataFrame(BCx_scaler1.transform(BCx_df.values),columns=BCx_df.columns)#Two-dimensional, size-mutable, potentially heterogeneous tabular data. #Call func on self producing a DataFrame with transformed values.

#Determine train, valid and test sets
    test_size = 0.1
    valid_size = 0.1
    X_train, X_test, y_train, y_test = train_test_split(BCx_df.values, BCy_df.values.flatten(), test_size=test_size , random_state=42,stratify=BCy_df.values.flatten())# Split arrays or matrices into random train and test subsets and flatten Return a copy of the array collapsed into one dimension (to convert a ndarray to 1D array).    
    X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=valid_size , random_state=42,stratify=y_train)# Split arrays or matrices into random train and test subsets

#Sklearn Support Vector Classification Model _ Model : classification
    nca = 30
    parameters = {'kernel':['sigmoid', 'rbf'], 'C':[1,0.5], 'gamma':[1/nca,1/np.sqrt(nca)],'probability':[True]}#Defin parameters
    BC_svc = GridSearchCV(SVC(random_state=23,class_weight='balanced'), parameters, cv=5, scoring='roc_auc',n_jobs=-1)#The parameters of the estimator used to apply these methods are optimized by cross-validated grid-search over a parameter grid. and estimator is SVC
    result = BC_svc.fit(X_train, y_train)#fit svc on dataset values
            
    print(result.best_estimator_)#Best estimator
    print(result.best_score_)#Best score

#Metrics
    #Fonction
                    
            #Find optimal threshold
                def Find_Optimal_threshold(target, predicted):
                    
                    target = target.reshape(-1,1)#Gives a new shape to an array without changing its data.
                    predicted = predicted.reshape(-1,1)
                    
                    rng = np.arange(0.0, 0.99, 0.001)
                    f1s = np.zeros((rng.shape[0],predicted.shape[1]))#Return a new array of given shape and type, filled with zeros.
                    for i in range(0,predicted.shape[1]):
                        for j,t in enumerate(rng):
                            p = np.array((predicted[:,i])>t, dtype=np.int8)
                            scoref1 = f1_score(target[:,i], p, average='binary')#F1 = 2 * (precision * recall) / (precision + recall)
                            f1s[j,i] = scoref1
                            
                    threshold = np.empty(predicted.shape[1])#Return a new array of given shape and type, without initializing entries.
                    for i in range(predicted.shape[1]):
                        threshold[i] = rng[int(np.where(f1s[:,i] == np.max(f1s[:,i]))[0][0])]#Return elements chosen from x or y depending on condition
                        
                    return threshold
                
            #Calibration Postprocessing :quality of the returned probabilities
            
            pred = BC_svc.predict_proba(X_valid)#predict prob on validation dataset values
            print(pred)

            BC_svc_calib = CalibratedClassifierCV(BC_svc, cv='prefit')#Probability calibration with isotonic regression or logistic regression.
            BC_svc_calib.fit(X_valid, y_valid)#fit calibratedclassifierCV on validation dataset values
            
            pred = BC_svc_calib.predict_proba(X_valid)#probability of the prediction
            print (pred)
            pred = pred[:,1]#prediction between 0 and 1
            print (pred)
            pred_svc_t = np.copy(pred)#copy the prediction
           
            threshold = Find_Optimal_threshold(y_valid, pred)#Find optimal threshold
            print (y_valid)
            print(threshold)#print the threshold
            
            #The F1 score can be interpreted as a weighted average of the precision and recall, where an F1 score reaches its best value at 1 and worst score at 0.
            pred = BC_svc_calib.predict(X_test)
            print (pred)
            print (y_test)
            f1_score(y_test,pred)
            
            #Compute Area Under the Receiver Operating Characteristic Curve (ROC AUC) from prediction scores.
            pred = BC_svc_calib.predict_proba(X_test)
            print (pred)
            ROC=roc_auc_score(y_test,pred[:,1])
            print (ROC)
            
            pred = pred[:,1]#prediction between 0 and 1
            pred_svc = np.copy(pred)#copy the prediction
            pred[pred<=threshold] = 0#if the prediction is under the threshold so pred == 0
            pred[pred>threshold] = 1#if the prediction is upper the threshold so pred == 1
            svc_score = f1_score(y_test,pred)#calculate f1 score
            print(svc_score)#print svc_score
            
            y = np.array(BCx_df.loc[24].values).reshape(1, -1)#The reshape() function is used to give a new shape to an array without changing its data.
            result = BC_svc.predict(y)#predict on BEE_descriptors
            prob = BC_svc.predict_proba(y)#predict probability on BEE_descriptor
            print(result)#print result
            print(prob)#print prob
            print(int(prob[:,1]>threshold))#print all probability upper than the threshold
            
#Gradient Boosting of Keras Model with XGBoost_Model  
        #Create XGBoost Model 
            parameters = {'learning_rate':[0.05,0.1,0.15],'n_estimators':[75,100,125], 'max_depth':[3,4,5],'booster':['gbtree','dart'],'reg_alpha':[0.,0.1,0.05],'reg_lambda':[0.,0.1,0.5,1.]} #defin parameters
            
            BC_xgb_gb = GridSearchCV(XGBClassifier(random_state=32), parameters, cv=5, scoring='roc_auc',n_jobs=-1)#The parameters of the estimator used to apply these methods are optimized by cross-validated grid-search over a parameter grid.
            result = BC_xgb_gb.fit(X_train, y_train)#fit the GridSearch in tain dataset
            
            print(result.best_estimator_)#print best estimator 
            print(result.best_score_)#print best score
            
            pred = BC_xgb_gb.predict_proba(X_valid)#probabilité prédiction on validation set
        
        #Calibration Postprocessing :quality of the returned probabilities    
            BC_xgb_gb_calib = CalibratedClassifierCV(BC_xgb_gb, cv='prefit')#Probability calibration with isotonic regression or logistic regression.
            BC_xgb_gb_calib.fit(X_valid, y_valid)#fit calibration on validation dataset
           
            pred = BC_xgb_gb.predict_proba(X_valid)#probability of the prediction validation dataset 
            pred = pred[:,1]#prediction between 0 and 1
            pred_xgb_gb_t= np.copy(pred)#copy the prediction
            
            threshold = Find_Optimal_threshold(y_valid, pred)#find optimal threshold
            print(threshold)#print the threshold
            
            #The F1 score can be interpreted as a weighted average of the precision and recall, where an F1 score reaches its best value at 1 and worst score at 0.
            pred = BC_xgb_gb_calib.predict(X_test)
            f1_score(y_test,pred)
            
            #Compute Area Under the Receiver Operating Characteristic Curve (ROC AUC) from prediction scores.
            pred = BC_xgb_gb_calib.predict_proba(X_test)
            roc_auc_score(y_test,pred[:,1])
            
            
            pred = pred[:,1]#prediction between 0 and 1
            pred_xgb_gb = np.copy(pred)#copy the prediction
            pred[pred<=threshold] = 0#if the prediction is under the threshold so pred == 0
            pred[pred>threshold] = 1#if the prediction is upper the threshold so pred == 1
            xgb_gb_score = f1_score(y_test,pred)#calculate f1 score
            print(xgb_gb_score)#print svc_score
            
            
            result = BC_xgb_gb_calib.predict(y)#predict on BEE_svc
            prob = BC_xgb_gb_calib.predict_proba(y)#predict probability on BEE_svc
            print(result)#print result
            print(prob)#print prob
            print(int(prob[:,1]>=threshold))#print all probability upper than the threshold
            
    
