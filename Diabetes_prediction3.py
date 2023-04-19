import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
import pickle
from streamlit_option_menu import option_menu

import sys
print(sys.executable)

st.set_page_config(page_title='Random Forest Algorithm', layout='wide')

st.write("""
# Random Forest Algorithm
""")

#Sidebar to navigate
with st.sidebar:
    selected = option_menu('Diabetes Prediction',
                           ['Model Execution',
                           'Model Prediction'],
                            default_index=0)
    

if (selected == "Model Execution"):
     #---------------------------------#
    # Sidebar - Collects user input features into dataframe
    st.header('Upload your Diabetes CSV data')

    uploaded_file = st.file_uploader("Upload your diabetes  input CSV file", type=["csv"])

    # Displays the dataset
    st.subheader('Dataset')

    #EDA
    def data_cleaning(data):
        zero_val_cols = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin','BMI']
        #Replace the 0 values with Mean for all the below listed columns
        for col in zero_val_cols:
            mean = data[col].mean()
            data[col] = data[col].replace(0,mean)
        return data 

    #Split the data into Train and test
    def train_test_split(cleaned_data,test_size,final_iv_cols):
        from sklearn.model_selection import train_test_split
        #X = cleaned_data.drop('Outcome' , axis=1) # independent variables
        X= cleaned_data[final_iv_cols]
        y = cleaned_data['Outcome'] # dependent variable 
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size,stratify = y)

        return X_train, X_test, y_train, y_test 

    def scaling_data(X_train, X_test):
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.fit_transform(X_test)

        return X_train,X_test

    def predict_on_test_data(model,X_test):
        y_pred = model.predict(X_test)
        return y_pred

    def predict_prob_on_test_data(model,X_test):
        y_pred_prob = model.predict_proba(X_test)
        return y_pred_prob

    def get_metrics(y_test, y_pred):
        from sklearn.metrics import accuracy_score,precision_score,recall_score
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        
        return {'accuracy': round(acc, 2), 'precision': round(prec, 2), 'recall': round(recall, 2)}

    #Random Forest Classifier
    def model_Random_forest_cls(X_train,y_train):
        from sklearn.metrics import accuracy_score,precision_score,recall_score
        from sklearn.model_selection import RandomizedSearchCV
        from sklearn.ensemble import RandomForestClassifier

        # n_estimators = 101 
        # min_samples_split = 10
        # min_samples_leaf = 3
        # max_features = 'sqrt'
        # max_depth = 20
        # bootstrap = True

        n_estimators = parameter_n_estimators
        min_samples_split = parameter_min_samples_split
        min_samples_leaf =parameter_min_samples_leaf
        #max_features = parameter_max_features
        max_depth = 20
        bootstrap = parameter_bootstrap

        random_grid = {'n_estimators': parameter_n_estimators,
                        #'max_features': parameter_max_features,
                        'max_depth': max_depth,
                        'min_samples_split': min_samples_split,
                        'min_samples_leaf': min_samples_leaf,
                        'bootstrap': bootstrap
                    }
        
        #model_tuned = RandomForestClassifier(n_estimators = n_estimators, min_samples_split = min_samples_split,
        #                                    min_samples_leaf= min_samples_leaf, max_features = max_features,
        #                                    max_depth= max_depth, bootstrap=bootstrap) 
        
        model_tuned = RandomForestClassifier(n_estimators = n_estimators, min_samples_split = min_samples_split,
                                            min_samples_leaf= min_samples_leaf,
                                            max_depth= max_depth, bootstrap=bootstrap) 
        model_tuned.fit(X_train, y_train)

        #Display Model Performance in streamlit
        st.subheader('Model Performance')

        y_pred_test = model_tuned.predict(X_test)
        st.write('Accuracy Score:')
        accuracy_score = accuracy_score(y_test, y_pred_test)
        acc_score =accuracy_score * 100
        st.info(f'Model is {round(acc_score, 2)} % accurate in predicting the results')
        
        prec = precision_score(y_test, y_pred_test)
        st.write('precision_score:')
        prec = prec *100
        st.info(f'{round(prec,2)}%')

        recall = recall_score(y_test, y_pred_test)
        st.write('recall_score:')
        recall = recall *100
        st.info(f'{round(recall,2)}%')

        #st.subheader('Parameters used for the model')
        #st.write(random_grid)
        
        return model_tuned,y_pred_test,prec,recall,random_grid
    
    def create_experiment(experiment_name,run_name, run_metrics,model,run_params=None):
        import mlflow
        mlflow.set_experiment(experiment_name)
    
        with mlflow.start_run():
            if not run_params == None:
                for param in run_params:
                    mlflow.log_param(param, run_params[param])
                
            for metric in run_metrics:
                mlflow.log_metric(metric, run_metrics[metric])
            
            mlflow.sklearn.log_model(model, "model")
            
            mlflow.set_tag("tag1", "Random Forest")
            mlflow.set_tags({"tag2":"Randomized Search CV", "tag3":"Testing"})
    
    #Main Function to perform analysis
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write(df)

         # Sidebar - Specify parameter settings
        st.header('Set Parameters')
        split_size = st.slider('Data split (% for Test Set)', 10, 20, 30, 10)

        st.subheader('Hyperparameter Tuning')

        #n_estimators
        parameter_n_estimators = st.slider('Number of estimators (n_estimators)',2 , 200, 2)

        #max_features
        #parameter_max_features = st.radio("Max Features", ('sqrt','auto'))

        #parameter_max_features = st.sidebar.select_slider('Max features (max_features)', options=['sqrt', 'auto'])

        parameter_min_samples_split = st.slider('Minimum number of samples required to split an internal node (min_samples_split)', 1, 10, 2)
        parameter_min_samples_leaf = st.slider('Minimum number of samples required to be at a leaf node (min_samples_leaf)', 1, 10, 2)

        parameter_bootstrap = st.radio("Bootstrap samples when building trees", (True,False))

       
        #Select Independent and Dependent variables
        st.write('Select Independent Variables')
        
        Pregnancies = st.checkbox("Pregnancies")
        Glucose= st.checkbox("Glucose")
        BloodPressure= st.checkbox("BloodPressure")
        SkinThickness= st.checkbox("SkinThickness")
        Insulin= st.checkbox("Insulin")
        BMI= st.checkbox("BMI")
        DiabetesPedigreeFunction= st.checkbox("DiabetesPedigreeFunction")
        Age= st.checkbox("Age")
        
        final_iv_cols=[]
        if Pregnancies:
            final_iv_cols.append("Pregnancies")

        if Glucose:
            final_iv_cols.append("Glucose")    

        if BloodPressure:
            final_iv_cols.append("BloodPressure")    

        if SkinThickness:
            final_iv_cols.append("SkinThickness")

        if Insulin:
            final_iv_cols.append("Insulin")

        if BMI:
            final_iv_cols.append("BMI")      

        if DiabetesPedigreeFunction:
            final_iv_cols.append("DiabetesPedigreeFunction")

        if Age:
            final_iv_cols.append("Age")             

        # iv_cols=[Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age]
        # final_iv_cols=[]
        
        # for col in iv_cols:
        #     if col:
        #         final_iv_cols.append(f'{col}')

        print(final_iv_cols)
        #cleaned_data = data_cleaning(df)
        X_train, X_test, y_train, y_test = train_test_split(df,split_size,final_iv_cols)
        #X_train, X_test = scaling_data(X_train,X_test)
        
        #button to execute model(train)
        if st.button("Check Model Performance"):
            model_Random_forest_cls, y_pred_test, prec, recall, random_grid = model_Random_forest_cls(X_train, y_train)
            
            #MLFlow work Starts from here
            #-------------------------------
            experiment_name = "RF_classifier" ##Random Forest classifier
            run_name="RF_tuned"
            run_metrics = get_metrics(y_test, y_pred_test)
            run_params = random_grid
            create_experiment(experiment_name,run_name,run_metrics,model_Random_forest_cls,run_params)
            #------------------------------------------------------------
            st.info("Model and the performance results are saved in Mlflow")
            st.subheader("Mlflow Experiment Name")
            st.write(experiment_name)
            #-------------------------
            st.subheader("Mlflow Run Name")
            st.write(run_name)
            #-------------------------

            st.subheader("Trained Model stored in Mlflow")
            st.info("model.pkl")

            #------------------------------------------
            st.subheader("Model Metrices stored in Mlflow")
            st.write("Accuracy")
            st.info(run_metrics["accuracy"])
            st.write("Precision")
            st.info(run_metrics["precision"])
            st.write("Recall")
            st.info(run_metrics["recall"])

            #--------------------------------
            st.subheader("Parameters Used:")
            st.write("N_Estimators")
            st.info(run_params["n_estimators"])
            st.write("Max Depth")
            st.info(run_params["max_depth"])
            st.write("Min Samples Split")
            st.info(run_params["min_samples_split"])
            st.write("Min Samples Leaf")
            st.info(run_params["min_samples_leaf"])
            st.write("Bootstrap")
            st.info(run_params["bootstrap"])
            st.write("--------------------------------")
    else:
         st.info('Awaiting for csv file to be uploaded')    


elif (selected=="Model Prediction"):
      # Sidebar - Collects user input features into dataframe
      st.header('Upload the saved model')


      uploaded_file_pkl = st.file_uploader("Upload your input Pkl file")  
  
      #loaded_model = pickle.load(open(uploaded_file_pkl),'rb', encoding='utf-8')
      
      if uploaded_file_pkl is not None:
         #loaded_model = pickle.load(open(uploaded_file_pkl, 'rb'))
         loaded_model = pickle.loads(uploaded_file_pkl.read())

         Pregnancies = st.text_input('Number of Pregnancies')
         Glucose = st.text_input('Glucose level')
         BloodPressure = st.text_input('Blood Pressure value')
         SkinThickness = st.text_input('Skin Thickness value')
         Insulin = st.text_input('Insulin level')
         BMI = st.text_input('BMI value')
         DiabetesPedigreeFunction = st.text_input('Diabetes Pedigree Function value')
         Age = st.text_input('Age')

         if st.button("Diabtetes Test Results"):
            input_data_arr = np.array([[int(Pregnancies),float(Glucose),float(BloodPressure),float(SkinThickness),float(Insulin),float(BMI),float(DiabetesPedigreeFunction),int(Age)]])
            
            new_df = pd.DataFrame(input_data_arr, columns=['Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age'])
            
            diabetes_diagnosis =loaded_model.predict(new_df)
            
            probability = round(max(loaded_model.predict_proba(new_df)[0]),2)
            

            #prob = loaded_model.predict_proba(input_data_arr)
            #print(max(probability))

            if (diabetes_diagnosis[0] == 1):
                st.warning("The person is Diabetic")
                st.warning("Confidence: {} %".format(probability*100))
            else:
                st.success("The person is Not Diabetic") 
                st.success(f"Confidence ")  


# filename = r"model.pkl"

# loaded_model = pickle.load(open(filename, 'rb'))

# Pregnancies = 6
# Glucose = 148
# BloodPressure = 72
# SkinThickness = 35
# Insulin = 79.79
# BMI = 33.6
# DiabetesPedigreeFunction = 0.627
# Age = 50


# input_data_arr = np.array([[int(Pregnancies),float(Glucose),float(BloodPressure),float(SkinThickness),float(Insulin),float(BMI),float(DiabetesPedigreeFunction),int(Age)]])
# diabetes_diagnosis =loaded_model.predict(input_data_arr)

# if (diabetes_diagnosis[0] == 1):
#             print("The person is Diabetic")
# else:    
#             print("The person is Not Diabetic")


