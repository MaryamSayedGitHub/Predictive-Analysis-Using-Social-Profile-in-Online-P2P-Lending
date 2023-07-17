# Predictive Analysis Using Social Profile in Online P2P Lending Decision

This model is to predict if borrowers in an online peer-to-peer (P2P) lending platform will pay back their loans on time and with lower interest rates. We look at factors related to the borrower, the loan, and their social profile to see how they affect loan performance. By analyzing this data, we can suggest ways for borrowers and lenders to increase their chances of successful lending and repayment.



## Table of Contents

- [Introduction](#introduction)
- [Data Collection](#data-collection)
- [Data Preprocessing](#data-preprocessing)
- [EDA](#data-visualization)
- [Feature Engineering](#feature-engineering)
- [Model Building](#model-building)
- [Hyperparameter Tuning](#hyperparameter-tuning)
- [Models Evaluation](#model-evaluation)
- [Pipelining](#pipelining)
- [Deployment](#deployment)


# Introduction

We study the borrower-, loan- and social- related determinants of performance predictability in an online P2P lending market by conceptualizing financial and social strength to predict whether the borrowers could be funded with lower interest, and the lenders would be timely paid. 

So, this project consists of two parts : 
 1) Binary classification prediction :
  - LoanStatus :  Whether the borrower will repay the loan or not.
 2) multi-regressor prediction : 
   - preferred EMI : Represents the ideal or desired amount that an individual or borrower would like to pay as their monthly installment towards a loan
   - preferred ROI : Represents the expected or desired rate of return on an investment. 
   -  ELA : Represents the maximum loan amount for each loan application based on the criteria set by the lending institution. It can be a significant feature in     analyzing and modeling loan data, providing insights into loan approval decisions and potential borrowing capacities.




# Data Collection

The data used was provided from Prosper company.




# Data Preprocessing

  The data was preprocessed by : 
  - Dropping features which aren't important.


   ![Screenshot 2023-07-09 232004](https://github.com/Technocolabs100/Predictive-Analysis-Using-Social-Profile-in-Online-P2P-Lending-Decision/assets/107771328/69f13a98-15a3-4049-8238-3033883abd66)


   # -------------------------------------------------------------------------


  ![Screenshot 2023-07-09 232217](https://github.com/Technocolabs100/Predictive-Analysis-Using-Social-Profile-in-Online-P2P-Lending-Decision/assets/107771328/2e09a812-4a45-49c4-b16f-3115aa7e3a7e)



  # Handling null values 

   - Missing values in the data  : in the columns containing less than 70% of data null values by replacing them with median value ( for numerical features ) 
     and with mode ( for categorical features ) otherwise these features were dropped.



   ![Screenshot 2023-07-09 232051](https://github.com/Technocolabs100/Predictive-Analysis-Using-Social-Profile-in-Online-P2P-Lending-Decision/assets/107771328/fc6b2170-4efc-4afb-bdbc-c779dbeb1944)


   # -------------------------------------------------------------------------


   ![Screenshot 2023-07-09 232113](https://github.com/Technocolabs100/Predictive-Analysis-Using-Social-Profile-in-Online-P2P-Lending-Decision/assets/107771328/8afd4b5b-3b09-47fc-b2da-5ded7fb9bb8c)


   # -------------------------------------------------------------------------


   ![Screenshot 2023-07-09 232132](https://github.com/Technocolabs100/Predictive-Analysis-Using-Social-Profile-in-Online-P2P-Lending-Decision/assets/107771328/386928eb-7f9a-4bd1-83bd-f9522b9819f8)



    
  # Handling outliers 

   - Values which are higher than the upper bound of values and lower than the lower bound of the values  : by replacing some with min or max value.

   ![Screenshot 2023-07-09 232652](https://github.com/Technocolabs100/Predictive-Analysis-Using-Social-Profile-in-Online-P2P-Lending-Decision/assets/107771328/b1305710-f20d-446b-874d-2d57f2c95b2d)


   # -------------------------------------------------------------------------


  ![Screenshot 2023-07-09 232925](https://github.com/Technocolabs100/Predictive-Analysis-Using-Social-Profile-in-Online-P2P-Lending-Decision/assets/107771328/4ba37928-58e4-43b0-9ece-9f4e23108eb2)


   # -------------------------------------------------------------------------


![Screenshot 2023-07-09 232823](https://github.com/Technocolabs100/Predictive-Analysis-Using-Social-Profile-in-Online-P2P-Lending-Decision/assets/107771328/abf93e1b-f709-4ab6-a383-382e4129f0c6)


   # -------------------------------------------------------------------------


![Screenshot 2023-07-09 232715](https://github.com/Technocolabs100/Predictive-Analysis-Using-Social-Profile-in-Online-P2P-Lending-Decision/assets/107771328/7fcd3492-d17f-455b-8ce8-23aeda1ca079)


   # -------------------------------------------------------------------------


![Screenshot 2023-07-09 233012](https://github.com/Technocolabs100/Predictive-Analysis-Using-Social-Profile-in-Online-P2P-Lending-Decision/assets/107771328/1f95366d-3957-41ad-a3eb-a90d439681af)


   # -------------------------------------------------------------------------


  - Converted the features ( with datatype : object ) including date and time values into suitable format ( datatype : datetime ) to be easily used in the data.

![Screenshot 2023-07-09 233051](https://github.com/Technocolabs100/Predictive-Analysis-Using-Social-Profile-in-Online-P2P-Lending-Decision/assets/107771328/391ad17b-70c9-4fa3-bebb-4479e3ac5dc9)


   # -------------------------------------------------------------------------

    
  - Created our four target variables and their relevant features needed in their calculations.

![Screenshot 2023-07-09 232248](https://github.com/Technocolabs100/Predictive-Analysis-Using-Social-Profile-in-Online-P2P-Lending-Decision/assets/107771328/b844c01b-b0b2-40bb-bc7d-b43f13d2b5e5)


   # -------------------------------------------------------------------------


![Screenshot 2023-07-09 233230](https://github.com/Technocolabs100/Predictive-Analysis-Using-Social-Profile-in-Online-P2P-Lending-Decision/assets/107771328/8f2927aa-a313-4dd4-be2e-97d7fef9e634)


   # -------------------------------------------------------------------------


![Screenshot 2023-07-09 233302](https://github.com/Technocolabs100/Predictive-Analysis-Using-Social-Profile-in-Online-P2P-Lending-Decision/assets/107771328/86a1a1a3-cf6b-4aba-9a1c-2f76718a8426)


   # -------------------------------------------------------------------------


![Screenshot 2023-07-09 233315](https://github.com/Technocolabs100/Predictive-Analysis-Using-Social-Profile-in-Online-P2P-Lending-Decision/assets/107771328/d1752f9f-7d91-4074-ae3b-fb24de81262f)


   # -------------------------------------------------------------------------


![Screenshot 2023-07-09 233403](https://github.com/Technocolabs100/Predictive-Analysis-Using-Social-Profile-in-Online-P2P-Lending-Decision/assets/107771328/6b33f72e-83a9-48da-98ed-eb47f16a9caf)


   # -------------------------------------------------------------------------


![Screenshot 2023-07-09 233444](https://github.com/Technocolabs100/Predictive-Analysis-Using-Social-Profile-in-Online-P2P-Lending-Decision/assets/107771328/a7a5d300-75dd-49a7-a174-6ec8d4ca9746)


   # -------------------------------------------------------------------------


![Screenshot 2023-07-09 233808](https://github.com/Technocolabs100/Predictive-Analysis-Using-Social-Profile-in-Online-P2P-Lending-Decision/assets/107771328/046701ca-c4a5-416b-b77a-510f2fd39369)


   # -------------------------------------------------------------------------


![Screenshot 2023-07-09 233519](https://github.com/Technocolabs100/Predictive-Analysis-Using-Social-Profile-in-Online-P2P-Lending-Decision/assets/107771328/8a1e4989-640e-43e2-a648-d952b7239c2a)


   # -------------------------------------------------------------------------


![Screenshot 2023-07-09 233705](https://github.com/Technocolabs100/Predictive-Analysis-Using-Social-Profile-in-Online-P2P-Lending-Decision/assets/107771328/00938682-1935-4cb6-bef5-72a30bf1a2be)


   # -------------------------------------------------------------------------


![Screenshot 2023-07-09 233728](https://github.com/Technocolabs100/Predictive-Analysis-Using-Social-Profile-in-Online-P2P-Lending-Decision/assets/107771328/aa50d9c8-4f9a-4d08-bdfe-29230cd95800)


   # -------------------------------------------------------------------------


![Screenshot 2023-07-09 233748](https://github.com/Technocolabs100/Predictive-Analysis-Using-Social-Profile-in-Online-P2P-Lending-Decision/assets/107771328/495666c5-8b1c-4fb9-a9d2-ae7f03a351e6)


   # -------------------------------------------------------------------------


# EDA ( Exploratory Data Analysis )

We used different plots to visualize the data and see the relationship between features : 

  # Univariate Analysis 

   - It explores each variable in the data set, separately.

   ![Screenshot 2023-07-09 230641](https://github.com/Technocolabs100/Predictive-Analysis-Using-Social-Profile-in-Online-P2P-Lending-Decision/assets/107771328/42d5ece7-e810-4aa6-9fa7-4bbd533adf0c)


   # -------------------------------------------------------------------------

   
   ![Screenshot 2023-07-09 230618](https://github.com/Technocolabs100/Predictive-Analysis-Using-Social-Profile-in-Online-P2P-Lending-Decision/assets/107771328/2a889558-9ffd-4314-91cc-3a318340225a)


   # -------------------------------------------------------------------------


   ![Screenshot 2023-07-09 230857](https://github.com/Technocolabs100/Predictive-Analysis-Using-Social-Profile-in-Online-P2P-Lending-Decision/assets/107771328/0b18646f-5876-4892-90de-d0655705c9ff)


   # -------------------------------------------------------------------------


   ![Screenshot 2023-07-10 192231](https://github.com/Technocolabs100/Predictive-Analysis-Using-Social-Profile-in-Online-P2P-Lending-Decision/assets/107771328/3a5405ea-fe4d-4323-9c9a-6fdc561548c7)


   # -------------------------------------------------------------------------


   ![Screenshot 2023-07-09 230912](https://github.com/Technocolabs100/Predictive-Analysis-Using-Social-Profile-in-Online-P2P-Lending-Decision/assets/107771328/28dc497c-9adf-4fa1-9af4-cbd95f930bad)

    
     
  # Bivariate Analysis

   - It is a statistical method examining how two different features in the data are related.

     ![Screenshot 2023-07-09 231119](https://github.com/Technocolabs100/Predictive-Analysis-Using-Social-Profile-in-Online-P2P-Lending-Decision/assets/107771328/da9f4326-a073-4b72-be86-62c1be566589)


   # -------------------------------------------------------------------------


   ![Screenshot 2023-07-09 231202](https://github.com/Technocolabs100/Predictive-Analysis-Using-Social-Profile-in-Online-P2P-Lending-Decision/assets/107771328/c70fa066-dc22-4b17-82cb-0273de28c5fa)


   # -------------------------------------------------------------------------


   ![Screenshot 2023-07-09 231658](https://github.com/Technocolabs100/Predictive-Analysis-Using-Social-Profile-in-Online-P2P-Lending-Decision/assets/107771328/733c3ec5-5068-45f1-b965-41a29d3bf76e)


   # -------------------------------------------------------------------------


   ![Screenshot 2023-07-09 231620](https://github.com/Technocolabs100/Predictive-Analysis-Using-Social-Profile-in-Online-P2P-Lending-Decision/assets/107771328/a160367a-65f3-4377-9892-ef706a593311)


   # -------------------------------------------------------------------------


   ![Screenshot 2023-07-09 231545](https://github.com/Technocolabs100/Predictive-Analysis-Using-Social-Profile-in-Online-P2P-Lending-Decision/assets/107771328/0530943a-1b09-41e7-a01f-4ffd33c7dcb7)


   # -------------------------------------------------------------------------


   ![Screenshot 2023-07-09 231448](https://github.com/Technocolabs100/Predictive-Analysis-Using-Social-Profile-in-Online-P2P-Lending-Decision/assets/107771328/2fba24ca-9000-45b9-acb2-28aecbb44111)


   # -------------------------------------------------------------------------


   ![Screenshot 2023-07-09 231411](https://github.com/Technocolabs100/Predictive-Analysis-Using-Social-Profile-in-Online-P2P-Lending-Decision/assets/107771328/6896c3e4-be00-40e7-87f0-b835b2f42058)


   # -------------------------------------------------------------------------  


   ![Screenshot 2023-07-09 231321](https://github.com/Technocolabs100/Predictive-Analysis-Using-Social-Profile-in-Online-P2P-Lending-Decision/assets/107771328/41cd0998-dad2-4249-8e56-1f9f862c5f0d)


   # -------------------------------------------------------------------------    


   ![Screenshot 2023-07-09 231202](https://github.com/Technocolabs100/Predictive-Analysis-Using-Social-Profile-in-Online-P2P-Lending-Decision/assets/107771328/0adf52f4-85e7-4366-a5fb-a45ae9674325)

   
   # -------------------------------------------------------------------------
       
    
  # Multivariate Analysis 

   - It involves evaluating multiple variables in the data (more than two) to identify any possible association among them.


   ![Screenshot 2023-07-09 231026](https://github.com/Technocolabs100/Predictive-Analysis-Using-Social-Profile-in-Online-P2P-Lending-Decision/assets/107771328/a88d02b3-7c12-4382-b0d2-b273f6762b6c)





# Feature Engineering

  # Drooping features 

   - We dropped all the features used in the four target variables calculation. Besides, dropping not important features.

  ![Screenshot 2023-07-09 233955](https://github.com/Technocolabs100/Predictive-Analysis-Using-Social-Profile-in-Online-P2P-Lending-Decision/assets/107771328/5464aefe-bce4-450e-9aae-f234854f19bf)


   # -------------------------------------------------------------------------

  
  # Feature correlation 

   - We dropped the features whuch have high correlation with our target variables and for the others we dropped all features and kept one for
 each heat map.

  ![Screenshot 2023-07-09 234252](https://github.com/Technocolabs100/Predictive-Analysis-Using-Social-Profile-in-Online-P2P-Lending-Decision/assets/107771328/75c7e63c-2aa5-4ef1-9ba2-196225666e85)


   # -------------------------------------------------------------------------

  
![Screenshot 2023-07-09 234225](https://github.com/Technocolabs100/Predictive-Analysis-Using-Social-Profile-in-Online-P2P-Lending-Decision/assets/107771328/1b5a4c9e-5cd4-40bf-9eb4-429b8cd67d57)


   # -------------------------------------------------------------------------
 

  ![Screenshot 2023-07-09 234151](https://github.com/Technocolabs100/Predictive-Analysis-Using-Social-Profile-in-Online-P2P-Lending-Decision/assets/107771328/4fcbed8e-01e2-4494-a931-fc36efd16578)


   # -------------------------------------------------------------------------

 
  # New features creation

   - We created two new features based on existing one in the data to be easily used.

  ![Screenshot 2023-07-09 234549](https://github.com/Technocolabs100/Predictive-Analysis-Using-Social-Profile-in-Online-P2P-Lending-Decision/assets/107771328/e4079e76-e7e5-4f97-b7c0-c0a3e01aa922)


   # -------------------------------------------------------------------------

  
![Screenshot 2023-07-09 234438](https://github.com/Technocolabs100/Predictive-Analysis-Using-Social-Profile-in-Online-P2P-Lending-Decision/assets/107771328/a634ad7c-a1e6-47eb-93ef-d58311838264)


   # -------------------------------------------------------------------------

 
![Screenshot 2023-07-09 234424](https://github.com/Technocolabs100/Predictive-Analysis-Using-Social-Profile-in-Online-P2P-Lending-Decision/assets/107771328/2993cb7f-e21a-4be9-a4a3-c57428256146)


   # -------------------------------------------------------------------------

 
![Screenshot 2023-07-09 234340](https://github.com/Technocolabs100/Predictive-Analysis-Using-Social-Profile-in-Online-P2P-Lending-Decision/assets/107771328/9a2b6ade-403e-4b2f-a6a6-ca9e1292e89e)


  # Data Encoding 

   - We used label encoding for multi-categories features and binary encoding for features which contains ( true , false).


![Screenshot 2023-07-09 234715](https://github.com/Technocolabs100/Predictive-Analysis-Using-Social-Profile-in-Online-P2P-Lending-Decision/assets/107771328/ceaa20c3-8bd0-4b53-9156-d4457fa854a9)

   # -------------------------------------------------------------------------


![Screenshot 2023-07-09 234642](https://github.com/Technocolabs100/Predictive-Analysis-Using-Social-Profile-in-Online-P2P-Lending-Decision/assets/107771328/61e8c417-029f-4e04-ad77-2757acaf8b6d)


  # Dropping target variables for data

   - The four target variables were removed from the data and stored in another variables.


![Screenshot 2023-07-09 234735](https://github.com/Technocolabs100/Predictive-Analysis-Using-Social-Profile-in-Online-P2P-Lending-Decision/assets/107771328/c7dd9147-1ca9-40ab-b33a-6ab3481e46cb)


  # Checking for nulls in target variables

   - We checked for null values existence in the the four targe variables. We reakizes tha there are missing values in 'EMI' target variable. So, we replaced
     them with the median value.


  ![Screenshot 2023-07-09 234748](https://github.com/Technocolabs100/Predictive-Analysis-Using-Social-Profile-in-Online-P2P-Lending-Decision/assets/107771328/d1808d8e-aeba-4e3d-9dc9-17a408113433)

  
   # -------------------------------------------------------------------------


  ![Screenshot 2023-07-09 235831](https://github.com/Technocolabs100/Predictive-Analysis-Using-Social-Profile-in-Online-P2P-Lending-Decision/assets/107771328/d2d13b4a-4454-4932-b02b-c132b727805f)


  # -------------------------------------------------------------------------

  
 ![Screenshot 2023-07-09 235734](https://github.com/Technocolabs100/Predictive-Analysis-Using-Social-Profile-in-Online-P2P-Lending-Decision/assets/107771328/a6952c57-98e8-48c7-a5db-68aefb8737ff)



  # Feature Selection 

   - We used three approaches tp get the relevant features for each target varaiable. These approaches are : 
      1) MI classification , chi-squared , Extra tree classifier for #binary target variable :
         - Then, we took the top 20 relevant features according to each approach.

     
      ![Screenshot 2023-07-09 234846](https://github.com/Technocolabs100/Predictive-Analysis-Using-Social-Profile-in-Online-P2P-Lending-Decision/assets/107771328/7790f730-a289-4a75-9a1a-11cd3a6d1d1e)

     
         -  Then, for the relevant features we took the intersection between the three approaches.
 
      ![Screenshot 2023-07-09 234831](https://github.com/Technocolabs100/Predictive-Analysis-Using-Social-Profile-in-Online-P2P-Lending-Decision/assets/107771328/7c30030d-fe5a-4148-9087-670191a8aa97)




      2) MI regression , f_regression in univariate selection , Extra tree regressor for continous target variables :
         - Then we took the top 20 relevant features according to each approach for each target variable.

      # EMI

      ![Screenshot 2023-07-12 034710](https://github.com/Technocolabs100/Predictive-Analysis-Using-Social-Profile-in-Online-P2P-Lending-Decision/assets/107771328/024ecb05-933d-4162-8782-71a1b9cccc34)



      # ROI

     
     ![Screenshot 2023-07-12 034740](https://github.com/Technocolabs100/Predictive-Analysis-Using-Social-Profile-in-Online-P2P-Lending-Decision/assets/107771328/f4cc3114-450b-4d54-ab05-54bcb6672329)


      # ELA

         
      ![Screenshot 2023-07-12 034811](https://github.com/Technocolabs100/Predictive-Analysis-Using-Social-Profile-in-Online-P2P-Lending-Decision/assets/107771328/2def7810-ef43-4fc0-994c-c5ee9400d23d)


       
                 
  - Then, for the relevant features for each target variable, we took the union between the three approaches relevamt features.
           
      # EMI


     ![Screenshot 2023-07-12 034722](https://github.com/Technocolabs100/Predictive-Analysis-Using-Social-Profile-in-Online-P2P-Lending-Decision/assets/107771328/12a44a5f-df8d-48a5-bfcb-a36c0d4d261f)



      # ROI


      ![Screenshot 2023-07-12 034753](https://github.com/Technocolabs100/Predictive-Analysis-Using-Social-Profile-in-Online-P2P-Lending-Decision/assets/107771328/e337935c-e191-4ec3-b941-e7ce775817e2)

           

      # ELA


      ![Screenshot 2023-07-12 034955](https://github.com/Technocolabs100/Predictive-Analysis-Using-Social-Profile-in-Online-P2P-Lending-Decision/assets/107771328/84abe77e-1faf-4d9d-8ce6-d6e39c27ac44)



     - Finally, for all of the three target variables we took the union between all the resulted features for the three target variables.
        

      ![Screenshot 2023-07-12 033945](https://github.com/Technocolabs100/Predictive-Analysis-Using-Social-Profile-in-Online-P2P-Lending-Decision/assets/107771328/6e13479b-9faf-49fc-8dd9-a4e94c00927b)



         

  # Standard Scaling 

   - After getting all relevant features for all of our target variables, we created new dataframes : one containing relevant features for binary prediction and other containing relevant features for regression prediction. Then we took a copy from each one so that we have a copy used in modeling after applying standard scaling on it and another one used in pipelining for both typed of predictions.

      1) For binary target variable ( LoanStatus ) 

      
      ![Screenshot 2023-07-09 235028](https://github.com/Technocolabs100/Predictive-Analysis-Using-Social-Profile-in-Online-P2P-Lending-Decision/assets/107771328/5f185348-3b48-4b43-8ebc-8692c600e23d)



      2) For the other three targe variables ( EMI , ROI , ELA )


       ![Screenshot 2023-07-10 222151](https://github.com/Technocolabs100/Predictive-Analysis-Using-Social-Profile-in-Online-P2P-Lending-Decision/assets/107771328/bd7bb1bb-b157-4407-8183-b959868a85e6)



# Model Building

We experimented with various machine learning models , including:

 # Classification models:

   # KNN 
   - accuracy score = 94 % --> this score is before Hyperparameter Tunning.


      ![Screenshot 2023-07-09 235346](https://github.com/Technocolabs100/Predictive-Analysis-Using-Social-Profile-in-Online-P2P-Lending-Decision/assets/107771328/5cc17084-3474-4178-8128-fcfd08e1ffec)

      ![Screenshot 2023-07-09 235420( 1 )](https://github.com/Technocolabs100/Predictive-Analysis-Using-Social-Profile-in-Online-P2P-Lending-Decision/assets/107771328/b2663e42-453f-4946-a495-da4cc3ff8fd5)


    
   # XGBoost Classifier 
   - accuracy score = 96.8% .
    
   ![Screenshot 2023-07-09 235647](https://github.com/Technocolabs100/Predictive-Analysis-Using-Social-Profile-in-Online-P2P-Lending-Decision/assets/107771328/058e270a-8b77-4943-9635-20fdaabbdb6e)



    
 # Multi Output Regression models:

   # Polynomial Features creation 

   ![Screenshot 2023-07-10 222222](https://github.com/Technocolabs100/Predictive-Analysis-Using-Social-Profile-in-Online-P2P-Lending-Decision/assets/107771328/776956e5-0972-4b29-a72a-a1a674c0179a)


   # Lasso regression model 
    - R-squared (R2) Score: 85.93 %


  
![Screenshot 2023-07-10 222248](https://github.com/Technocolabs100/Predictive-Analysis-Using-Social-Profile-in-Online-P2P-Lending-Decision/assets/107771328/ccdc381e-41b2-4325-a307-f15c8d96f67d)


   

   # Polynomial Regression models  

   # 1) Multiple linear regression model 
     - R-squared (R2) Score: 85.89%

     
 ![Screenshot 2023-07-10 222322](https://github.com/Technocolabs100/Predictive-Analysis-Using-Social-Profile-in-Online-P2P-Lending-Decision/assets/107771328/f09ca6e0-1a84-40d3-b4e2-30f3b6bec193)




   # 2) XGB regression  model 
     - R-squared (R2) Score: 96.46 %


   ![Screenshot 2023-07-11 004644](https://github.com/Technocolabs100/Predictive-Analysis-Using-Social-Profile-in-Online-P2P-Lending-Decision/assets/107771328/22f336c1-3213-4d3c-bb6a-4bc36c10fac3)



   # 3) Decision Tree regression model 
     - R-squared (R2) Score: 90.3 %


   ![Screenshot 2023-07-11 004550](https://github.com/Technocolabs100/Predictive-Analysis-Using-Social-Profile-in-Online-P2P-Lending-Decision/assets/107771328/3be0d4b8-589a-4abf-b7d5-d51c03888b81)




   # 4) Ridge regression model 
     - R-squared (R2) Score : 85.89 %


 ![Screenshot 2023-07-10 222505](https://github.com/Technocolabs100/Predictive-Analysis-Using-Social-Profile-in-Online-P2P-Lending-Decision/assets/107771328/29df5e4c-2821-425a-87fa-2c56cf400bf6)




- Each model was trained using the engineered features, and their performance was evaluated on a test dataset. This iterative process enabled us to identify the strengths and weaknesses of each model and select the most suitable one for our project.




# Hyperparameter Tunning

- We performed hyperparameter tuning using : Elbow method --> for knn model , grid search cv --> for ridge pipeline .This process involved adjusting various parameters within the model to optimize its performance on the training data. By fine-tuning the model's hyperparameters, we were able to achieve better results and enhance the overall accuracy of our predictions. 
- After tunning :

   1) Using elbow method , we got accuarcy for knn model : 94.3% which is a little bit higher than before tunning. 


   ![Screenshot 2023-07-09 235555 ( 1 )](https://github.com/Technocolabs100/Predictive-Analysis-Using-Social-Profile-in-Online-P2P-Lending-Decision/assets/107771328/1e4e26e9-00f6-4489-8d5e-3df9c1d7c9e0)


  # -------------------------------------------------------------------------

  
![Screenshot 2023-07-09 235535](https://github.com/Technocolabs100/Predictive-Analysis-Using-Social-Profile-in-Online-P2P-Lending-Decision/assets/107771328/84947531-062f-4c9f-9ab6-917cec30de6a)



   2) Using GridSearchCV, we got r2 score for ridge pipeline : 85.7% which is a little bit lower than before tunning.


   ![Screenshot 2023-07-12 040607](https://github.com/Technocolabs100/Predictive-Analysis-Using-Social-Profile-in-Online-P2P-Lending-Decision/assets/107771328/7d78307b-3284-4052-979b-86d564ebba96)




# Models Evaluation

We evaluated:
- The Classification models using accuracy, precision, recall, and F1-score metrics. 
 These metrics provided a comprehensive view of each model's performance, allowing us to compare them objectively and select the best model for deployment.

   # KNN model evaluation ( before tunning ) 


  ![Screenshot 2023-07-09 235420](https://github.com/Technocolabs100/Predictive-Analysis-Using-Social-Profile-in-Online-P2P-Lending-Decision/assets/107771328/694feaf0-e185-4d49-a0c8-9d2e05f81b06)


   # -------------------------------------------------------------------------


![Screenshot 2023-07-09 235433](https://github.com/Technocolabs100/Predictive-Analysis-Using-Social-Profile-in-Online-P2P-Lending-Decision/assets/107771328/e9fba522-c83b-4b92-b101-d1d773ad6c7d)



   # KNN model ( after tunning ) 


  ![Screenshot 2023-07-09 235608](https://github.com/Technocolabs100/Predictive-Analysis-Using-Social-Profile-in-Online-P2P-Lending-Decision/assets/107771328/ffbbdbf6-02dd-4996-850d-07bc1477c71f)


   # -------------------------------------------------------------------------

  
![Screenshot 2023-07-09 235555](https://github.com/Technocolabs100/Predictive-Analysis-Using-Social-Profile-in-Online-P2P-Lending-Decision/assets/107771328/3dc429ec-00ac-47f4-8d72-aee7ba52a291)


   # XGB classification model  :
   
   ![Screenshot 2023-07-09 235710](https://github.com/Technocolabs100/Predictive-Analysis-Using-Social-Profile-in-Online-P2P-Lending-Decision/assets/107771328/9c9e474e-6853-4be1-9276-7f4ed74c5802)
   


- The Multi_Output Regression models using Test accuracy, R-squared (R2) Score and Mean Squared Error.

  # Lasso Regression model


   ![Screenshot 2023-07-12 035448](https://github.com/Technocolabs100/Predictive-Analysis-Using-Social-Profile-in-Online-P2P-Lending-Decision/assets/107771328/048dfe75-6c95-4bca-aa0a-57576c960780)


  # Polynomial Regression models :
  
   # 1) Multiple linear regression model 
 

  ![Screenshot 2023-07-12 035506](https://github.com/Technocolabs100/Predictive-Analysis-Using-Social-Profile-in-Online-P2P-Lending-Decision/assets/107771328/65245773-0be8-49a2-bd2a-8387c8f7c6f9)


   # 2) XGB regression model 
    

   ![Screenshot 2023-07-12 035518](https://github.com/Technocolabs100/Predictive-Analysis-Using-Social-Profile-in-Online-P2P-Lending-Decision/assets/107771328/e2e28a10-53bc-40d2-8c46-1d93d0db6dfc)


   # 3) Decision Tree regression model 
  

![Screenshot 2023-07-12 035538](https://github.com/Technocolabs100/Predictive-Analysis-Using-Social-Profile-in-Online-P2P-Lending-Decision/assets/107771328/db41d17b-2024-4070-b5bb-09948706504c)



   # 4) Ridge regression model 
    
  
   ![Screenshot 2023-07-12 035555](https://github.com/Technocolabs100/Predictive-Analysis-Using-Social-Profile-in-Online-P2P-Lending-Decision/assets/107771328/f82efcf1-8b3e-4614-bccc-3a0036e9b93b)





# Pipelining

After data splitting ( x --> relevant features data , y --> target variable/s ), we created :

   # Classification pipelines : 

     1) KNN pipeline was created using standard scaling , PCA with number of components = 4 , KNN model. We got accuarcy : 94.78%


   ![Screenshot 2023-07-11 010715](https://github.com/Technocolabs100/Predictive-Analysis-Using-Social-Profile-in-Online-P2P-Lending-Decision/assets/107771328/8cd6c8e6-ab5d-4279-ba3f-c7dc29ca5e62)
   

   # -------------------------------------------------------------------------


   ![Screenshot 2023-07-11 011055](https://github.com/Technocolabs100/Predictive-Analysis-Using-Social-Profile-in-Online-P2P-Lending-Decision/assets/107771328/d2714f98-189e-4b47-81d2-c69fcbde8bc3)


    2) XGB classifier pipeline was created using standard scaling , PCA with number of components = 4 , XGB classifier model. We got accuarcy : 95.22%


   ![Screenshot 2023-07-11 011126](https://github.com/Technocolabs100/Predictive-Analysis-Using-Social-Profile-in-Online-P2P-Lending-Decision/assets/107771328/e00c3b16-faa4-478c-9bd9-81c6a3c341e2)


   # -------------------------------------------------------------------------


   ![Screenshot 2023-07-11 011139](https://github.com/Technocolabs100/Predictive-Analysis-Using-Social-Profile-in-Online-P2P-Lending-Decision/assets/107771328/1752409a-0bdf-4712-8fbe-86f278bf7e4b)


   # Multi-regression pipelines : 

  1) MultiTaskLasso pipeline was created using standard scaling , polynomial features with degree = 2 ,  MultiTaskLasso model with alpha=1e-05. We got accuarcy : 86.35%

    
   ![Screenshot 2023-07-12 041043](https://github.com/Technocolabs100/Predictive-Analysis-Using-Social-Profile-in-Online-P2P-Lending-Decision/assets/107771328/c388b908-2c84-43f5-8ec8-77b2fc938b87)

    

  2) Multiple Linear Regression pipeline was created using standard scaling , polynomial features with degree = 2 , LinearRegression model. We got accuarcy : 86.33%


   ![Screenshot 2023-07-12 041202](https://github.com/Technocolabs100/Predictive-Analysis-Using-Social-Profile-in-Online-P2P-Lending-Decision/assets/107771328/011c13bd-6001-4cc3-a7c6-0313fa2ad64c)


  4) XGB Regression pipeline was created using standard scaling , XGBRegressor model. We got accuarcy : 96.5%


   ![Screenshot 2023-07-12 041215](https://github.com/Technocolabs100/Predictive-Analysis-Using-Social-Profile-in-Online-P2P-Lending-Decision/assets/107771328/72f1a766-ed3e-44d4-b689-e7cd3a15322d)


  6) Decision Tree Regression pipeline was created using standard scaling  , DecisionTreeRegressor model. We got accuarcy : 90.4%


   ![Screenshot 2023-07-12 041236](https://github.com/Technocolabs100/Predictive-Analysis-Using-Social-Profile-in-Online-P2P-Lending-Decision/assets/107771328/43a87798-6729-4344-82d6-02ff53e05d7c)


  8) Ridge Regression pipeline was created using standard scaling , olynomial features with degree = 2  , Ridge model. We got accuarcy : 86.34%


   ![Screenshot 2023-07-12 041247](https://github.com/Technocolabs100/Predictive-Analysis-Using-Social-Profile-in-Online-P2P-Lending-Decision/assets/107771328/10fa31f4-791d-4b84-b8cc-2e09e17f396c)


   __Note : We concluded that by using polynomial features in XGB Regressor , Decision tree regressor models and pipelines , the scores decreased. So, we used it only in the other multi-regression models.__ 



# Deployment 

Once we have selected and fine-tuned our model : 
- We saved our two models with highest and best scores one is for classification prediction and the other is for milti-regression prediction using pickle. We provided detailed steps and code for deploying the model.


   ![Screenshot 2023-07-12 041303](https://github.com/Technocolabs100/Predictive-Analysis-Using-Social-Profile-in-Online-P2P-Lending-Decision/assets/107771328/b7c8415f-6038-4ae9-8000-100d0bbbe937)


- This included deploying the model on a local server, creating a streamlit website with user-friendly interface and used the saved models files to run the modeling code for prediction.


![WhatsApp Image 2023-07-13 at 1 46 55 AM](https://github.com/Technocolabs100/Predictive-Analysis-Using-Social-Profile-in-Online-P2P-Lending-Decision/assets/107771328/4ef82314-4b1a-4ce9-8e9b-ae3b255febd4)


   # -------------------------------------------------------------------------

  
  
![WhatsApp Image 2023-07-13 at 1 43 35 AM](https://github.com/Technocolabs100/Predictive-Analysis-Using-Social-Profile-in-Online-P2P-Lending-Decision/assets/107771328/ad34a187-450c-46fa-8a94-adebe9feca22)


 # -------------------------------------------------------------------------


![WhatsApp Image 2023-07-13 at 1 44 23 AM](https://github.com/Technocolabs100/Predictive-Analysis-Using-Social-Profile-in-Online-P2P-Lending-Decision/assets/107771328/17379d12-1e24-48c2-9672-6114831241d5)


# -------------------------------------------------------------------------


![WhatsApp Image 2023-07-13 at 1 49 16 AM](https://github.com/Technocolabs100/Predictive-Analysis-Using-Social-Profile-in-Online-P2P-Lending-Decision/assets/107771328/6fe52610-ef32-4464-9427-b7e26d62a4f0)


# -------------------------------------------------------------------------


- Upon using our website :
  
    # LoanStatus prediction


  ![WhatsApp Image 2023-07-13 at 1 53 56 AM](https://github.com/Technocolabs100/Predictive-Analysis-Using-Social-Profile-in-Online-P2P-Lending-Decision/assets/107771328/4efb8d7f-6355-42ea-a120-98514ea45855)


# -------------------------------------------------------------------------


![WhatsApp Image 2023-07-13 at 1 57 29 AM](https://github.com/Technocolabs100/Predictive-Analysis-Using-Social-Profile-in-Online-P2P-Lending-Decision/assets/107771328/a03bcdc3-657b-46d0-ae0c-c920cc1accbb)



   # EMI , ROI , ELA prediction


![WhatsApp Image 2023-07-13 at 2 05 55 AM](https://github.com/Technocolabs100/Predictive-Analysis-Using-Social-Profile-in-Online-P2P-Lending-Decision/assets/107771328/36108a80-f7df-4b84-ab3c-b2bdcf4acada)
