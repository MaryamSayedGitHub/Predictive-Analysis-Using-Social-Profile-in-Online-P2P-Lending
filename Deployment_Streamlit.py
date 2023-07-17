import streamlit as st
import requests
from streamlit_lottie import st_lottie
import pandas as pd
import numpy as np
import pickle
import requests
import json
from streamlit_option_menu import option_menu

def load_lottie(url):
    try:
        r = requests.get(url)
        r.raise_for_status()
        return json.loads(r.content.decode('utf-8'))
    except json.JSONDecodeError as e:
        print(f"JSON decoding error: {e}")
        return None
    except requests.exceptions.RequestException as e:
        print(f"Request error: {e}")
        return None

with st.sidebar:
  selected=option_menu(
    menu_title=" P2P Lending Platform ",
    options=['Home Page','Loan Status','EMI_ROI_ELA']
  )
if selected=='Home Page':

        st.write("#  We're glad to see you here....")

        st_lottie(load_lottie('https://assets6.lottiefiles.com/packages/lf20_mKMcjgVTY6.json'), speed=2, height=500,width=1400)

        
                
if selected=='Loan Status':
    #here the code of the page of Loun Status
    def prepare_input_data_for_model(CreditScoreRangeLower,CurrentDelinquencies,EmploymentStatus ,
                                           LP_CustomerPayments,LP_GrossPrincipalLoss,LP_NetPrincipalLoss,
                                           LP_NonPrincipalRecoverypayments,Occupation,ProsperRating_numeric
                                           ,TotalInquiries):
      A = [CreditScoreRangeLower,CurrentDelinquencies,EmploymentStatus ,
                                           LP_CustomerPayments,LP_GrossPrincipalLoss,LP_NetPrincipalLoss,
                                           LP_NonPrincipalRecoverypayments,Occupation,ProsperRating_numeric
                                           ,TotalInquiries]
      sample = np.array(A).reshape(-1,len(A))
      return sample

    with open("classification_model.pkl", "rb") as f:
       loaded_model = pickle.load(f)
     
    st.write('# Loan Status')

    st.write('---')
    st.subheader('Enter your details to predict your Loun Status')

    name = st.text_input('Name:')
    CreditScoreRangeLower = st.number_input('Credit Score Range Lower: ' )
    CurrentDelinquencies = st.number_input('Current Delinquencies: ')
    EmploymentStatus = st.number_input('Employment Status: ')
    LP_CustomerPayments = st.number_input('LP_Customer Payments: ')
    LP_GrossPrincipalLoss = st.number_input('LP_Gross Principal Loss: ')
    LP_NetPrincipalLoss = st.number_input('LP_Net Principal Loss: ')
    LP_NonPrincipalRecoverypayments = st.number_input('LP_Non Principal Recovery payments: ')
    Occupation = st.number_input('Occupation: ')
    ProsperRating_numeric = st.number_input('Prosper Rating_numeric: ')
    TotalInquiries = st.number_input('Total Inquiries: ')
    
    sample = prepare_input_data_for_model(CreditScoreRangeLower,CurrentDelinquencies,EmploymentStatus ,
                                           LP_CustomerPayments,LP_GrossPrincipalLoss,LP_NetPrincipalLoss,
                                           LP_NonPrincipalRecoverypayments,Occupation,ProsperRating_numeric
                                           ,TotalInquiries)
            

    if st.button('Predict'):
                pred_Y = loaded_model.predict(sample)
                if pred_Y == 1:
                    st.write("## Predicted Status ")
                    st.write('### 1 : Congratulations,', name,'will be able to repay the loan.')
                    st.balloons()
                else:
                    st.write("### 0 : Unfortunately, ", name," won't be able to repay the loan.")
        
if selected=='EMI_ROI_ELA':
    def prepare_input_data_for_model(AmountDelinquent, AvailableBankcardCredit,
        BankcardUtilization, BorrowerState, CreditScoreRangeLower,
        CurrentDelinquencies, CurrentlyInGroup, DebtToIncomeRatio,
        DelinquenciesLast7Years, EmploymentStatus,
        EmploymentStatusDuration, EstimatedReturn, IncomeRange,
        IncomeVerifiable, Investors, IsBorrowerHomeowner,
        LP_CollectionFees, LP_CustomerPayments,
        LP_GrossPrincipalLoss, LP_NetPrincipalLoss,
        LP_NonPrincipalRecoverypayments, LP_ServiceFees, LenderYield,
        ListingCategory_numeric, Occupation, OpenCreditLines,
        OriginationQuarter, OriginationYear, ProsperRating_numeric,
         RevolvingCreditBalance,
        TotalInquiries, TotalTrades,
        TradesNeverDelinquent_percentage):
      A = [AmountDelinquent, AvailableBankcardCredit,
        BankcardUtilization, BorrowerState, CreditScoreRangeLower,
        CurrentDelinquencies, CurrentlyInGroup, DebtToIncomeRatio,
        DelinquenciesLast7Years, EmploymentStatus,
        EmploymentStatusDuration, EstimatedReturn, IncomeRange,
        IncomeVerifiable, Investors, IsBorrowerHomeowner,
        LP_CollectionFees, LP_CustomerPayments,
        LP_GrossPrincipalLoss, LP_NetPrincipalLoss,
        LP_NonPrincipalRecoverypayments, LP_ServiceFees, LenderYield,
        ListingCategory_numeric, Occupation, OpenCreditLines,
        OriginationQuarter, OriginationYear, ProsperRating_numeric,
        RevolvingCreditBalance,
                TotalInquiries, TotalTrades,
        TradesNeverDelinquent_percentage]
      sample = np.array(A).reshape(-1,len(A))
      return sample

    with open("multi-regressor_model.pkl", "rb") as f:
       loaded_model_regression = pickle.load(f)
     
    st.write('# EMI_ROI_ELA')

    st.write('---')
    st.subheader('Enter your details to predict')
    name = st.text_input('Name:')
    AmountDelinquent = st.number_input('AmountDelinquent : ' )
    AvailableBankcardCredit = st.number_input('AvailableBankcardCredit : ')
    BankcardUtilization = st.number_input('BankcardUtilization : ')
    BorrowerState = st.number_input('BorrowerState : ')
    CreditScoreRangeLower = st.number_input('CreditScoreRangeLower : ')
    CurrentDelinquencies = st.number_input('CurrentDelinquencies : ')
    CurrentlyInGroup = st.number_input('CurrentlyInGroup : ')
    DebtToIncomeRatio = st.number_input('DebtToIncomeRatio : ')
    DelinquenciesLast7Years = st.number_input('DelinquenciesLast7Years : ')
    EmploymentStatus = st.number_input('EmploymentStatus : ')
    EmploymentStatusDuration = st.number_input('EmploymentStatusDuration : ', key='employment_status_duration')
    EstimatedReturn = st.number_input('Estimated Return: ', key='Estimated_Return')
    IncomeRange = st.number_input('IncomeRange : ', key='IncomeRange')
    IncomeVerifiable = st.number_input('IncomeVerifiable : ', key='IncomeVerifiable')
    Investors = st.number_input('Investors : ', key='Investors')
    IsBorrowerHomeowner = st.number_input('IsBorrowerHomeowner : ', key='IsBorrowerHomeowner')
    LP_CollectionFees = st.number_input('LP_CollectionFees : ', key='LP_CollectionFees')
    LP_CustomerPayments = st.number_input('LP_CustomerPayments : ', key='LP_CustomerPayments')
    LP_GrossPrincipalLoss = st.number_input('LP_GrossPrincipalLoss : ', key='LP_GrossPrincipalLoss')
    LP_NetPrincipalLoss = st.number_input('LP_Net Principal Loss: ', key='LP_NetPrincipalLoss')
    LP_NonPrincipalRecoverypayments = st.number_input('LP_Non Principal Recovery payments: ', key='LP_NonPrincipalRecoverypayments')
    LP_ServiceFees = st.number_input('LP_Service Fees : ', key='LP_ServiceFees')
    LenderYield = st.number_input('Lender Yield: ', key='LenderYield')
    ListingCategory_numeric = st.number_input('Listing Category_numeric: ', key='ListingCategory_numeric')
    Occupation = st.number_input('Occupation: ', key='Occupation')
    OpenCreditLines = st.number_input('Open Credit Lines: ', key='OpenCreditLines')
    OriginationQuarter = st.number_input('Origination Quarter: ', key='OriginationQuarter')
    OriginationYear = st.number_input('Origination Year: ', key='OriginationYear')
    ProsperRating_numeric = st.number_input('Prosper Rating_numeric: ', key='ProsperRating_numeric')
    # PublicRecordsLast10Years = st.number_input('Public Records Last 10 Years: ', key='PublicRecordsLast10Years')
    RevolvingCreditBalance = st.number_input('Revolving Credit Balance: ', key='RevolvingCreditBalance' )
    TotalInquiries = st.number_input('Total Inquiries: ', key='TotalInquiries')
    TotalTrades = st.number_input('Total Trades: ', key='TotalTrades')
    TradesNeverDelinquent_percentage = st.number_input('Trades Never Delinquent percentage: ', key='TradesNeverDelinquent_percentage')
    # TradesOpenedLast6Months = st.number_input('Trades Opened Last6 Months: ', key='Trades_Opened_Last_6_Months')
    sample = prepare_input_data_for_model(AmountDelinquent, AvailableBankcardCredit,
        BankcardUtilization, BorrowerState, CreditScoreRangeLower,
        CurrentDelinquencies, CurrentlyInGroup, DebtToIncomeRatio,
        DelinquenciesLast7Years, EmploymentStatus,
        EmploymentStatusDuration, EstimatedReturn, IncomeRange,
        IncomeVerifiable, Investors, IsBorrowerHomeowner,
        LP_CollectionFees, LP_CustomerPayments,
        LP_GrossPrincipalLoss, LP_NetPrincipalLoss,
        LP_NonPrincipalRecoverypayments, LP_ServiceFees, LenderYield,
        ListingCategory_numeric, Occupation, OpenCreditLines,
        OriginationQuarter, OriginationYear, ProsperRating_numeric,
         RevolvingCreditBalance,
        TotalInquiries, TotalTrades,
        TradesNeverDelinquent_percentage)
    if st.button('Predict'):
                pred_Y = loaded_model_regression.predict(sample)
                st.write(pred_Y)
                st.write("This value",pred_Y[0][0],"Represents  the loan ideal or desired monthly installment  payment for the loan.")
                st.write("This value",pred_Y[0][1],"Represents  the rate of return on investment.")
                st.write("This value",pred_Y[0][2],"Represents   the maximum loan amount for borrowing.")