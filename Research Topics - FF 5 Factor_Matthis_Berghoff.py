# -*- coding: utf-8 -*-
"""
Created on Thu Jun  6 13:48:52 2019

@author: Matthis Berghoff

Research Topics (Summer term 2019)

A five-factor asset pricing model (Fama/French 2014)

In the first part I have tried to replicate the main tables and regressions of the paper.
This is followed by trying to augment the FF5 Model with the momentum factor and compare
the average absolute intercepts they produce by being regressed on the different
portfolios excess returns.
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats


"""Data & data transformation"""
    #FF 5 Factors (2x3)
ff5 = pd.read_csv('F-F_Research_Data_5_Factors_2x3.csv', index_col = 0, skiprows = 2, nrows = 670) 
ff5_196307to201312 = pd.DataFrame(ff5.loc[196307:201312,:]) #FF5 with respective time period

    #Three- and four-factor models that augment Mkt-RF and SMB

ff3 = ff5_196307to201312[['Mkt-RF','SMB','HML']]
ff4_hml_rmw = ff5_196307to201312[['Mkt-RF','SMB','HML','RMW']]
ff4_hml_cma = ff5_196307to201312[['Mkt-RF','SMB','HML','CMA']]
ff4_rmw_cma = ff5_196307to201312[['Mkt-RF','SMB','RMW','CMA']]


    #Portfolios
pf_2x3_me_beme = pd.read_csv('6_Portfolios_2x3.csv', index_col = 0, skiprows = 15, nrows = 1114 )
pf_2x3_me_inv = pd.read_csv('6_Portfolios_ME_INV_2x3.csv', index_col = 0, skiprows = 16, nrows = 670)
pf_2x3_me_op = pd.read_csv('6_Portfolios_ME_OP_2x3.csv', index_col = 0, skiprows = 16, nrows = 670)

pf_5x5_me_beme = pd.read_csv('25_Portfolios_5x5.csv', index_col = 0, skiprows = 15, nrows = 1114)
pf_5x5_beme_inv = pd.read_csv('25_Portfolios_BEME_INV_5x5.csv', index_col = 0, skiprows = 15, nrows = 670)
pf_5x5_beme_op = pd.read_csv('25_Portfolios_BEME_OP_5x5.csv', index_col = 0, skiprows = 15, nrows = 670)
pf_5x5_me_inv = pd.read_csv('25_Portfolios_ME_INV_5x5.csv', index_col = 0, skiprows = 16, nrows = 670)
pf_5x5_me_op = pd.read_csv('25_Portfolios_ME_OP_5x5.csv', index_col = 0, skiprows = 16, nrows = 670)
pf_5x5_op_inv = pd.read_csv('25_Portfolios_OP_INV_5x5.csv', index_col = 0, skiprows = 18, nrows = 670)

pf_2x4x4_me_beme_inv = pd.read_csv('32_Portfolios_ME_BEME_INV_2x4x4.csv', index_col = 0, skiprows = 17, nrows = 670)
pf_2x4x4_me_beme_op = pd.read_csv('32_Portfolios_ME_BEME_OP_2x4x4.csv', index_col = 0, skiprows = 17, nrows = 670)
pf_2x4x4_me_op_inv = pd.read_csv('32_Portfolios_ME_OP_INV_2x4x4.csv', index_col = 0, skiprows = 18, nrows = 670)

    

"""Table 1: Disentanglig effects of factors"""
    #Panel A
    
#For Tables 1 and 2, first dataframes were created that cover time period from July 1963 to December 2013
#For each value in the tables change respective item (e.g. SMALL LoBM, ME BM2, etc.)

me_beme_5x5_196307to201312 = pd.DataFrame(pf_5x5_me_beme.loc[196307:201312,:]) 
avg_mth_excessreturn_table1A = (me_beme_5x5_196307to201312['SMALL LoBM'] - ff5_196307to201312['RF']).mean() 
#print(avg_mth_excessreturn_table1A)

    #Panel B
me_op_5x5_196307to201312 = pd.DataFrame(pf_5x5_me_op.loc[196307:201312,:])
avg_mth_excessreturn_table1B = (me_op_5x5_196307to201312['ME3 OP3'] - ff5_196307to201312['RF']).mean()
#print(avg_mth_excessreturn_table1B)

    # Panel C
me_inv_5x5_196307to201312 = pd.DataFrame(pf_5x5_me_inv.loc[196307:201312,:])
avg_mth_excessreturn_table1C = (me_inv_5x5_196307to201312['ME2 INV4'] - ff5_196307to201312['RF']).mean()
#print(avg_mth_excessreturn_table1C)



"""Table 2: Disentanglig effects of factors part 2"""
    #Panel A
me_beme_op_196307to201312 = pd.DataFrame(pf_2x4x4_me_beme_op.loc[196307:201312,:])
avg_mth_excessreturn_table2A = (me_beme_op_196307to201312['ME2 BM3 OP3'] - ff5_196307to201312['RF']).mean()
#print(avg_mth_excessreturn_table2A) 

    #Panel B
me_beme_inv_196307to201312 = pd.DataFrame(pf_2x4x4_me_beme_inv.loc[196307:201312,:])
avg_mth_excessreturn_table2B = (me_beme_inv_196307to201312['ME2 BM3 INV3'] - ff5_196307to201312['RF']).mean()   
#print(avg_mth_excessreturn_table2B)
    
    #Panel C
me_op_inv_196307to201312 = pd.DataFrame(pf_2x4x4_me_op_inv.loc[196307:201312,:])
avg_mth_excessreturn_table2C = (me_op_inv_196307to201312['SMALL LoOP LoINV'] - ff5_196307to201312['RF']).mean()
#print(avg_mth_excessreturn_table2C)



"""Table 4: Summary statistics for monthly factor percent returns"""
#The following for-loop prints the means, standard deviations and t-statistics for the 2x3 FF5 factors
#After that the second part of table 4 panel A is being calculated and printed
#This is followed by the  correlation matrix of the 2x3 FF5 factors in panel C

"""
    #Panel A for 2x3 Factors
for i, col in enumerate(ff5_196307to201312, start = 1):
    print(str(col) + " Mean: " + str(round(ff5_196307to201312[col].mean(),2)) 
    + ", Std dev.: " + str(round(ff5_196307to201312[col].std(),2))
    + ", t-Statistic: " + str(round(((len(ff5_196307to201312[col])) ** (1/2)) * (ff5_196307to201312[col].mean() / ff5_196307to201312[col].std()),2)))
    if i == 5: break
"""

print("\n")

me_beme_2x3_196307to201312 = pd.DataFrame(pf_2x3_me_beme.loc[196307:201312,:])
me_op_2x3_196307to201312 = pd.DataFrame(pf_2x3_me_op.loc[196307:201312,:])
me_inv_2x3_196307to201312 = pd.DataFrame(pf_2x3_me_inv.loc[196307:201312,:])

hmls_mean = round((me_beme_2x3_196307to201312['SMALL HiBM'] - me_beme_2x3_196307to201312['SMALL LoBM']).mean(),2)
hmls_std = round((me_beme_2x3_196307to201312['SMALL HiBM'] - me_beme_2x3_196307to201312['SMALL LoBM']).std(),2)
hmls_t = round(((len(me_beme_2x3_196307to201312) ** (1/2)) * (hmls_mean / hmls_std)),2)

hmlb_mean = round((me_beme_2x3_196307to201312['BIG HiBM'] - me_beme_2x3_196307to201312['BIG LoBM']).mean(),2)
hmlb_std = round((me_beme_2x3_196307to201312['BIG HiBM'] - me_beme_2x3_196307to201312['BIG LoBM']).std(),2)
hmlb_t = round(((len(me_beme_2x3_196307to201312) ** (1/2)) * (hmlb_mean / hmlb_std)),2)

hmlsb_mean = round(hmls_mean - hmlb_mean,2)
hmlsb_std = round(((me_beme_2x3_196307to201312['SMALL HiBM'] - me_beme_2x3_196307to201312['SMALL LoBM']) - (me_beme_2x3_196307to201312['BIG HiBM'] - me_beme_2x3_196307to201312['BIG LoBM'])).std(),2)
hmlsb_t = round(((len(me_beme_2x3_196307to201312) ** (1/2)) * (hmlsb_mean / hmlsb_std)),2)

rmws_mean = round((me_op_2x3_196307to201312['SMALL HiOP'] - me_op_2x3_196307to201312['SMALL LoOP']).mean(),2)
rmws_std = round((me_op_2x3_196307to201312['SMALL HiOP'] - me_op_2x3_196307to201312['SMALL LoOP']).std(),2)
rmws_t = round(((len(me_op_2x3_196307to201312) ** (1/2)) * (rmws_mean / rmws_std)),2)

rmwb_mean = round((me_op_2x3_196307to201312['BIG HiOP'] - me_op_2x3_196307to201312['BIG LoOP']).mean(),2)
rmwb_std = round((me_op_2x3_196307to201312['BIG HiOP'] - me_op_2x3_196307to201312['BIG LoOP']).std(),2)
rmwb_t = round(((len(me_op_2x3_196307to201312) ** (1/2)) * (rmwb_mean / rmwb_std)),2)

rmwsb_mean = round(rmws_mean - rmwb_mean,2)
rmwsb_std = round(((me_op_2x3_196307to201312['SMALL HiOP'] - me_op_2x3_196307to201312['SMALL LoOP']) - (me_op_2x3_196307to201312['BIG HiOP'] - me_op_2x3_196307to201312['BIG LoOP'])).std(),2)
rmwsb_t = round(((len(me_op_2x3_196307to201312) ** (1/2)) * (rmwsb_mean / rmwsb_std)),2)

cmas_mean = round((me_inv_2x3_196307to201312['SMALL LoINV'] - me_inv_2x3_196307to201312['SMALL HiINV']).mean(),2)
cmas_std = round((me_inv_2x3_196307to201312['SMALL LoINV'] - me_inv_2x3_196307to201312['SMALL HiINV']).std(),2)
cmas_t = round(((len(me_inv_2x3_196307to201312) ** (1/2)) * (cmas_mean / cmas_std)),2)

cmab_mean = round((me_inv_2x3_196307to201312['BIG LoINV'] - me_inv_2x3_196307to201312['BIG HiINV']).mean(),2)
cmab_std = round((me_inv_2x3_196307to201312['BIG LoINV'] - me_inv_2x3_196307to201312['BIG HiINV']).std(),2)
cmab_t = round(((len(me_inv_2x3_196307to201312) ** (1/2)) * (cmab_mean / cmab_std)),2)

cmasb_mean = round(cmas_mean - cmab_mean,2)
cmasb_std = round(((me_inv_2x3_196307to201312['SMALL LoINV'] - me_inv_2x3_196307to201312['SMALL HiINV']) - (me_inv_2x3_196307to201312['BIG LoINV'] - me_inv_2x3_196307to201312['BIG HiINV'])).std(),2)
cmasb_t = round(((len(me_inv_2x3_196307to201312) ** (1/2)) * (cmasb_mean / cmasb_std)),2)


"""
print("HMLs Mean: " + str(hmls_mean) +", Std. dev.: " + str(hmls_std) + (", t-Statistic: " ) + str(hmls_t))
print("HMLb Mean: " + str(hmlb_mean) +", Std. dev.: " + str(hmlb_std) + (", t-Statistic: " ) + str(hmlb_t))
print("HMLs-b Mean: " + str(hmlsb_mean) +", Std. dev.: " + str(hmlsb_std) + (", t-Statistic: " ) + str(hmlsb_t))

print("\n")

print("RMWs Mean: " + str(rmws_mean) +", Std. dev.: " + str(rmws_std) + (", t-Statistic: " ) + str(rmws_t))
print("RMWb Mean: " + str(rmwb_mean) +", Std. dev.: " + str(rmwb_std) + (", t-Statistic: " ) + str(rmwb_t))
print("RMWs-b Mean: " + str(rmwsb_mean) +", Std. dev.: " + str(rmwsb_std) + (", t-Statistic: " ) + str(rmwsb_t))

print("\n")

print("CMAs Mean: " + str(cmas_mean) +", Std. dev.: " + str(cmas_std) + (", t-Statistic: " ) + str(cmas_t))
print("CMAb Mean: " + str(cmab_mean) +", Std. dev.: " + str(cmab_std) + (", t-Statistic: " ) + str(cmab_t))
print("CMAs-b Mean: " + str(cmasb_mean) +", Std. dev.: " + str(cmasb_std) + (", t-Statistic: " ) + str(cmasb_t))
"""

    #Panel C: Correlations between different 2x3 factors

ff5_196307to201312_ex_RF = ff5_196307to201312[['Mkt-RF','SMB','HML','RMW','CMA']]

corr_matrix_ff5 = round(ff5_196307to201312_ex_RF.corr(),2)

#print(corr_matrix_ff5)



"""Table 5: Summary statistics for tests of three-, four, and five factor models !?!?!? Fokus auf Intercepts"""
#This section  prints out the intercepts of regressing a three, four or five factor model on the different portfolios
#For different factor models and portfolios, the column names have to be adjusted respectively
#I have manually put the intercepts in an array and the calculated the absolute average of them
#Unfortunately I was only able to replicate the average absolute intercepts here 

excess_return_table5 = pd.DataFrame(me_beme_5x5_196307to201312['BIG HiBM']-ff5_196307to201312['RF'])
endog = excess_return_table5[0]
exog = sm.add_constant(ff5_196307to201312[['Mkt-RF','SMB','HML']])
reg1 = sm.OLS(endog, exog)
results = reg1.fit()
#print(results.params['const'])


#Intercepts of regressing FF3 on 25 Size-B/M portfolios

results_intercepts = np.array([-0.48,	0.00,	-0.02,	0.18,	0.15,
-0.18,	-0.02,	0.09,	0.09,	-0.02,
-0.08,	0.09,	0.00,	0.09,	0.10,
0.12,	-0.07,	-0.06,	0.10,	-0.09,
0.17,	0.02,	-0.01,	-0.24,	-0.18])

abs_intercepts_results = np.absolute(results_intercepts)
abs_avg_intercepts_results = round(np.average(abs_intercepts_results),3)

#print(abs_avg_intercepts_results)



"""#Table 6: Using four factors regressions to explain average returns on the fifth factor:"""


endog_mktrf = ff5_196307to201312['Mkt-RF']
exog_ff5_wout_mktrf = sm.add_constant(ff5_196307to201312[['SMB','HML','RMW','CMA']])
reg_mktrf = sm.OLS(endog_mktrf, exog_ff5_wout_mktrf)
results_mktrf = reg_mktrf.fit()
#print(results_mktrf.summary())

endog_smb = ff5_196307to201312['SMB']
exog_ff5_wout_smb = sm.add_constant(ff5_196307to201312[['Mkt-RF','HML','RMW','CMA']])
reg_smb = sm.OLS(endog_smb, exog_ff5_wout_smb)
results_smb = reg_smb.fit()
#print(results_smb.summary())

endog_hml = ff5_196307to201312['HML']
exog_ff5_wout_hml = sm.add_constant(ff5_196307to201312[['Mkt-RF','SMB','RMW','CMA']])
reg_hml = sm.OLS(endog_hml, exog_ff5_wout_hml)
results_hml = reg_hml.fit()
#print(results_hml.summary())

endog_rmw = ff5_196307to201312['RMW']
exog_ff5_wout_rmw = sm.add_constant(ff5_196307to201312[['Mkt-RF','SMB','HML','CMA']])
reg_rmw = sm.OLS(endog_rmw, exog_ff5_wout_rmw)
results_rmw = reg_rmw.fit()
#print(results_rmw.summary())

endog_cma = ff5_196307to201312['CMA']
exog_ff5_wout_cma = sm.add_constant(ff5_196307to201312[['Mkt-RF','SMB','HML','RMW']])
reg_cma = sm.OLS(endog_cma, exog_ff5_wout_cma)
results_cma = reg_cma.fit()
#print(results_cma.params['const'])


"""Table 7: Regressions for 25 value-weight Size-B/M portfolios""" 

#Change column name of respective 5x5 portfolio to get the wanted intercept and t-statistics

#Panel A: Three-factor intercets for Rm-Rf, SMB, and HML
excess_return_table7A = pd.DataFrame(me_beme_5x5_196307to201312['SMALL LoBM'] - ff5_196307to201312['RF'])
endog_table7A = excess_return_table7A[0]
exog_table7A = sm.add_constant(ff5_196307to201312[['Mkt-RF', 'SMB', 'HML']])
reg_table7A = sm.OLS(endog_table7A, exog_table7A)
results_table7A = reg_table7A.fit()
#print(results_table7A.summary())



#Panel B: Five-factor coefficients: Intercepts, HML(h), RMW(r), CMA(c)

#Change column name of respective 5x5 portfolio to get the wanted intercepts, slopes and t-statistics

excess_return_table7B = pd.DataFrame(me_beme_5x5_196307to201312['ME2 BM1'] - ff5_196307to201312['RF'])
endog_table7B = excess_return_table7B[0]
exog_table7B = sm.add_constant(ff5_196307to201312[['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA']])
reg_table7B = sm.OLS(endog_table7B, exog_table7B)
results_table7B = reg_table7B.fit()
#print(results_table7B.summary())



"""Table 8: Time-series averages"""

#Change the column name in the print section below to get the wanted value in table 8

size_bm_vw_avg_beme = pd.read_csv('25_Portfolios_5x5.csv', index_col = 0, skiprows = 4683, nrows = 1114)
size_bm_vw_avg_beme_196307to201312 = size_bm_vw_avg_beme.loc[196307:201312,:]

size_bm_vw_avg_op = pd.read_csv('25_Portfolios_5x5.csv', index_col = 0, skiprows = 6926, nrows = 670)
size_bm_vw_avg_op_196307to201312 = size_bm_vw_avg_op.loc[196307:201312,:]

size_bm_vw_avg_inv = pd.read_csv('25_Portfolios_5x5.csv', index_col = 0, skiprows = 7603, nrows = 670)
size_bm_vw_avg_inv_196307to201312 = size_bm_vw_avg_inv.loc[196307:201312,:]


size_op_vw_avg_beme = pd.read_csv('25_Portfolios_ME_OP_5x5.csv', index_col = 0, skiprows = 2834, nrows = 670)
size_op_vw_avg_beme_196307to201312 = size_op_vw_avg_beme.loc[196307:201312,:]

size_op_vw_avg_op = pd.read_csv('25_Portfolios_ME_OP_5x5.csv', index_col = 0, skiprows = 4189, nrows = 670)
size_op_vw_avg_op_196307to201312 = size_op_vw_avg_op.loc[196307:201312,:]

size_op_vw_avg_inv = pd.read_csv('25_Portfolios_ME_OP_5x5.csv', index_col = 0, skiprows = 4866, nrows = 670)
size_op_vw_avg_inv_196307to201312 = size_op_vw_avg_inv.loc[196307:201312,:]


size_inv_vw_avg_beme = pd.read_csv('25_Portfolios_ME_INV_5x5.csv', index_col = 0, skiprows = 2834, nrows = 670)
size_inv_vw_avg_beme_196307to201312 = size_inv_vw_avg_beme.loc[196307:201312,:]

size_inv_vw_avg_op = pd.read_csv('25_Portfolios_ME_INV_5x5.csv', index_col = 0, skiprows = 4189, nrows = 670)
size_inv_vw_avg_op_196307to201312 = size_inv_vw_avg_op.loc[196307:201312,:]

size_inv_vw_avg_inv = pd.read_csv('25_Portfolios_ME_INV_5x5.csv', index_col = 0, skiprows = 4866, nrows = 670)
size_inv_vw_avg_inv_196307to201312 = size_inv_vw_avg_inv.loc[196307:201312,:]



size_op_inv_32pf_vw_avg_beme = pd.read_csv('32_Portfolios_ME_OP_INV_2x4x4.csv', index_col = 0, skiprows = 2836, nrows = 670)
size_op_inv_32pf_vw_avg_beme_196307to201312 = size_op_inv_32pf_vw_avg_beme.loc[196307:201312,:]

size_op_inv_32pf_vw_avg_op = pd.read_csv('32_Portfolios_ME_OP_INV_2x4x4.csv', index_col = 0, skiprows = 4191, nrows = 670)
size_op_inv_32pf_vw_avg_op_196307to201312 = size_op_inv_32pf_vw_avg_op.loc[196307:201312,:]

size_op_inv_32pf_vw_avg_inv = pd.read_csv('32_Portfolios_ME_OP_INV_2x4x4.csv', index_col = 0, skiprows = 4868, nrows = 670)
size_op_inv_32pf_vw_avg_inv_196307to201312 = size_op_inv_32pf_vw_avg_inv.loc[196307:201312,:]

"""
print(round(size_bm_vw_avg_beme_196307to201312['ME1 BM2'].mean(),2))
print(round(size_bm_vw_avg_op_196307to201312['ME1 BM3'].mean(),2))
print(round(size_bm_vw_avg_inv_196307to201312['ME1 BM4'].mean(),2))

print("\n")

print(round(size_op_vw_avg_beme_196307to201312['ME1 OP2'].mean(),2))
print(round(size_op_vw_avg_op_196307to201312['ME1 OP3'].mean(),2))
print(round(size_op_vw_avg_inv_196307to201312['BIG HiOP'].mean(),2))

print("\n")

print(round(size_bm_vw_avg_beme_196307to201312['ME1 BM2'].mean(),2))
print(round(size_bm_vw_avg_op_196307to201312['ME1 BM3'].mean(),2))
print(round(size_bm_vw_avg_inv_196307to201312['ME1 BM4'].mean(),2))
print("\n")

print(round(size_op_inv_32pf_vw_avg_beme_196307to201312['SMALL LoOP LoINV'].mean(),2))
print(round(size_op_inv_32pf_vw_avg_op_196307to201312['SMALL LoOP LoINV'].mean(),2))
print(round(size_op_inv_32pf_vw_avg_inv_196307to201312['SMALL LoOP LoINV'].mean(),2))

#Change column names in order to get specific value
"""



"""Table 9: Regressions for 25 value-weight Size-OP portfolios:"""

#Panel A: Three-factor intercets for Rm-Rf, SMB, and HML

#Change column name of respective 5x5 portfolio to get the wanted intercepts and t-statistics

excess_return_table9A = pd.DataFrame(me_op_5x5_196307to201312['BIG HiOP'] - ff5_196307to201312['RF'])
endog_table9A = excess_return_table9A[0]
exog_table9A = sm.add_constant(ff5_196307to201312[['Mkt-RF', 'SMB', 'HML']])
reg_table9A = sm.OLS(endog_table9A, exog_table9A)
results_table9A = reg_table9A.fit()
#print(results_table9A.summary())


#Panel B: Five-factor coefficients: Intercepts, HML(h), RMW(r), CMA(c)

#Change column name of respective 5x5 portfolio to get the wanted intercepts, slopes and t-statistics

excess_return_table9B = pd.DataFrame(me_op_5x5_196307to201312['BIG HiOP'] - ff5_196307to201312['RF'])
endog_table9B = excess_return_table9B[0]
exog_table9B = sm.add_constant(ff5_196307to201312[['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA']])
reg_table9B = sm.OLS(endog_table9B, exog_table9B)
results_table9B = reg_table9B.fit()
#print(results_table9B.summary())


"""Table 10: Regressions for 25 value-weight Size-Inv portfolios:"""
#Panel A: Three-factor intercets for Rm-Rf, SMB, and HML

#Change column name of respective 5x5 portfolio to get the wanted intercept and t-statistic

excess_return_table10A = pd.DataFrame(me_inv_5x5_196307to201312['BIG HiINV'] - ff5_196307to201312['RF'])
endog_table10A = excess_return_table10A[0]
exog_table10A = sm.add_constant(ff5_196307to201312[['Mkt-RF', 'SMB', 'HML']])
reg_table10A = sm.OLS(endog_table10A, exog_table10A)
results_table10A = reg_table10A.fit()
#print(results_table10A.summary())



#Panel B: Five-factor coefficients: Intercepts, HML(h), RMW(r), CMA(c)

#Change column name of respective 5x5 portfolio to get the wanted intercepts, slopes and t-statistics

excess_return_table10B = pd.DataFrame(me_inv_5x5_196307to201312['BIG HiINV'] - ff5_196307to201312['RF'])
endog_table10B = excess_return_table10B[0]
exog_table10B = sm.add_constant(ff5_196307to201312[['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA']])
reg_table10B = sm.OLS(endog_table10B, exog_table10B)
results_table10B = reg_table10B.fit()
#print(results_table10B.summary())



"""Table 11: Regressions for 32 value-weight Size-OP-Inv portfolios:"""

#Panel A: Three-factor intercets for Rm-Rf, SMB, and HML

#Change column name of respective 2x4x4 portfolio to get the wanted intercepts and t-statistics

excess_return_table11A = pd.DataFrame(me_op_inv_196307to201312['ME1 OP3 INV4'] - ff5_196307to201312['RF'])
endog_table11A = excess_return_table11A[0]
exog_table11A = sm.add_constant(ff5_196307to201312[['Mkt-RF', 'SMB', 'HML']])
reg_table11A = sm.OLS(endog_table11A, exog_table11A)
results_table11A = reg_table11A.fit()
#print(results_table11A.summary())


#Panel B: Five-factor coefficients: Intercepts, HML(h), RMW(r), CMA(c)

#Change column name of respective 2x4x4 portfolio to get the wanted intercepts, slopes and t-statistics

excess_return_table11B = pd.DataFrame(me_op_inv_196307to201312['ME1 OP3 INV4'] - ff5_196307to201312['RF'])
endog_table11B = excess_return_table11B[0]
exog_table11B = sm.add_constant(ff5_196307to201312[['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA']])
reg_table11B = sm.OLS(endog_table11B, exog_table11B)
results_table11B = reg_table11B.fit()
#print(results_table11B.summary())



"""Augmenting FF 5 Factor Model with Momentum-Factor for 'FF6' and compare average absolute value of intercepts with FF5"""

#In this section I have extended the the Fama French 5 Factor model with the momentum factor.
#I then calculated the average absolute intercepts of the new new 'FF6' model
#and comapared them to the average absolute intercepts of the FF5 model


mom_factor = pd.read_csv('F-F_Momentum_Factor.csv', index_col = 0, skiprows = 13, nrows = 1108)

mom_factor_196307to201312 = mom_factor.loc[196307:201312,:]

ff6 = pd.merge(ff5_196307to201312, mom_factor_196307to201312, left_index = True, right_index = True)

ff6.rename({'Mom   ': 'MOM'}, axis = 1, inplace = True)

#print(ff6.columns.values)

excess_return_ff6 = pd.DataFrame(me_inv_5x5_196307to201312['BIG HiINV'] - ff6['RF'])
endog_ff6 = excess_return_ff6[0]
exog_ff6 = sm.add_constant(ff6[['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA', 'MOM']])
reg_ff6 = sm.OLS(endog_ff6, exog_ff6)
results_ff6 = reg_ff6.fit()
#print(results_ff6.summary())

#print(round(results_ff6.params['const'],2))

intercepts_5x5_me_beme = np.array([-0.25,	0.12,	0,	0.15,	0.15,
-0.07,	-0.02,	0.02,	0.01,	-0.03,
0.05,	0.03,	-0.05,	0.02,	0.04,
0.18,	-0.16,	-0.14,	0.08,	-0.06,
0.12,	-0.09,	-0.1,	-0.22,	0.15])
    
abs_intercepts_5x5_me_beme = np.absolute(intercepts_5x5_me_beme)
abs_avg_intercepts_5x5_me_beme = round(np.average(abs_intercepts_5x5_me_beme),3)

#print(abs_avg_intercepts_5x5_me_beme)


intercepts_5x5_me_op = np.array([-0.11,	0.04,	-0.06,	0.04,	-0.14,
-0.06,	-0.08,	-0.05,	-0.05,	0.04,
0.14,	-0.01,	-0.04,	-0.06,	0.07,
0.2,	0.04,	-0.1,	-0.05,	0.01,
0.14,	-0.09,	0.07,	0.02,	0.05])
    
abs_intercepts_5x5_me_op = np.absolute(intercepts_5x5_me_op)
abs_avg_intercepts_5x5_me_op = round(np.average(abs_intercepts_5x5_me_op),3)

#print(abs_avg_intercepts_5x5_me_op)


intercepts_5x5_me_inv = np.array([0.21,	0.12,	0.09,	0.02,	-0.33,
0.02,	0,	0.07,	0.05,	-0.13,
0.07,	0.1,	0.01,	0.06,	0,
-0.04,	-0.03,	0,	0.04,	0.13,
0.02,	-0.06,	-0.06,	0.02,	-0.33])
    
abs_intercepts_5x5_me_inv = np.absolute(intercepts_5x5_me_inv)
abs_avg_intercepts_5x5_me_inv = round(np.average(abs_intercepts_5x5_me_inv),3)

#print(abs_avg_intercepts_5x5_me_inv)


intercepts_32pf_size_beme_op = np.array([-0.36,	-0.01,	-0.11,	-0.08,	0.28,	0.13,	0.08,	0.06,
0.02,	-0.06,	-0.06,	0.09,	-0.06,	-0.16,	-0.06,	-0.15,
-0.05,	-0.07,	0.11,	0.21,	-0.11,	-0.19,	0,	-0.02,
-0.08,	0.08,	0.24,	0.38,	-0.13,	-0.14,	0.08,	-0.27])
    
abs_intercepts_32pf_size_beme_op = np.absolute(intercepts_32pf_size_beme_op)
abs_avg_intercepts_32pf_size_beme_op = round(np.average(abs_intercepts_32pf_size_beme_op),3)

#print(abs_avg_intercepts_32pf_size_beme_op)


intercepts_32pf_size_beme_inv = np.array([-0.07,	0.07,	0.1,	-0.17,	-0.08,	0.01,	0.08,	0.35,
0.12,	0.02,	0,	-0.07,	-0.04,	-0.01,	-0.11,	-0.11,
0.18,	-0.05,	0.13,	-0.08,	-0.14,	-0.06,	-0.12,	-0.21,
0.09,	0.11,	-0.07,	-0.04,	-0.08,	-0.18,	0,	-0.08])
    
abs_intercepts_32pf_size_beme_inv = np.absolute(intercepts_32pf_size_beme_inv)
abs_avg_intercepts_32pf_size_beme_inv = round(np.average(abs_intercepts_32pf_size_beme_inv),3)

#print(abs_avg_intercepts_32pf_size_beme_inv)


intercepts_32pf_size_op_inv = np.array([0.05,	0.12,	-0.14,	-0.46,	0.12,	-0.15,	0.07,	0.15,
0,	0,	0.12,	-0.18,	-0.07,	-0.05,	0,	-0.13,
0.15,	0.03,	0.06,	-0.08,	-0.04,	0.04,	-0.07,	0.08,
0.13,	0.03,	0.15,	-0.09,	-0.06,	0.01,	0.01,	0.28])
    
abs_intercepts_32pf_size_op_inv = np.absolute(intercepts_32pf_size_op_inv)
abs_avg_intercepts_32pf_size_op_inv = round(np.average(abs_intercepts_32pf_size_op_inv),3)

#print(abs_avg_intercepts_32pf_size_op_inv)

"""
The average absolute intercepts of the FF5 Model by Fama/French augmented with
the momentum factor are slightly lower than the average absolute intercepts of 
the FF5 Model for almost all portfolios but the 32 Size-B/M-INV portfolios.
See results file for overview of the values.
"""
