import pandas as pd
from scipy.stats import shapiro, levene, ttest_ind,mannwhitneyu
import os

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)

def ab_test(dataframe,col,target):
    test_stat, pvalue1 = shapiro(dataframe.loc[(dataframe["Data_Type"]=="test") , target])
    test_stat, pvalue2 = shapiro(dataframe[~dataframe[col].str.contains(dataframe[dataframe["Data_Type"] == "test"][col].unique()[0])][dataframe["Data_Type"] == "train"][target])
    if pvalue1 > 0.05 and pvalue2 > 0.05:
        test_stat, pvalue_levene = levene(dataframe.loc[(dataframe["Data_Type"]=="test"), target],
                                   dataframe[~dataframe[col].str.contains(dataframe[dataframe["Data_Type"] == "test"][col].unique()[0])][dataframe["Data_Type"] == "train"][target])
        if pvalue_levene > 0.05:
            test_stat, pvalue_ttest_ind = ttest_ind(dataframe.loc[(dataframe["Data_Type"]=="test") , target],
                                          dataframe[~dataframe[col].str.contains(dataframe[dataframe["Data_Type"] == "test"][col].unique()[0])][dataframe["Data_Type"] == "train"][target],
                                          equal_var=True)
            if pvalue_ttest_ind > 0.05:
                return "According to the parametric test results, there is no statistically significant difference between the two groups."
        else:
            test_stat, pvalue_ttest_ind_2 = ttest_ind(dataframe.loc[(dataframe["Data_Type"]=="test") , target],
                                                    dataframe[~dataframe[col].str.contains(dataframe[dataframe["Data_Type"] == "test"][col].unique()[0])][dataframe["Data_Type"] == "train"][target],
                                                    equal_var=False)
            if pvalue_ttest_ind_2 < 0.05:
                return "According to the weltch test results, there is no statistically significant difference between the two groups."
            else:
                return "There is a statistically significant difference between the two groups according to the weltch test result."

    elif pvalue1 < 0.05 or pvalue2 < 0.05:
        test_stat, pvalue_man = mannwhitneyu(dataframe.loc[(dataframe["Data_Type"]=="test") , target],
                                         dataframe[~dataframe[col].str.contains(dataframe[dataframe["Data_Type"] == "test"][col].unique()[0])][dataframe["Data_Type"] == "train"][target])
        if pvalue_man < 0.05:
            return "According to the mannwhitneyu test result, there is no statistically significant difference between the two groups."
        else:
            return "According to the mannwhitneyu test result, there is a statistically significant difference between the two groups."



