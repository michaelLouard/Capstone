"""Data Exploration Assignment 1
Michael Louard
ECON 395
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
from scipy import stats
from scipy.stats import chi2
import seaborn as sn
from sklearn.covariance import MinCovDet
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.formula.api import ols
import researchpy as rp
from stargazer.stargazer import Stargazer
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
from IPython.core.display import HTML

# Read in data
original_dataset = pd.read_csv("complete_data.csv")
original_dataset.info()

dataset = original_dataset.copy()
# Create dataframe
data = pd.DataFrame(dataset)

data.info()
data = data.drop(columns=['fips_state_code', 'fips_county_code'], axis=1)
data.info()

# Rename Columns
data = data.rename(columns={"year": "Year", "state": "State", "agency_name": "Agency Name",
                            "fips_state_county_code": "FIPS Code", "population": "Population",
                            "rec_legal": "Recreationally Legal", "med_legal": "Medically Legal",
                            "narc_per_cap": "Narcotic Arrests per capita",
                            "synth_per_cap": "Synthetic Arrests per capita",
                            "other_per_cap": "Other Drug Arrests per capita",
                            "total_drug_per_cap": "Total Drug Arrests per capita",
                            "violent_per_cap": "Violent Crime Arrests per capita",
                            "prop_per_cap": "Property Crime Arrests per capita", "cpi": "CPI",
                            "hhic": "Median Household Income", "hhic_inf": "Inf-Adj Median Household Income",
                            "educ": "Education", "unemp": "Unemployment", "white": "White", "black": "Black",
                            "hisp": "Hispanic", "dem_gov": "Democratic Governor",
                            "officer_emp": "Officer Employees", "officer_per_cap": "Officers Per Capita",
                            "state_agency": "State Agency"})
data.info()

no_names = data.drop(["Year", "State", "Agency Name", "FIPS Code", "State Agency",
                      "CPI", "Median Household Income"], axis=1)
drop_for_sum = data.drop(["State", "Agency Name", "FIPS Code", "State Agency",
                          "CPI", "Median Household Income", "Officer Employees"], axis=1)
drop_for_sum.to_csv('drop_for_sum.csv')
no_names.info()
descriptors = ['mean', 'std', 'median', 'min', 'max', 'count', 'percentile']

summary_stats = no_names.agg({"Recreationally Legal": descriptors,
                              "Medically Legal": descriptors,
                              "Narcotic Arrests per capita": descriptors,
                              "Synthetic Arrests per capita": descriptors,
                              "Other Drug Arrests per capita": descriptors,
                              "Total Drug Arrests per capita": descriptors,
                              "Violent Crime Arrests per capita": descriptors,
                              "Property Crime Arrests per capita": descriptors,
                              "Inf-Adj Median Household Income": descriptors,
                              "Education": descriptors,
                              "Unemployment": descriptors,
                              "White": descriptors, "Black": descriptors, "Hispanic": descriptors,
                              "Democratic Governor": descriptors,
                              "Officers Per Capita": descriptors,
                              })
summary_stats.to_csv('summary_stats1.csv')
summary_stats = no_names.describe()
summary_stats = summary_stats.transpose()
summary_stats.to_csv('summary_stats2.csv')


# Manually added frequency into summary_stats.csv
freq = data.groupby('Year')['Year'].count()

dep_and_controls = no_names[['Narcotic Arrests per capita', 'Synthetic Arrests per capita',
                             'Other Drug Arrests per capita', 'Violent Crime Arrests per capita',
                             'Property Crime Arrests per capita', 'Inf-Adj Median Household Income',
                             'Education', 'Unemployment', 'White', 'Black', 'Hispanic', 'Democratic Governor',
                             'Officer Employees']]

controls = no_names[['Violent Crime Arrests per capita', 'Property Crime Arrests per capita',
                     'Inf-Adj Median Household Income', 'Education', 'Unemployment',
                     'White', 'Black', 'Hispanic', 'Democratic Governor', 'Officers Per Capita']]

corr_matrix = no_names.corr()
sn.set(rc={'figure.figsize': (12, 12)})
sn.heatmap(corr_matrix, annot=True, linewidths=0.1, annot_kws={"size": 8})
plt.subplots_adjust(left=0.25, bottom=0.25)
plt.savefig('main_corr.png')
plt.show()
small_corr_1 = dep_and_controls.corr()
sn.set(rc={'figure.figsize': (12, 12)})
sn.heatmap(small_corr_1, annot=True, linewidths=0.1, annot_kws={"size": 8})
plt.subplots_adjust(left=0.25, bottom=0.25)
plt.savefig('dep_and_controls_corr.png')
plt.show()
small_corr_2 = controls.corr()
sn.set(rc={'figure.figsize': (12, 12)})
sn.heatmap(small_corr_2, annot=True, linewidths=0.1, annot_kws={"size": 8}, cmap='coolwarm')
plt.subplots_adjust(left=0.25, bottom=0.25)
plt.title("Figure 1: Correlation Matrix of Control Variables")
plt.savefig('controls_corr.png')
plt.show()


# Exporting CSVs
corr_matrix.to_csv('corr_matrix.csv')
small_corr_1.to_csv('dep_and_controls.csv')
small_corr_2.to_csv('controls.csv')

#t tests
data["Year"].value_counts()
data.info()

mask1 = data["Year"] >= 2008
mask2 = data["Year"] <= 2012
bucket1 = data[mask1 & mask2]

mask1 = data["Year"] >= 2013
mask2 = data["Year"] <= 2019
bucket2 = data[mask1 & mask2]
# ttest for buckets 2008-2012 and 2013-2019
summary1, result1 = rp.ttest(group1=bucket1['Total Drug Arrests per capita'],
                             group1_name='Total Drug Arrests per capita (2008-2012)',
                             group2=bucket2['Total Drug Arrests per capita'],
                             group2_name='Total Drug Arrests per capita (2013-2019)')
print(summary1)
print(result1)

# t test for total crime between rec legal and illegal states
# import rec and illegal data
rec_data = pd.read_csv('rec_data.csv')
illegal_data = pd.read_csv('illegal_data.csv')
# adding rec total crimes
rec_data['Total Crimes per capita'] = rec_data['total_drug_per_cap'] + \
                                     rec_data['violent_per_cap'] + \
                                     rec_data['prop_per_cap']
# adding illegal total crimes
illegal_data['Total Crimes per capita'] = illegal_data['total_drug_per_cap'] + \
                                     illegal_data['violent_per_cap'] + \
                                     illegal_data['prop_per_cap']

summary2, result2 = rp.ttest(group1=rec_data['Total Crimes per capita'],
                             group1_name='Recreationally Legal States Total Crimes per capita',
                             group2=illegal_data['Total Crimes per capita'],
                             group2_name='Illegal States Total Crimes per capita')
print(summary2)
print(result2)

summary1.to_csv('summary_for_total_drug_per_cap.csv')
summary2.to_csv('summary_for_rec_vs_illegal_crime.csv')
result1.to_csv('ttest_for_total_drug_per_cap.csv')
result2.to_csv('tttest_for_rec_vs_illegal_crime.csv')

def tukeys_method(no_names, variable):

    # 2 parameters: dataframe and variable of interest as string
    q1 = no_names[variable].quantile(0.25)  # first quartile
    q3 = no_names[variable].quantile(0.75)  # third quartile
    iqr = q3 - q1  # inter-quartile range
    inner_fence = 1.5 * iqr
    outer_fence = 3 * iqr

    # inner fence lower and upper end
    inner_fence_le = q1 - inner_fence
    inner_fence_ue = q3 + inner_fence

    print("Inner fence for " + str(variable) + ": " + str(inner_fence_le) +
          " - " + str(inner_fence_ue))

    # outer fence lower and upper end
    outer_fence_le = q1 - outer_fence
    outer_fence_ue = q3 + outer_fence

    print("Outer fence for " + str(variable) + ": " + str(outer_fence_le) +
          " - " + str(outer_fence_ue))

    outliers_prob = []
    outliers_poss = []
    for index, x in enumerate(no_names[variable]):
        if x < outer_fence_le or x > outer_fence_ue:
            outliers_prob.append(index)
    for index, x in enumerate(no_names[variable]):
        if x < inner_fence_le or x > inner_fence_ue:
            outliers_poss.append(index)
    return outliers_prob, outliers_poss


probable_outliers_tm, possible_outliers_tm = tukeys_method(no_names, "Population")
print("Probable Outliers for Population: ", probable_outliers_tm)
print("Possible Outliers for Population: ", possible_outliers_tm)
print("\n ------ \n")

probable_outliers_tm, possible_outliers_tm = tukeys_method(no_names, "Narcotic Arrests per capita")
print("Probable Outliers for Narcotic Arrests per capita: ", probable_outliers_tm)
print("Possible Outliers for Narcotic Arrests per capita: ", possible_outliers_tm)
print("\n ------ \n")

probable_outliers_tm, possible_outliers_tm = tukeys_method(no_names, "Synthetic Arrests per capita")
print("Probable Outliers for Synthetic Arrests per capita: ", probable_outliers_tm)
print("Possible Outliers for Synthetic Arrests per capita: ", possible_outliers_tm)
print("\n ------ \n")

probable_outliers_tm, possible_outliers_tm = tukeys_method(no_names, "Other Drug Arrests per capita")
print("Probable Outliers for Other Drug Arrests per capita: ", probable_outliers_tm)
print("Possible Outliers for Other Drug Arrests per capita: ", possible_outliers_tm)
print("\n ------ \n")

probable_outliers_tm, possible_outliers_tm = tukeys_method(no_names, "Violent Crime Arrests per capita")
print("Probable Outliers for Violent Crime Arrests per capita: ", probable_outliers_tm)
print("Possible Outliers for Violent Crime Arrests per capita: ", possible_outliers_tm)
print("\n ------ \n")

probable_outliers_tm, possible_outliers_tm = tukeys_method(no_names, "Property Crime Arrests per capita")
print("Probable Outliers for Property Crime Arrests per capita: ", probable_outliers_tm)
print("Possible Outliers for Property Crime Arrests per capita: ", possible_outliers_tm)
print("\n ------ \n")

probable_outliers_tm, possible_outliers_tm = tukeys_method(no_names, "Inf-Adj Median Household Income")
print("Probable Outliers for Inflation-Adjusted Household Income: ", probable_outliers_tm)
print("Possible Outliers for Inflation-Adjusted Household Income: ", possible_outliers_tm)
print("\n ------ \n")

probable_outliers_tm, possible_outliers_tm = tukeys_method(no_names, "Education")
print("Probable Outliers for Education: ", probable_outliers_tm)
print("Possible Outliers for Education: ", possible_outliers_tm)
print("\n ------ \n")

probable_outliers_tm, possible_outliers_tm = tukeys_method(no_names, "Unemployment")
print("Probable Outliers for Unemployment: ", probable_outliers_tm)
print("Possible Outliers for Unemployment: ", possible_outliers_tm)
print("\n ------ \n")

# Multivariate Outliers
# First Set of Scatters (x = Narcotic Arrests per capita y = all other relevant variables)
fig1 = plt.figure(figsize=(16, 16))
ax1 = fig1.add_subplot(6, 2, 1)
ax2 = fig1.add_subplot(6, 2, 2)
ax3 = fig1.add_subplot(6, 2, 3)
ax4 = fig1.add_subplot(6, 2, 4)
ax5 = fig1.add_subplot(6, 2, 5)
ax6 = fig1.add_subplot(6, 2, 6)
ax7 = fig1.add_subplot(6, 2, 7)
ax8 = fig1.add_subplot(6, 2, 8)
ax9 = fig1.add_subplot(6, 2, 9)
ax10 = fig1.add_subplot(6, 2, 10)
ax11 = fig1.add_subplot(6, 2, 11)
ax12 = fig1.add_subplot(6, 2, 12)

ax1.scatter('Narcotic Arrests per capita', 'Recreationally Legal', data=no_names, color='Red')
ax1.set_xlabel("Narcotic Arrests per capita")
ax1.set_ylabel("Recreationally Legal")

# Scatter plot for Narcotic Arrests per capita and Medically Legal
ax2.scatter('Narcotic Arrests per capita', 'Medically Legal', data=no_names, color='Red')
ax2.set_xlabel("Narcotic Arrests per capita")
ax2.set_ylabel("Medically Legal")

ax3.scatter('Narcotic Arrests per capita', 'Officer Employees', data=no_names, color='Red')
ax3.set_xlabel("Narcotic Arrests per capita")
ax3.set_ylabel("Officer Employees")

ax4.scatter('Narcotic Arrests per capita', 'Violent Crime Arrests per capita', data=no_names, color='Red')
ax4.set_xlabel("Narcotic Arrests per capita")
ax4.set_ylabel("Violent Crime Arrests per capita")

ax5.scatter('Narcotic Arrests per capita', 'Property Crime Arrests per capita', data=no_names, color='Red')
ax5.set_xlabel("Narcotic Arrests per capita")
ax5.set_ylabel("Property Crime Arrests per capita")

ax6.scatter('Narcotic Arrests per capita', 'Inf-Adj Median Household Income', data=no_names, color='Red')
ax6.set_xlabel("Narcotic Arrests per capita")
ax6.set_ylabel("Inflation-Adjusted Median Household Income")

ax7.scatter('Narcotic Arrests per capita', 'Education', data=no_names, color='Red')
ax7.set_xlabel("Narcotic Arrests per capita")
ax7.set_ylabel("Education")

ax8.scatter('Narcotic Arrests per capita', 'Unemployment', data=no_names, color='Red')
ax8.set_xlabel("Narcotic Arrests per capita")
ax8.set_ylabel("Unemployment")

ax9.scatter('Narcotic Arrests per capita', 'White', data=no_names, color='Red')
ax9.set_xlabel("Narcotic Arrests per capita")
ax9.set_ylabel("White")

ax10.scatter('Narcotic Arrests per capita', 'Black', data=no_names, color='Red')
ax10.set_xlabel("Narcotic Arrests per capita")
ax10.set_ylabel("Black")

ax11.scatter('Narcotic Arrests per capita', 'Hispanic', data=no_names, color='Red')
ax11.set_xlabel("Narcotic Arrests per capita")
ax11.set_ylabel("Hispanic")

ax12.scatter('Narcotic Arrests per capita', 'Democratic Governor', data=no_names, color='Red')
ax12.set_xlabel("Narcotic Arrests per capita")
ax12.set_ylabel("Democratic Governor")
fig1.tight_layout()
plt.show()

# Second set of Scatter Plots (x = Synthetic Arrests per capita, y = all other relevant variables)
fig2 = plt.figure(figsize=(16, 16))
ax1 = fig2.add_subplot(6, 2, 1)
ax2 = fig2.add_subplot(6, 2, 2)
ax3 = fig2.add_subplot(6, 2, 3)
ax4 = fig2.add_subplot(6, 2, 4)
ax5 = fig2.add_subplot(6, 2, 5)
ax6 = fig2.add_subplot(6, 2, 6)
ax7 = fig2.add_subplot(6, 2, 7)
ax8 = fig2.add_subplot(6, 2, 8)
ax9 = fig2.add_subplot(6, 2, 9)
ax10 = fig2.add_subplot(6, 2, 10)
ax11 = fig2.add_subplot(6, 2, 11)
ax12 = fig2.add_subplot(6, 2, 12)

ax1.scatter('Synthetic Arrests per capita', 'Recreationally Legal', data=no_names, color='Blue')
ax1.set_xlabel("Synthetic Arrests per capita")
ax1.set_ylabel("Recreationally Legal")

# Synthetic Arrests per capita and Medically Legal
ax2.scatter('Synthetic Arrests per capita', 'Medically Legal', data=no_names, color='Blue')
ax2.set_xlabel("Synthetic Arrests per capita")
ax2.set_ylabel("Medically Legal")

ax3.scatter('Synthetic Arrests per capita', 'Officer Employees', data=no_names, color='Blue')
ax3.set_xlabel("Synthetic Arrests per capita")
ax3.set_ylabel("Officer Employees")

ax4.scatter('Synthetic Arrests per capita', 'Violent Crime Arrests per capita', data=no_names, color='Blue')
ax4.set_xlabel("Synthetic Arrests per capita")
ax4.set_ylabel("Violent Crime Arrests per capita")

ax5.scatter('Synthetic Arrests per capita', 'Property Crime Arrests per capita', data=no_names, color='Blue')
ax5.set_xlabel("Synthetic Arrests per capita")
ax5.set_ylabel("Property Crime Arrests per capita")

ax6.scatter('Synthetic Arrests per capita', 'Inf-Adj Median Household Income', data=no_names, color='Blue')
ax6.set_xlabel("Synthetic Arrests per capita")
ax6.set_ylabel("Inflation-Adjusted Median Household Income")

ax7.scatter('Synthetic Arrests per capita', 'Education', data=no_names, color='Blue')
ax7.set_xlabel("Synthetic Arrests per capita")
ax7.set_ylabel("Education")

ax8.scatter('Synthetic Arrests per capita', 'Unemployment', data=no_names, color='Blue')
ax8.set_xlabel("Synthetic Arrests per capita")
ax8.set_ylabel("Unemployment")

ax9.scatter('Synthetic Arrests per capita', 'White', data=no_names, color='Blue')
ax9.set_xlabel("Synthetic Arrests per capita")
ax9.set_ylabel("White")

ax10.scatter('Synthetic Arrests per capita', 'Black', data=no_names, color='Blue')
ax10.set_xlabel("Synthetic Arrests per capita")
ax10.set_ylabel("Black")

ax11.scatter('Synthetic Arrests per capita', 'Hispanic', data=no_names, color='Blue')
ax11.set_xlabel("Synthetic Arrests per capita")
ax11.set_ylabel("Hispanic")

ax12.scatter('Synthetic Arrests per capita', 'Democratic Governor', data=no_names, color='Blue')
ax12.set_xlabel("Synthetic Arrests per capita")
ax12.set_ylabel("Democratic Governor")
fig2.tight_layout()
plt.show()

# Third set of Scatter Plots (x = Other Drug Arrests per capita, y = all other relevant variables)
fig3 = plt.figure(figsize=(16, 16))
ax1 = fig3.add_subplot(6, 2, 1)
ax2 = fig3.add_subplot(6, 2, 2)
ax3 = fig3.add_subplot(6, 2, 3)
ax4 = fig3.add_subplot(6, 2, 4)
ax5 = fig3.add_subplot(6, 2, 5)
ax6 = fig3.add_subplot(6, 2, 6)
ax7 = fig3.add_subplot(6, 2, 7)
ax8 = fig3.add_subplot(6, 2, 8)
ax9 = fig3.add_subplot(6, 2, 9)
ax10 = fig3.add_subplot(6, 2, 10)
ax11 = fig3.add_subplot(6, 2, 11)
ax12 = fig3.add_subplot(6, 2, 12)

ax1.scatter('Other Drug Arrests per capita', 'Recreationally Legal', data=no_names, color='Green')
ax1.set_xlabel("Other Drug Arrests per capita")
ax1.set_ylabel("Recreationally Legal")

ax2.scatter('Other Drug Arrests per capita', 'Medically Legal', data=no_names, color='Green')
ax2.set_xlabel("Other Drug Arrests per capita")
ax2.set_ylabel("Medically Legal")

ax3.scatter('Other Drug Arrests per capita', 'Officer Employees', data=no_names, color='Green')
ax3.set_xlabel("Other Drug Arrests per capita")
ax3.set_ylabel("Officer Employees")

ax4.scatter('Other Drug Arrests per capita', 'Violent Crime Arrests per capita', data=no_names, color='Green')
ax4.set_xlabel("Other Drug Arrests per capita")
ax4.set_ylabel("Violent Crime Arrests per capita")

ax5.scatter('Other Drug Arrests per capita', 'Property Crime Arrests per capita', data=no_names, color='Green')
ax5.set_xlabel("Other Drug Arrests per capita")
ax5.set_ylabel("Property Crime Arrests per capita")

ax6.scatter('Other Drug Arrests per capita', 'Inf-Adj Median Household Income', data=no_names, color='Green')
ax6.set_xlabel("Other Drug Arrests per capita")
ax6.set_ylabel("Inflation-Adjusted Median Household Income")

ax7.scatter('Other Drug Arrests per capita', 'Education', data=no_names, color='Green')
ax7.set_xlabel("Other Drug Arrests per capita")
ax7.set_ylabel("Education")

ax8.scatter('Other Drug Arrests per capita', 'Unemployment', data=no_names, color='Green')
ax8.set_xlabel("Other Drug Arrests per capita")
ax8.set_ylabel("Unemployment")

ax9.scatter('Other Drug Arrests per capita', 'White', data=no_names, color='Green')
ax9.set_xlabel("Other Drug Arrests per capita")
ax9.set_ylabel("White")

ax10.scatter('Other Drug Arrests per capita', 'Black', data=no_names, color='Green')
ax10.set_xlabel("Other Drug Arrests per capita")
ax10.set_ylabel("Black")

ax11.scatter('Other Drug Arrests per capita', 'Hispanic', data=no_names, color='Green')
ax11.set_xlabel("Other Drug Arrests per capita")
ax11.set_ylabel("Hispanic")

ax12.scatter('Other Drug Arrests per capita', 'Democratic Governor', data=no_names, color='Green')
ax12.set_xlabel("Other Drug Arrests per capita")
ax12.set_ylabel("Democratic Governor")
fig3.tight_layout()
plt.show()

# Fourth Scatter Plot Controls Against Each Other: Officer Employees against all other relevant variables
fig4 = plt.figure(figsize=(16, 16))
ax1 = fig4.add_subplot(5, 2, 1)
ax2 = fig4.add_subplot(5, 2, 2)
ax3 = fig4.add_subplot(5, 2, 3)
ax4 = fig4.add_subplot(5, 2, 4)
ax5 = fig4.add_subplot(5, 2, 5)
ax6 = fig4.add_subplot(5, 2, 6)
ax7 = fig4.add_subplot(5, 2, 7)
ax8 = fig4.add_subplot(5, 2, 8)
ax9 = fig4.add_subplot(5, 2, 9)
ax10 = fig4.add_subplot(5, 2, 10)

ax1.scatter('Officer Employees', 'Violent Crime Arrests per capita',  data=no_names, color='Red')
ax1.set_xlabel("Officer Employees")
ax1.set_ylabel("Violent Crime Arrests per capita")

ax2.scatter('Officer Employees', 'Property Crime Arrests per capita', data=no_names, color='Red')
ax2.set_xlabel("Officer Employees")
ax2.set_ylabel("Property Crime Arrests per capita")

ax3.scatter('Officer Employees', 'Population', data=no_names, color='Red')
ax3.set_xlabel("Officer Employees")
ax3.set_ylabel("Population")

ax4.scatter('Officer Employees', 'Inf-Adj Median Household Income', data=no_names, color='Red')
ax4.set_xlabel("Officer Employees")
ax4.set_ylabel("Inflation-Adjusted Median Household Income")

ax5.scatter('Officer Employees', 'Education', data=no_names, color='Red')
ax5.set_xlabel("Officer Employees")
ax5.set_ylabel("Education")

ax6.scatter('Officer Employees', 'Unemployment', data=no_names, color='Red')
ax6.set_xlabel("Officer Employees")
ax6.set_ylabel("Unemployment")

ax7.scatter('Officer Employees', 'White', data=no_names, color='Red')
ax7.set_xlabel("Officer Employees")
ax7.set_ylabel("White")

ax8.scatter('Officer Employees', 'Black', data=no_names, color='Red')
ax8.set_xlabel("Officer Employees")
ax8.set_ylabel("Black")

ax9.scatter('Officer Employees', 'Hispanic', data=no_names, color='Red')
ax9.set_xlabel("Officer Employees")
ax9.set_ylabel("Hispanic")

ax10.scatter('Officer Employees', 'Democratic Governor', data=no_names, color='Red')
ax10.set_xlabel("Officer Employees")
ax10.set_ylabel("Democratic Governor")
fig4.tight_layout()
plt.show()

# Fifth Scatter Plot: Violent Crime Arrests per capita against all other relevant variables
fig5 = plt.figure(figsize=(16, 16))
ax1 = fig5.add_subplot(5, 2, 1)
ax2 = fig5.add_subplot(5, 2, 2)
ax3 = fig5.add_subplot(5, 2, 3)
ax4 = fig5.add_subplot(5, 2, 4)
ax5 = fig5.add_subplot(5, 2, 5)
ax6 = fig5.add_subplot(5, 2, 6)
ax7 = fig5.add_subplot(5, 2, 7)
ax8 = fig5.add_subplot(5, 2, 8)
ax9 = fig5.add_subplot(5, 2, 9)


ax1.scatter('Violent Crime Arrests per capita', 'Property Crime Arrests per capita', data=no_names, color='Blue')
ax1.set_xlabel("Violent Crime Arrests per capita")
ax1.set_ylabel("Property Crime Arrests per capita")

ax2.scatter('Violent Crime Arrests per capita', 'Population', data=no_names, color='Blue')
ax2.set_xlabel("Violent Crime Arrests per capita")
ax2.set_ylabel("Population")

ax3.scatter('Violent Crime Arrests per capita', 'Inf-Adj Median Household Income', data=no_names, color='Blue')
ax3.set_xlabel("Violent Crime Arrests per capita")
ax3.set_ylabel("Inflation-Adjusted Median Household Income")

ax4.scatter('Violent Crime Arrests per capita', 'Education', data=no_names, color='Blue')
ax4.set_xlabel("Violent Crime Arrests per capita")
ax4.set_ylabel("Education")

ax5.scatter('Violent Crime Arrests per capita', 'Unemployment', data=no_names, color='Blue')
ax5.set_xlabel("Violent Crime Arrests per capita")
ax5.set_ylabel("Unemployment")

ax6.scatter('Violent Crime Arrests per capita', 'White', data=no_names, color='Blue')
ax6.set_xlabel("Officer Employees")
ax6.set_ylabel("White")

ax7.scatter('Violent Crime Arrests per capita', 'Black', data=no_names, color='Blue')
ax7.set_xlabel("Violent Crime Arrests per capita")
ax7.set_ylabel("Black")

ax8.scatter('Violent Crime Arrests per capita', 'Hispanic', data=no_names, color='Blue')
ax8.set_xlabel("Violent Crime Arrests per capita")
ax8.set_ylabel("Hispanic")

ax9.scatter('Violent Crime Arrests per capita', 'Democratic Governor', data=no_names, color='Blue')
ax9.set_xlabel("Violent Crime Arrests per capita")
ax9.set_ylabel("Democratic Governor")
plt.show()

# Sixth Scatter Plot: Democratic Governor vs all other relevant variables
fig6 = plt.figure(figsize=(16, 16))
ax1 = fig6.add_subplot(4, 2, 1)
ax2 = fig6.add_subplot(4, 2, 2)
ax3 = fig6.add_subplot(4, 2, 3)
ax4 = fig6.add_subplot(4, 2, 4)
ax5 = fig6.add_subplot(4, 2, 5)
ax6 = fig6.add_subplot(4, 2, 6)
ax7 = fig6.add_subplot(4, 2, 7)
ax8 = fig6.add_subplot(4, 2, 8)

ax1.scatter('Democratic Governor', 'Property Crime Arrests per capita', data=no_names, color='Green')
ax1.set_xlabel("Democratic Governor")
ax1.set_ylabel("Property Crime Arrests per capita")

ax2.scatter('Democratic Governor', 'Population', data=no_names, color='Green')
ax2.set_xlabel("Democratic Governor")
ax2.set_ylabel("Population")

ax3.scatter('Democratic Governor', 'Inf-Adj Median Household Income', data=no_names, color='Green')
ax3.set_xlabel("Democratic Governor")
ax3.set_ylabel("Inflation-Adjusted Median Household Income")

ax4.scatter('Democratic Governor', 'Education', data=no_names, color='Green')
ax4.set_xlabel("Democratic Governor")
ax4.set_ylabel("Education")

ax5.scatter('Democratic Governor', 'Unemployment', data=no_names, color='Green')
ax5.set_xlabel("Democratic Governor")
ax5.set_ylabel("Unemployment")

ax6.scatter('Democratic Governor', 'White', data=no_names, color='Green')
ax6.set_xlabel("Democratic Governor")
ax6.set_ylabel("White")

ax7.scatter('Democratic Governor', 'Black', data=no_names, color='Green')
ax7.set_xlabel("Democratic Governor")
ax7.set_ylabel("Black")

ax8.scatter('Democratic Governor', 'Hispanic', data=no_names, color='Green')
ax8.set_xlabel("Democratic Governor")
ax8.set_ylabel("Hispanic")
plt.show()

# Scatter Plot 7: Hispanic vs all other variables (excluding other racial demographics)
fig7 = plt.figure(figsize=(16, 16))
ax1 = fig7.add_subplot(3, 2, 1)
ax2 = fig7.add_subplot(3, 2, 2)
ax3 = fig7.add_subplot(3, 2, 3)
ax4 = fig7.add_subplot(3, 2, 4)
ax5 = fig7.add_subplot(3, 2, 5)

ax1.scatter('Hispanic', 'Property Crime Arrests per capita', data=no_names, color='darkred')
ax1.set_xlabel("Hispanic")
ax1.set_ylabel("Property Crime Arrests per capita")

ax2.scatter('Hispanic', 'Population', data=no_names, color='darkred')
ax2.set_xlabel("Hispanic")
ax2.set_ylabel("Population")

ax3.scatter('Hispanic', 'Inf-Adj Median Household Income', data=no_names, color='darkred')
ax3.set_xlabel("Hispanic")
ax3.set_ylabel("Inflation-Adjusted Median Household Income")

ax4.scatter('Hispanic', 'Education', data=no_names, color='darkred')
ax4.set_xlabel("Hispanic")
ax4.set_ylabel("Education")

ax5.scatter('Hispanic', 'Unemployment', data=no_names, color='darkred')
ax5.set_xlabel("Hispanic")
ax5.set_ylabel("Unemployment")
plt.show()

# Scatter plot 8: White vs all other relevant variables (excluding other racial demographics)
fig8 = plt.figure(figsize=(16, 16))
ax1 = fig8.add_subplot(3, 2, 1)
ax2 = fig8.add_subplot(3, 2, 2)
ax3 = fig8.add_subplot(3, 2, 3)
ax4 = fig8.add_subplot(3, 2, 4)
ax5 = fig8.add_subplot(3, 2, 5)

ax1.scatter('White', 'Property Crime Arrests per capita', data=no_names, color='Navy')
ax1.set_xlabel("White")
ax1.set_ylabel("Property Crime Arrests per capita")

ax2.scatter('White', 'Population', data=no_names, color='Navy')
ax2.set_xlabel("White")
ax2.set_ylabel("Population")

ax3.scatter('White', 'Inf-Adj Median Household Income', data=no_names, color='Navy')
ax3.set_xlabel("White")
ax3.set_ylabel("Inflation-Adjusted Median Household Income")

ax4.scatter('White', 'Education', data=no_names, color='Navy')
ax4.set_xlabel("White")
ax4.set_ylabel("Education")

ax5.scatter('White', 'Unemployment', data=no_names, color='Navy')
ax5.set_xlabel("White")
ax5.set_ylabel("Unemployment")
plt.show()

# Scatter plot 9: Black vs all other relevant variables (excluding other racial demographics)
fig9 = plt.figure(figsize=(16, 16))
ax1 = fig9.add_subplot(3, 2, 1)
ax2 = fig9.add_subplot(3, 2, 2)
ax3 = fig9.add_subplot(3, 2, 3)
ax4 = fig9.add_subplot(3, 2, 4)
ax5 = fig9.add_subplot(3, 2, 5)

ax1.scatter('Black', 'Property Crime Arrests per capita', data=no_names, color='darkgreen')
ax1.set_xlabel("Black")
ax1.set_ylabel("Property Crime Arrests per capita")

ax2.scatter('Black', 'Population', data=no_names, color='darkgreen')
ax2.set_xlabel("Black")
ax2.set_ylabel("Population")

ax3.scatter('Black', 'Inf-Adj Median Household Income', data=no_names, color='darkgreen')
ax3.set_xlabel("Black")
ax3.set_ylabel("Inflation-Adjusted Median Household Income")

ax4.scatter('Black', 'Education', data=no_names, color='darkgreen')
ax4.set_xlabel("Black")
ax4.set_ylabel("Education")

ax5.scatter('Black', 'Unemployment', data=no_names, color='darkgreen')
ax5.set_xlabel("Black")
ax5.set_ylabel("Unemployment")
plt.show()

# Scatter plot 10: Unemployment vs all other relevant variables (variable pairs already plotted not visualized)
fig10 = plt.figure(figsize=(16, 16))
ax1 = fig10.add_subplot(2, 2, 1)
ax2 = fig10.add_subplot(2, 2, 2)
ax3 = fig10.add_subplot(2, 2, 3)
ax4 = fig10.add_subplot(2, 2, 4)

ax1.scatter('Unemployment', 'Property Crime Arrests per capita', data=no_names, color='darkorange')
ax1.set_xlabel("Unemployment")
ax1.set_ylabel("Property Crime Arrests per capita")

ax2.scatter('Unemployment', 'Population', data=no_names, color='darkorange')
ax2.set_xlabel("Unemployment")
ax2.set_ylabel("Population")

ax3.scatter('Unemployment', 'Inf-Adj Median Household Income', data=no_names, color='darkorange')
ax3.set_xlabel("Unemployment")
ax3.set_ylabel("Inflation-Adjusted Median Household Income")

ax4.scatter('Unemployment', 'Education', data=no_names, color='darkorange')
ax4.set_xlabel("Unemployment")
ax4.set_ylabel("Education")
plt.show()

# Scatter plot 11: Education vs all other relevant variables (variable pairs already plotted not visualized)
fig11 = plt.figure(figsize=(16, 16))
ax1 = fig11.add_subplot(2, 2, 1)
ax2 = fig11.add_subplot(2, 2, 2)
ax3 = fig11.add_subplot(2, 2, 3)

ax1.scatter('Education', 'Property Crime Arrests per capita', data=no_names, color='purple')
ax1.set_xlabel("Education")
ax1.set_ylabel("Property Crime Arrests per capita")

ax2.scatter('Education', 'Population', data=no_names, color='purple')
ax2.set_xlabel("Education")
ax2.set_ylabel("Population")

ax3.scatter('Education', 'Inf-Adj Median Household Income', data=no_names, color='purple')
ax3.set_xlabel("Education")
ax3.set_ylabel("Inflation-Adjusted Median Household Income")
plt.show()

# Scatter plot 12: Inf-Adj HHIC vs all other relevant variables (variable pairs already plotted not visualized)
fig12 = plt.figure(figsize=(16, 16))
ax1 = fig12.add_subplot(1, 2, 1)
ax2 = fig12.add_subplot(1, 2, 2)

ax1.scatter('Inf-Adj Median Household Income',
            'Property Crime Arrests per capita', data=no_names, color='dodgerblue')
ax1.set_xlabel("Inflation-Adjusted Median Household Income")
ax1.set_ylabel("Property Crime Arrests per capita")

ax2.scatter('Inf-Adj Median Household Income', 'Population', data=no_names, color='dodgerblue')
ax2.set_xlabel("Inflation-Adjusted Median Household Income")
ax2.set_ylabel("Population")
plt.show()

# Final Scatter Plot: Population against Property Crime Arrests per Capita
no_names.plot(kind='scatter', x='Population', y='Property Crime Arrests per capita', color='midnightblue')
plt.show()

def robust_mahalanobis_method(df):
    # Minimum covariance determinant
    rng = np.random.RandomState(0)
    real_cov = np.cov(df.values.T)
    X = rng.multivariate_normal(mean=np.mean(df, axis=0), cov=real_cov, size=506)
    cov = MinCovDet(random_state=0).fit(X)
    mcd = cov.covariance_  # robust covariance metric
    robust_mean = cov.location_  # robust mean
    inv_covmat = sp.linalg.inv(mcd)  # inverse covariance metric

    # Robust M-Distance
    x_minus_mu = df - robust_mean
    left_term = np.dot(x_minus_mu, inv_covmat)
    mahal = np.dot(left_term, x_minus_mu.T)
    md = np.sqrt(mahal.diagonal())

    # Flag as outlier
    outlier = []
    C = np.sqrt(chi2.ppf((1 - 0.001), df=df.shape[1]))  # degrees of freedom = number of variables
    for index, value in enumerate(md):
        if value > C:
            outlier.append(index)
        else:
            continue
    return outlier, md


dropped_totals = no_names.drop(['Total Drug Arrests per capita'], axis=1)
outliers_mahal_rob, md_rb = robust_mahalanobis_method(df=dropped_totals)

print('List of multivariate outliers')
print(outliers_mahal_rob)
# Checking how many outliers there are
print('Shape before removing outliers:', dropped_totals.shape)
dropped_no_outliers = dropped_totals[(np.abs(stats.zscore(dropped_totals)) < 3).all(axis=1)]
print('Shape after rejecting outliers: ', dropped_no_outliers.shape)

