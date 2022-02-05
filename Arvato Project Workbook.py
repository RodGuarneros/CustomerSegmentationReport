#!/usr/bin/env python
# coding: utf-8

# In[86]:


get_ipython().system('pip install --upgrade scikit-learn')


# In[88]:


# import libraries here; add more as necessary
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.impute import SimpleImputer

# magic word for producing visualizations in notebook
get_ipython().run_line_magic('matplotlib', 'inline')


# # Objective
# 
# Create a customer segmentation report based on census information and e-mail sales by a company, using demographic information to determine how customers are different to general population. Then use this analysis to make predictions to figure out wich members of the general population are more lekely to become a customer for the e-mail order company, <b>based on a unsupervised machine learning model</b>.
# 
# Based on this report, the company would be able to define a marketing strategy so as to reach more consumer out.

# ## Get to Know the Data (Metadata)
# 
# There are four data files associated with this project:
# 
# - `Udacity_AZDIAS_052018.csv`: Demographics data for the general population of Germany; 891 211 persons (rows) x 366 features (columns).
# - `Udacity_CUSTOMERS_052018.csv`: Demographics data for customers of a mail-order company; 191 652 persons (rows) x 369 features (columns).
# - `Udacity_MAILOUT_052018_TRAIN.csv`: Demographics data for individuals who were targets of a marketing campaign; 42 982 persons (rows) x 367 (columns).
# - `Udacity_MAILOUT_052018_TEST.csv`: Demographics data for individuals who were targets of a marketing campaign; 42 833 persons (rows) x 366 (columns).
# 
# Each row of the demographics files represents a single person, but also includes information outside of individuals, including information about their household, building, and neighborhood. Use the information from the first two files to figure out how customers ("CUSTOMERS") are similar to or differ from the general population at large ("AZDIAS"), then use your analysis to make predictions on the other two files ("MAILOUT"), predicting which recipients are most likely to become a customer for the mail-order company.
# 
# The "CUSTOMERS" file contains three extra columns ('CUSTOMER_GROUP', 'ONLINE_PURCHASE', and 'PRODUCT_GROUP'), which provide broad information about the customers depicted in the file. The original "MAILOUT" file included one additional column, "RESPONSE", which indicated whether or not each recipient became a customer of the company. For the "TRAIN" subset, this column has been retained, but in the "TEST" subset it has been removed; it is against that withheld column that your final predictions will be assessed in the Kaggle competition.
# 
# Otherwise, all of the remaining columns are the same between the three data files. For more information about the columns depicted in the files, you can refer to two Excel spreadsheets provided in the workspace. [One of them](./DIAS Information Levels - Attributes 2017.xlsx) is a top-level list of attributes and descriptions, organized by informational category. [The other](./DIAS Attributes - Values 2017.xlsx) is a detailed mapping of data values for each feature in alphabetical order.
# 
# In the below cell, we've provided some initial code to load in the first two datasets. Note for all of the `.csv` data files in this project that they're semicolon (`;`) delimited, so an additional argument in the [`read_csv()`](https://pandas.pydata.org/pandas-docs/stable/generated/pandas.read_csv.html) call has been included to read in the data properly. Also, considering the size of the datasets, it may take some time for them to load completely.
# 
# You'll notice when the data is loaded in that a warning message will immediately pop up. Before you really start digging into the modeling and analysis, you're going to need to perform some cleaning. Take some time to browse the structure of the data and look over the informational spreadsheets to understand the data values. Make some decisions on which features to keep, which features to drop, and if any revisions need to be made on data formats. It'll be a good idea to create a function with pre-processing steps, since you'll need to clean all of the datasets before you work with them.
# 
# The DIAS information level includes the sort of mindful by every row:
# 
# - social minded
# - familiar minded
# - religious
# - material minded
# - dreamily
# - sensual minded
# - eventful orientated
# - cultural minded
# - rational mind
# - critical minded
# - dominant minded
# - fightfull attitude
# - traditional minded
# - traditional minded

# # The approach used to know can be split in the following steps:
# 
# ###    A. Extract, Transform and Load
# 
#  <a font-color: blue> Extract:</a> Gathering the information from every dataset.
#  <p>
#  <a font-color: blue> Transform:</a> Data cleaning, summarization, selection, joining, filtering and aggregating.
#  <p>
#  <a font-color: blue> Load:</a> Relational or not relational database, locally or in AWS.
# 
# ###    B. Exploratory Data Analysis (EDA)
# ###    C. Unsupervised Machine Learning Model: Clustering analysis
# ###    D. Customer segmentation report
# ###    E. Supervised Machine Learning Model: Targeting for Marketing
# 

# <h1 style="color:blue"> A. Extract, Transform and Load</h1>
#     
# ### Extract

# In[3]:


# load in the data
azdias = pd.read_csv('../../data/Term2/capstone/arvato_data/Udacity_AZDIAS_052018.csv', sep=';')
customers = pd.read_csv('../../data/Term2/capstone/arvato_data/Udacity_CUSTOMERS_052018.csv', sep=';')


# In[4]:


#display all the columns of dataset
pd.set_option('display.max_columns',None)


# In[5]:


#display all the rows of dataset
pd.set_option('display.max_rows',None)


# In[6]:


# Shape and head of general data
print(f"General population shape: {azdias.shape}")
print(f"General costumer shape: {customers.shape}")


# 
# 

# 
# ### Transform

# In[7]:


# Overview of features
azdias.head()


# In[8]:


# Verify the shape (observations and variables)
print(f'Azdias, the general population dataset, has {azdias.shape[0]} observations and {azdias.shape[1]} columns')
print(f'customers, the customers dataset, has {customers.shape[0]} observations and {customers.shape[1]} columns')
      


# In[9]:


# We need to confirm which are the 3 additional variables are in customers
print(f'These are the additional variables in {customers.columns.symmetric_difference(azdias.columns)}')


# In[10]:


# Colomuns information, knowing more about the series
toplevel_features = pd.read_excel('./DIAS Information Levels - Attributes 2017.xlsx') 
detailed_features = pd.read_excel("./DIAS Attributes - Values 2017.xlsx")


# In[11]:


detailed_features.tail()


# In[12]:


toplevel_features.tail()


# In[13]:


#Getting the general information in census and users
print(f'The azdias: {azdias.info()}')
print(f'The customers: {customers.info()}')


# In[14]:


# we need the same types in every column, but we would like to know the unique for every column
for col in azdias:
    print(f'{col} {azdias[col].unique()}')


# In[15]:


for col in customers:
    print(f'{col} {customers[col].unique()}')


# In[16]:


# There are 6 columns object. The dtype object comes from NumPy, it describes the type of element in a ndarray. 
# Every element in an ndarray must have the same size in bytes. For int64 and float64, they are 8 bytes. 
# But for strings, the length of the string is not fixed. So instead of saving the bytes of strings in the ndarray directly, 
# Pandas uses an object ndarray, which saves pointers to objects; because of this the dtype of this kind ndarray is object.

# Lets see the 6 object series in azdias and 8 object series in customers dataset

print(f'The columns as object in customers: {customers.select_dtypes(object).columns}')
print(f'The columns as object in azdias: {azdias.select_dtypes(object).columns}')


# In[17]:


# lets understand the variables dtype objects:
azdias_objects = azdias[['CAMEO_DEU_2015', 'CAMEO_DEUG_2015', 'CAMEO_INTL_2015',
       'D19_LETZTER_KAUF_BRANCHE', 'EINGEFUEGT_AM', 'OST_WEST_KZ']]

for col in azdias_objects:
    print(f'{col} {azdias[col].unique()}')


# In[18]:


# lets understand the variables dtype objects:
customers_objects = customers[['CAMEO_DEU_2015', 'CAMEO_DEUG_2015', 'CAMEO_INTL_2015',
       'D19_LETZTER_KAUF_BRANCHE', 'EINGEFUEGT_AM', 'OST_WEST_KZ',
       'PRODUCT_GROUP', 'CUSTOMER_GROUP']]

for col in customers_objects:
    print(f'{col} {customers[col].unique()}')


# In[19]:


# Given the last information we need to transform the unknown values in missing values in three columns for every dataset
# Azdias and customers: 'CAMEO_DEU_2015', 'CAMEO_DEUG_2015', 'CAMEO_INTL_2015
# The problem of mixed types in columns is because there is actually a mix of int and string values. 
# Therefore, I will remove the string value and conver the column data type to int. Particularly for 
# 'CAMEO_DEUG_2015' and 'CAMEO_INTL_2015'

def mixed_types(df):
    '''This function is created for formating improper 
    values in columns CAMEO_DEUG_2015 and CAMEO_INTL_2015.
    Args:
    df: dataframe
    returns: transformed dataframe
    '''
    
    cols_nan = ['CAMEO_DEU_2015', 'CAMEO_DEUG_2015', 'CAMEO_INTL_2015']
    cols = ['CAMEO_DEUG_2015', 'CAMEO_INTL_2015'] #cols with int dtypes
    if set(cols_nan).issubset(df.columns):
        df[cols_nan] = df[cols_nan].replace({'X': np.nan, 'XX': np.nan})
        df[cols] = df[cols].astype(float)

    return df


# In[20]:


# using the function to eliminate unknown values in customeres
customers2 = mixed_types(customers)

print(f'The columns as object in customers2: {customers2.select_dtypes(object).columns}')


# In[21]:


customers_objects = customers[['CAMEO_DEU_2015', 'CAMEO_DEUG_2015', 'CAMEO_INTL_2015',
       'D19_LETZTER_KAUF_BRANCHE', 'EINGEFUEGT_AM', 'OST_WEST_KZ',
       'PRODUCT_GROUP', 'CUSTOMER_GROUP']]

for col in customers_objects:
    print(f'{col} {customers2[col].unique()}')


# In[22]:


# using the function to eliminate unknown values in azdias (census dataset)
azdias2 = mixed_types(azdias)

print(f'The columns as object in azdias2: {azdias2.select_dtypes(object).columns}')


# In[24]:


azdias2_objects = azdias2[['CAMEO_DEU_2015', 'D19_LETZTER_KAUF_BRANCHE', 'EINGEFUEGT_AM',
       'OST_WEST_KZ']]

for col in azdias_objects:
    print(f'{col} {azdias2[col].unique()}')


# In[25]:


# How look the missing values proportion by column at census dataset?

#number of missing values by each column in azdias dataset
col_nul_percent_azdias=(azdias2.isnull().sum()/azdias2.shape[0])*100


# In[27]:


plt.title("Distribution of missing values in azdias dataset",fontsize=13,fontweight="bold")
plt.xlabel("colums names",fontsize=13)
plt.ylabel("Precent of missing values",fontsize=13)
(col_nul_percent_azdias.sort_values(ascending=False)[:50].plot(kind='bar', figsize=(20,8), fontsize=13));


# In[28]:


# How look the missing values proportion by column at customers dataset?

# number of missing values by each column in costumers dataset
col_nul_percent_customers=(customers2.isnull().sum()/customers2.shape[0])*100


# In[29]:


plt.title("Distribution of missing values in customers dataset",fontsize=13,fontweight="bold")
plt.xlabel("colums names",fontsize=13)
plt.ylabel("Precent of missing values",fontsize=13)
(col_nul_percent_customers.sort_values(ascending=False)[:50].plot(kind='bar', figsize=(20,8), fontsize=13));


# ### Missing values reduction
# The cleaning process includes missing and unknow values treatment, we have several columns with missings and unknown data. 
# We need to clean and eliminate variables with a "reasonable" number of missings and unknown values. But... <b>What is reasonable?</b> 
# Based in the principled missing data methods for researchers, available at: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3701793/. 
# I decided to use 20% as a limit, having in mind that the the proportion of missing data is directly related to the quality of statistical inferences. 
# 
# For azdias dataset we'd eliminate 16 columns and for customers dataset we're talking about 6 features.

# In[30]:


#droping columns that have more than 20% of missing values
column_nans = (azdias2.isnull().sum()/azdias2.shape[0])*100
drop_cols = azdias2.columns[column_nans > 20]
print('columns to drop: ', drop_cols)


# In[31]:


#droping columns that have more than 20% of missing values
column_nans = (customers2.isnull().sum()/customers2.shape[0])*100
drop_cols = customers2.columns[column_nans >= 20]
print('columns to drop: ', drop_cols)


# In[32]:


# let's see the distribution of missing values by column in azdias dataset
plt.hist(col_nul_percent_azdias, bins=40);

plt.xlabel('Proportion of missing values (%)')
plt.ylabel('Number of AZDIAS features')
plt.title('Proportion of missing values in AZDIAS features')
plt.show()


# As we can see in the histogram and having, must of the feateres has less of 20% of missing values.
# This histogram told us that a 20% is a reasonable limit for missing values.
# 

# In[33]:


# Inter quantile range
quartiles = col_nul_percent_azdias.quantile([.20,.40,.60, .80])

lowerq = quartiles[.20]
upperq = quartiles[.80]

IQR = upperq- lowerq

upperbound = lowerq-(IQR*1.4)
lowerbound = upperq+(IQR*1.4)


# In[34]:


# Potential outliers
outliers = col_nul_percent_azdias>lowerbound


# In[35]:


first_quartile = col_nul_percent_azdias.quantile(0.99)
first_quartile


# In[36]:


# Defining the features to drop
features_to_drop = col_nul_percent_azdias[col_nul_percent_azdias > 20].index
features_to_drop


# In[39]:


azdias2.drop(labels=features_to_drop, axis=1, inplace=True)


# In[40]:


azdias2.shape


# # Missing values per row

# In[41]:


percent_missing_row = azdias2.isnull().mean(axis=1) * 100
percent_missing_row


# In[43]:


plt.hist(percent_missing_row, bins=40);

plt.xlabel('Proportion of missing values (%)')
plt.ylabel('Number of AZDIAS rows')
plt.title('Proportion of missing values in AZDIAS rows')
plt.show()


# - The largest number of rows have few missing values, even zero, however there are some that have up to 60 missing values
# - It is reasonable to eliminate the rows with more than 10% missing values. 

# In[44]:


azdias3 = azdias2[percent_missing_row <= 10]


# In[45]:


print('AZDIAS|Data shape (rows, cols): ', azdias3.shape)


# * LP_LEBENSPHASE_FEIN and LP_LEBENSPHASE_GROB are another two mixed features. Among the six columns that start with LP, there are duplicated information and we might need to do some additional data cleaning.

# In[46]:


print('LP_LEBENSPHASE_FEIN: ', len(azdias["LP_LEBENSPHASE_FEIN"].unique()),       'values, ', azdias["LP_LEBENSPHASE_FEIN"].unique())

print('LP_LEBENSPHASE_GROB: ', len(azdias["LP_LEBENSPHASE_GROB"].unique()),       'values, ', azdias["LP_LEBENSPHASE_GROB"].unique())

print('LP_STATUS_FEIN: ', len(azdias["LP_STATUS_FEIN"].unique()),       'values, ', azdias["LP_STATUS_FEIN"].unique())

print('LP_STATUS_GROB: ', len(azdias["LP_STATUS_GROB"].unique()),       'values, ', azdias["LP_STATUS_GROB"].unique())

print('LP_FAMILIE_FEIN: ', len(azdias["LP_FAMILIE_FEIN"].unique()),       'values, ', azdias["LP_FAMILIE_FEIN"].unique())

print('LP_FAMILIE_GROB: ', len(azdias["LP_FAMILIE_GROB"].unique()),       'values, ', azdias["LP_FAMILIE_GROB"].unique())


# Columns LP_LEBENSPHASE_FEIN, LP_LEBENSPHASE_GROB, LP_FAMILIE_FEIN and LP_FAMILIE_GROB have 0 as one category. However, 0 should not be included according to DIAS Attributes - Values 2017.xlsx. Therefore, any 0s in the four columns will be converted into nan.
# 
# If we further look into DIAS Attributes - Values 2017.xlsx, we will find that LP_FAMILIE_GROB and LP_STATUS_GROB are broderer categorization of LP_FAMILIE_FEIN and LP_STATUS_FEIN, respectively. Since each pair contains the same information and the two fine categorizations have over 10 values, LP_FAMILIE_FEIN and LP_STATUS_FEIN will be dropped. Similarly, both LP_LEBENSPHASE_GROB and LP_LEBENSPHASE_FEIN will be dropped since they are complex mixed-features.
# 
# LP_STATUS_GROB and LP_FAMILIE_GROB can be re-encoded as the following.
# 
# LP_STATUS_GROB:
# 
# Social status
# 
# 1: low-income earners
# 
# 2: average earners
# 
# 3: independants
# 
# 4: houseowners
# 
# 5: top earners
# 
# LP_FAMILIE_GROB:
# 
# Family type
# 
# 1: single
# 
# 2: couple
# 
# 3: single parent
# 
# 4: family
# 
# 5: multiperson household

# # Creating a function to LP_ functions
# - replace 0 with nan values
# - drop complex and/or duplicate columns
# - split columns with mixed features.

# In[47]:


# lp is short for LP_*
def encode_lp_files(df):
    """
    Re-encode LP_* columns:
    - replace 0s with nan values
    - drop complex and/or duplicate columns
    - split columns with mixed features.
    """
    # replace 0s with nan values
    cols = ["LP_LEBENSPHASE_FEIN","LP_LEBENSPHASE_GROB", "LP_FAMILIE_FEIN", "LP_FAMILIE_GROB"]
    df[cols] = df[cols].replace({0: np.nan})
    
    # drop complex and/or duplicate columns
    df.drop(['LP_FAMILIE_FEIN','LP_STATUS_FEIN','LP_LEBENSPHASE_FEIN','LP_LEBENSPHASE_GROB'], axis=1, inplace=True)
    
    # re-encode LP_STATUS_GROB and LP_FAMILIE_GROB
    status = {1: 1, 2: 1, 3: 2, 
              4: 2, 5: 2, 6: 3, 
              7: 3, 8: 4, 9: 4, 
              10: 5}
    familie = {1: 1, 2: 2, 3: 3, 
               4: 3, 5: 3, 6: 4, 
               7: 4, 8: 4, 9: 5, 
               10: 5, 11: 5}
    df["LP_STATUS_GROB"] = df["LP_STATUS_GROB"].map(status)                                              
    df["LP_FAMILIE_GROB"] = df["LP_FAMILIE_GROB"].map(familie)    
    
    return df


# In[48]:


azdias = encode_lp_files(azdias3)


# In[49]:


azdias.shape


# - The 4th mixed-feature column is PRAEGENDE_JUGENDJAHRE. This feature takes unknown values being -1 and 0 and known values ranging from 1 to 15 and. It represents two types of donimating movements in the person's youth.

# In[50]:


# How this looks like?

print('PRAEGENDE_JUGENDJAHRE: ', len(azdias["PRAEGENDE_JUGENDJAHRE"].unique()),       'values, ', azdias["PRAEGENDE_JUGENDJAHRE"].unique())


# # Function to re-encode this feature to contain two values being Mainstream and Avantgarde, respectively.
# 
# * Mainstream: 0 -> [1, 3, 5, 8, 10, 12, 14]
# * Avantgarde: 1 -> [2, 4, 6, 7, 9, 11, 13, 15]

# In[51]:


# encode PRAEGENDE
def encode_PRAEGENDE(x):
    """
    Re-encode PRAEGENDE_JUGENDJAHRE:
    0: Mainstream
    1: Avantgarde
    """
    mainstream = [1, 3, 5, 8, 10, 12, 14]
    avantgarde = [2, 4, 6, 7, 9, 11, 13, 15]
    if x in mainstream: 
        return 0
    elif x in avantgarde: 
        return 1
    else:
        return x


# In[52]:


azdias['PRAEGENDE_JUGENDJAHRE'] = azdias['PRAEGENDE_JUGENDJAHRE'].apply(lambda x: encode_PRAEGENDE(x))


# In[53]:


# See the unique values in column PRAEGENDE_JUGENDJAHRE
print('PRAEGENDE_JUGENDJAHRE: ', len(azdias["PRAEGENDE_JUGENDJAHRE"].unique()),       'values, ', azdias["PRAEGENDE_JUGENDJAHRE"].unique())


# In[54]:


azdias.shape


# # High correlated features treatment (avoiding the impact of "Multicollinearity" in the performance of the model)
# 
# We will be implementing feature correlation to determine too high-correlated features since they many over-inflate the importance of a single feature. Too highly-correlated features are defined as having correlations with a column over 0.9.
# 
# Correlated features means that they bring the same information.
# 
# Data and feature correlation is considered one important step in the feature selection phase of the data pre-processing especially if the data type for the features is continuous.
# 
# 
# 

# In[55]:


correlation_matrix = azdias.corr().abs().round(2)


# In[61]:


columns_total = list(azdias.columns.values)


# In[70]:


# plot correlation matrix
fig = plt.figure(figsize=(40, 30))
ax = fig.add_subplot(111)

names = columns_total

cax = ax.matshow(correlation_matrix, vmin=-1, vmax=1)
ticks = np.arange(0,343,1)
ax.set_xticks(ticks)
ax.set_yticks(ticks)
ax.set_xticklabels(names, rotation="vertical")
ax.set_yticklabels(names)
fig.colorbar(cax)
ax.set_title("Feature Selection Heatmap", size=30)
plt.savefig('heatmap.png')
plt.show()


# * There are high crrelated features, as we can see in the heatmap.
# * Lets drop those high correlated features in the following limits:
# - correlation > +0.99
# - correlation < -0.99

# In[72]:


corr_features = set()
for i in range(len(correlation_matrix.columns)):
    for j in range(i):
        if abs(correlation_matrix.iloc[i, j]) > 0.9:
            col = correlation_matrix.columns[i]
            corr_features.add(col)


# In[74]:


# There are 16 features highly correlated
len(corr_features)


# In[75]:


azdias.drop(labels=corr_features, axis=1, inplace=True)


# In[76]:


azdias.shape


# # Imputing missing values
# 
# We are going to use a simple imputer (univariate imputation by mean).
# 

# In[77]:


imputer = SimpleImputer(missing_values=np.nan, strategy='median')
azdias_imputed = pd.DataFrame(imputer.fit_transform(azdias))


# ## Part 1: Customer Segmentation Report
# 
# The main bulk of your analysis will come in this part of the project. Here, you should use unsupervised learning techniques to describe the relationship between the demographics of the company's existing customers and the general population of Germany. By the end of this part, you should be able to describe parts of the general population that are more likely to be part of the mail-order company's main customer base, and which parts of the general population are less so.

# In[ ]:





# ## Part 2: Supervised Learning Model
# 
# Now that you've found which parts of the population are more likely to be customers of the mail-order company, it's time to build a prediction model. Each of the rows in the "MAILOUT" data files represents an individual that was targeted for a mailout campaign. Ideally, we should be able to use the demographic information from each individual to decide whether or not it will be worth it to include that person in the campaign.
# 
# The "MAILOUT" data has been split into two approximately equal parts, each with almost 43 000 data rows. In this part, you can verify your model with the "TRAIN" partition, which includes a column, "RESPONSE", that states whether or not a person became a customer of the company following the campaign. In the next part, you'll need to create predictions on the "TEST" partition, where the "RESPONSE" column has been withheld.

# In[ ]:


mailout_train = pd.read_csv('../../data/Term2/capstone/arvato_data/Udacity_MAILOUT_052018_TRAIN.csv', sep=';')


# In[ ]:




