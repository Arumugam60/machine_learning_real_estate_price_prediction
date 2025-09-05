import matplotlib
import pandas as pd
import numpy as np
import pickle
import json
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn import linear_model
df1=pd.read_csv(r"C:\Users\STAR\Downloads\bengaluru_house_prices.csv")
#---------------Drop unwanted columns to maintain the simplicity--------------------
df1.drop(['area_type','society','balcony','availability'],axis='columns',inplace=True)
#---------------Data cleaning and preprocessing--------------------------
df1.dropna(inplace=True)
      #---------'size' feature have different values(preprocess)-----
#print(df3['size'].unique())
df1["bhk"]=df1["size"].apply(lambda x:int(x.split(' ')[0])) # size col data number is divided from text based on space
#------A function to check the given value is float or not------------------
def is_float(x):
    try:
        float(x)
    except:
        return False
    return True
#print(df1[~df1['total_sqft'].apply(is_float)].head(10))
#-------A function to convert ranges to a value-----------------------------
def convert_sqft_to_num(x):
    tokens = x.split('-')
    if len(tokens) == 2:
        return (float(tokens[0])+float(tokens[1]))/2
    try:
        return float(x)
    except:
        return None
df2=df1.copy()
df2["total_sqft"]=df2["total_sqft"].apply(convert_sqft_to_num) # take care of NA values as the function may written null values
df2.dropna(inplace=True)
#----------Add a new column price_per_sqft for better info----------------
df3=df2.copy()
df3["price_per_sqft"]=df3["price"]*100000/df3["total_sqft"]
#----------Location's dimensionality reduction-------------------------
df3["location"]=df3["location"].apply(lambda x:x.strip()) #Just to remove extra space
location_stat=df3.groupby("location")["location"].agg("count")
df3["location"]=df3["location"].apply(lambda x: 'other' if location_stat[x]<=10 else x)
#----------Outlier Removal Using Business Logic------------------------
df3=df3[~(df3["total_sqft"]/df3["bhk"]<300)]
#-----------Outlier Removal Using Standard Deviation and Mean-----------------
      #------A function is defined to find and remove data points according to location
def remove_pps_outliers(df):
    df_out = pd.DataFrame()
    for key, sub_df in df.groupby('location'):
        m = np.mean(sub_df.price_per_sqft)
        st = np.std(sub_df.price_per_sqft)
        reduced_df = sub_df[(sub_df.price_per_sqft>(m-st)) & (sub_df.price_per_sqft<=(m+st))]
        df_out = pd.concat([df_out,reduced_df],ignore_index=True)
    return df_out
df3=remove_pps_outliers(df3)
    #-------A function is defined to view the pricing of 2bhk and 3bhk in  same location-------
def plot_scatter_chart(df,location):
    bhk2 = df[(df.location==location) & (df.bhk==2)]
    bhk3 = df[(df.location==location) & (df.bhk==3)]
    matplotlib.rcParams['figure.figsize'] = (15,10)
    plt.scatter(bhk2.total_sqft,bhk2.price,color='blue',label='2 BHK', s=50)
    plt.scatter(bhk3.total_sqft,bhk3.price,marker='+', color='green',label='3 BHK', s=50)
    plt.xlabel("Total Square Feet Area")
    plt.ylabel("Price (Lakh Indian Rupees)")
    plt.title(location)
    plt.legend()
    plt.show()
#plot_scatter_chart(df3, "Hebbal")
    #--------A function is defined to remove the outliner (price of 3bhk less than 2bhk)-------
def remove_bhk_outliers(df):
    exclude_indices = np.array([])
    for location, location_df in df.groupby('location'):
        bhk_stats = {}
        for bhk, bhk_df in location_df.groupby('bhk'):
            bhk_stats[bhk] = {
                'mean': np.mean(bhk_df.price_per_sqft),
                'std': np.std(bhk_df.price_per_sqft),
                'count': bhk_df.shape[0]
            }
        for bhk, bhk_df in location_df.groupby('bhk'):
            stats = bhk_stats.get(bhk-1)
            if stats and stats['count']>5:
                exclude_indices = np.append(exclude_indices, bhk_df[bhk_df.price_per_sqft<(stats['mean'])].index.values)
    return df.drop(exclude_indices,axis='index')
df4=remove_bhk_outliers(df3)
#------------------Histogram view of prices per sqft(done to see the type of distribution(normal))-----------------------------------
matplotlib.rcParams["figure.figsize"] = (20,10)
plt.hist(df4.price_per_sqft,rwidth=0.8)
plt.xlabel("Price Per Sqft")
plt.ylabel("Count")
#-----------------Outlier Removal Using Bathrooms Feature------------------------
df4["bath"] = df4["bath"].where(df4["bath"] < df4["bhk"] + 2, None) # remove when bathrooms are greater than and equal to bedrooms+2
df4.dropna(inplace=True)
#------------------Drop the unnecessary columns before training---------------------
df5=df4.drop(["size","price_per_sqft"],axis=1)
#-------------------One hot encoding for text features-------------------------
dummies=pd.get_dummies(df5["location"]).astype(int)
df6=pd.concat([df5,dummies],axis=1)
df6.drop(["location","other"],axis=1,inplace=True)
#--------------------Training the model------------------------------------
x=df6.drop(["price"],axis=1)
y=df6["price"]
x_train,x_test,y_train,y_test=model_selection.train_test_split(x,y,test_size=0.2,random_state=10)
lr_model=linear_model.LinearRegression()
lr_model.fit(x_train,y_train)
#---------------------Checking the score using different methods-------------------------------
#cv=model_selection.ShuffleSplit(n_splits=5,test_size=0.2,random_state=0)
#print(model_selection.cross_val_score(linear_model.LinearRegression(),x,y,cv=cv))
#---------------------Predicting the prices---------------------------
      #-----A function to give input for prediction as there are 243 columns----------
def predict_price(location,sqft,bath,bhk):
    loc = np.where(x.columns == location)[0]
    loc_index = loc[0] if loc.size > 0 else -1

    x_in = np.zeros(len(x.columns))
    x_in[0] = sqft
    x_in[1] = bath
    x_in[2] = bhk
    if loc_index >= 0:
        x_in[loc_index] = 1
    pdf=pd.DataFrame([x_in],columns=x.columns)
    return lr_model.predict(pdf)[0]
#print(predict_price('Indira Nagar',1000, 3, 3))
#---------------Export the tested model to a pickle file-----------------------
with open('home_prices_model.pickle','wb') as f:
    pickle.dump(lr_model,f)
#--------------Export location and column information to a file-------------------------
columns = {
    'data_columns' : [col.lower() for col in x.columns]
}
with open("columns.json","w") as f:
    f.write(json.dumps(columns))
