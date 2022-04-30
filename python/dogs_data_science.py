"""
Name:       Ken Ko
Email:      ken.ko17@myhunter.cuny.edu
Resources:  http://www.textbook.ds100.org/ch/24/classification_cost.html.
            http://www.textbook.ds100.org/ch/19/mult_model.html
            https://data.beta.nyc/dataset/nyc-zip-code-tabulation-areas/resource/6df127b1-6d04-4bb7-b983-07402a2c3f90
            https://data.cityofnewyork.us/Health/NYC-Dog-Licensing-Dataset/nu7n-tubp
            https://data.cityofnewyork.us/Health/DOHMH-Dog-Bite-Data/rsgh-akpg
            https://www.akc.org/dog-breeds/
            http://jmcglone.com/guides/github-pages/
Title:      All Bite and No Bark - Predicting the Probability of Getting Bitten by a Random Dog
URL:        https://kenko1290.github.io
"""

import pandas as pd
import numpy as np
import random
import folium
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
import seaborn as sns

#############################################################################################################################################################
######################################################### INITIAL CLEANING ##################################################################################
#############################################################################################################################################################


def make_license_df_lvl_1(file_name): # open csv file and filter out rows/columns
    df = pd.read_csv(file_name)
    df.rename(columns={"AnimalGender":"Gender"}, inplace=True)
    df = df.loc[~df['BreedName'].str.contains(pat='Unknown')]   # drop all columns that contain "Unknown" BreedName
    return df


def make_bite_df_lvl_1(file_name): # open csv file and filter out rows/columns
    df = pd.read_csv(file_name)
    df.rename(columns={"Breed":"BreedName"}, inplace=True)
    df = df.dropna(axis=0, subset=['BreedName'])                # drop all columns that contain no BreedName
    return df

# This function takes as input the name of a csv file containing all the dog species recognized by AKC to be in one of the 7 dog groups: working, herding,
# sporting, hound, terrier, toy, and non-sporting. The function takes the data and converts it into a dictionary with the dog groups as the keys and breed
# names in a string array as the values.
def make_group_dict(file_name):
    df = pd.read_csv(file_name)
    df = df.drop(columns=['RowNumber'])                         # drop the RowNumber Column of the dataset
    group_dict = df.to_dict()
    return group_dict

# This function converts the entries in the LicenseIssuedDate and LicenseExpiredDate columns to datetime objects. Create a new column for each year from
# 2015-2021. The entry in a particular year column is true if that year falls between the LicenseIssuedDate and LicenseExpiredDate for that row. For example,
# if a license is valid from 2015-2018, it will be assumed that a corresponding dog exists for that license for those years and thus, the value in columns
# 2015, 2016, 2017, and 2018 for that row will be true.
def dogs_in_20xx(df):
    #df = df.head(100)
    start_date = pd.to_datetime(df['LicenseIssuedDate'])
    end_date = pd.to_datetime(df['LicenseExpiredDate'])

    df['2015 Total'] = (2015 >= start_date.dt.year) & (2015 <= end_date.dt.year)
    df['2016 Total'] = (2016 >= start_date.dt.year) & (2016 <= end_date.dt.year)
    df['2017 Total'] = (2017 >= start_date.dt.year) & (2017 <= end_date.dt.year)
    df['2018 Total'] = (2018 >= start_date.dt.year) & (2018 <= end_date.dt.year)
    df['2019 Total'] = (2019 >= start_date.dt.year) & (2019 <= end_date.dt.year)
    df['2020 Total'] = (2020 >= start_date.dt.year) & (2020 <= end_date.dt.year)
    df['2021 Total'] = (2021 >= start_date.dt.year) & (2021 <= end_date.dt.year)

    return df

# This function converts the entries in the DateOfBite columns to datetime objects. Create a new column for each year from 2015-2021. The entry in a
# particular year column for a given bite is true if the bite occurred in that year. For example, if a bite occurred in 2015, the value in column
# 2015 for that row will be true.
def bites_in_20xx(df):
    #df = df.head(100)
    bite_date = pd.to_datetime(df['DateOfBite'], format="%B %d %Y")

    df['2015 Bites'] = (2015 == bite_date.dt.year)
    df['2016 Bites'] = (2016 == bite_date.dt.year)
    df['2017 Bites'] = (2017 == bite_date.dt.year)
    df['2018 Bites'] = (2018 == bite_date.dt.year)
    df['2019 Bites'] = (2019 == bite_date.dt.year)
    df['2020 Bites'] = (2020 == bite_date.dt.year)
    df['2021 Bites'] = (2021 == bite_date.dt.year)

    return df

# This function takes an array of strings and checks if all the words in the array are contained in a given string. For example, it checks if all the words in
# ['French', 'Bulldog'] can be found in 'French Bulldog / Boston Terrier Mix'. If so, it returns true. This will be used to help match each dog breed in the
# license and bite datasets with a dog group. Input str will be a string containing a single dog breed from license or bites dataset. Input arr will be an array
# made up of the words comprising a single breed in the Dog_Groups.csv file.
def allArrInString(str, arr):
    for i in arr:
        if i.lower() not in str.lower():
            return False
    return True

# This function takes a single string (breed) and a dictionary containing dog groups as the keys and all the breed names for that group in an array as
# the values (group_dict). The function breaks each breed name in the dictionary into an array of words, then checks if all the words in the array can be found
# in the given string breed. If so, it returns the dog group that that array is part of (i.e. the dictionary key for that particular array)
def find_dog_group(breed, group_dict):
    for i in group_dict:                                # for each Dog Group
        for j in group_dict[i]:                         # for each breed in a Dog Group
            if not pd.isna(group_dict[i][j]):           # ignore NaN values
                words = group_dict[i][j].split()        # split each dog breed into an array of words
                if(allArrInString(breed, words)):       # if all the words in the array are found in given breed, return the current Dog Group it's in
                    return i
    return None

# This function goes through a data set, finds a matching dog group for each dog breed entry and puts it in a new column. It drops any rows that do not have
# a matching dog group.
def add_dog_column(df, group_dict):
    df['Dog Group'] = df['BreedName'].apply(lambda row: find_dog_group(row,group_dict))
    df = df.dropna(axis=0, subset=['Dog Group'])        # drop any rows that could not find a matching Dog Group
    return df


#############################################################################################################################################################
######################################################### SECONDARY CLEANING ################################################################################
#############################################################################################################################################################


def make_license_df_lvl_2(file_name): # open csv file and filter out rows/columns
    df = pd.read_csv(file_name)
    df = df.drop(columns=['RowNumber','AnimalName', 'AnimalBirthMonth', 'Borough', 'Extract Year', 'Unique Dog ID'])
    return df

def make_bite_df_lvl_2(file_name): # open csv file and filter out rows/columns
    df = pd.read_csv(file_name)
    df = df.drop(columns=['UniqueID','Species'])
    df = df.loc[df['Borough'] != 'Other',:]
    compiled_zips = compile_zip_codes(df)
    df['ZipCode'] = df['ZipCode'].fillna(df['Borough'].apply(lambda row: pick_random_zipcode(row,compiled_zips)))
    df['ZipCode'] = df['ZipCode'].apply(lambda row: str(int(row)))
    #df = df.loc[df['Gender'] != "U"]
    return df

# This function will be used to fill in blank entries in the ZipCode column of the bites dataset. It adds all the zip codes encountered for a
# borough into an array. A random function will later pick ones of these zip codes at random, ensuring a distribution of zip codes that keeps the same
# proportion as the original data.
def compile_zip_codes(df):
    compiled_zips = {}

    df = df.loc[df['ZipCode'] >= 10001]                             # filter out invalid zip codes
    df = df.loc[df['ZipCode'] <= 11697]                             # filter out invalid zip codes
    df = df.dropna(axis=0, subset=['ZipCode'])

    boroughs = ['Brooklyn', 'Bronx', 'Manhattan', 'Staten Island', 'Queens']
    for i in boroughs:
        borough = df.loc[df['Borough'] == i]
        borough = borough.groupby(['Borough','ZipCode'])['ZipCode'].count().reset_index(name='Count')
        borough = borough.drop(columns=['Borough'])
        borough = borough.to_dict('index')
        borough_list = []
        for j in borough:
            for k in range(borough[j]['Count']):
                borough_list.append(borough[j]['ZipCode'])
        compiled_zips[i] = borough_list
    return compiled_zips

def pick_random_zipcode(borough, compiled_dict):
    random_list = compiled_dict[borough]
    return random.choice(random_list)


#############################################################################################################################################################
######################################################### CHOROPLETH MAPS ###################################################################################
#############################################################################################################################################################

def make_license_df_lvl_3(file_name): # open csv file and filter out rows/columns
    df = pd.read_csv(file_name)
    df = df.loc[df['ZipCode'] >= 10001]                             # filter out invalid zip codes
    df = df.loc[df['ZipCode'] <= 11697]                             # filter out invalid zip codes
    return df

def make_bite_df_lvl_3(file_name): # open csv file and filter out rows/columns
    df = pd.read_csv(file_name)
    df = df.loc[df['ZipCode'] >= 10001]                             # filter out invalid zip codes
    df = df.loc[df['ZipCode'] <= 11697]                             # filter out invalid zip codes
    return df

def avg_dogs(df): # used to find the number of dogs in every zip code for a particular year. If year=None, return the avg licenses per zip code per year
    # Group dogs by zip code, then count the number of trues in 2015-2022 columns (to get total number of dogs in a given zip code in a given year)
    df = df.groupby(['ZipCode'], as_index=False)[['2015 Total','2016 Total','2017 Total','2018 Total','2019 Total','2020 Total','2021 Total']].sum()
    df['Avg Licenses'] = (df['2015 Total']+df['2016 Total'] + df['2017 Total'] + df['2018 Total'] + df['2019 Total'] + df['2020 Total'] + df['2021 Total']) // 6
    df = df.loc[:,['ZipCode', 'Avg Licenses']]
    return df

def avg_bites(df):  # used to find the number of bite incidences in every zip code for a particular year. If year=None, return the avg bites per zip code per year
    # Group dogs by zip code, then count the number of trues in 2015-2021 columns
    df = df.groupby('ZipCode', as_index=False)[['2015 Bites','2016 Bites','2017 Bites','2018 Bites','2019 Bites','2020 Bites','2021 Bites']].sum()
    df['Avg Bites'] = (df['2015 Bites']+df['2016 Bites'] + df['2017 Bites'] + df['2018 Bites'] + df['2019 Bites'] + df['2020 Bites'] + df['2021 Bites']) // 6
    df = df.loc[:,['ZipCode', 'Avg Bites']]
    return df

def choropeth_map_licenses(df):
    mapNYC = folium.Map(location=[40.75, -74.125], zoom_start=10)
    mapNYC.choropleth(geo_data='nyc.geojson', data=df, columns=['ZipCode', 'Avg Licenses'],
                    key_on='feature.properties.postalCode', fill_color='BuPu', fill_opacity=0.75, line_opacity=0.5)
    mapNYC.save('Avg_Yearly_Licenses_Map.html') 
    return df

def choropeth_map_bites(df):
    mapNYC = folium.Map(location=[40.75, -74.125], zoom_start=10)
    mapNYC.choropleth(geo_data='nyc.geojson', data=df, columns=['ZipCode', 'Avg Bites'],
                    key_on='feature.properties.postalCode', fill_color='OrRd', fill_opacity=0.75, line_opacity=0.5,)
    mapNYC.save('Avg_Yearly_Bites_Map.html')
    return df


#############################################################################################################################################################
######################################################### BAR GRAPHS ########################################################################################
#############################################################################################################################################################

def make_license_df_lvl_4(file_name): # open csv file and filter out rows/columns
    df = pd.read_csv(file_name)
    # take the average number of licenses b/t 2016-2021. 2015 is not included since the license numbers seem really low
    df = df.groupby(['Dog Group','Gender'])[['2015 Total','2016 Total','2017 Total','2018 Total','2019 Total','2020 Total','2021 Total']].sum().reset_index()
    return df

def make_bite_df_lvl_4(file_name): # open csv file and filter out rows/columns
    df = pd.read_csv(file_name)
    # take the average number of bite incidences b/t 2016-2021. 2015 is not included since the license numbers seem really low
    df = df.groupby(['Dog Group','Gender'])[['2015 Bites','2016 Bites','2017 Bites','2018 Bites','2019 Bites','2020 Bites','2021 Bites']].sum().reset_index()
    return df

def replace_unknown_gender(df):
    groups = ['Herding','Hound','Non-Sporting','Sporting','Terrier','Toy','Working']
    for i in groups:
        for year in range(2015,2022):
            unknown = int(df[(df['Gender'] == 'U') & (df['Dog Group'] == i)][str(year) + " Bites"])
            females = int(df[(df['Gender'] == 'F') & (df['Dog Group'] == i)][str(year) + " Bites"])
            males = int(df[(df['Gender'] == 'M') & (df['Dog Group'] == i)][str(year) + " Bites"])
            total = females+males
            df.loc[(df['Gender'] == 'F') & (df['Dog Group'] == i), [str(year) + " Bites"]] += int(unknown*females/total)
            df.loc[(df['Gender'] == 'M') & (df['Dog Group'] == i), [str(year) + " Bites"]] += int(unknown*males/total)
    df = df.loc[df['Gender'] != 'U']
    return df

def merge_df(license_df, bite_df):
    df = pd.merge(license_df, bite_df, on=['Dog Group','Gender'])
    df['Avg Bites'] = (df['2015 Bites']+df['2016 Bites'] + df['2017 Bites'] + df['2018 Bites'] + df['2019 Bites'] + df['2020 Bites'] + df['2021 Bites']) // 6
    df['Avg Licenses'] = (df['2015 Total']+df['2016 Total'] + df['2017 Total'] + df['2018 Total'] + df['2019 Total'] + df['2020 Total'] + df['2021 Total']) // 6
    return df

def avg_bar_plot_group(df):   # plots avg bites vs avg population for 2016-2021 comparing b/t different dog groups
    df_by_group = df.groupby(['Dog Group'])[['Avg Licenses','Avg Bites']].sum().reset_index()
    df_by_group['Percentage'] = df_by_group["Avg Bites"] / df_by_group["Avg Licenses"]*100
    df_by_group.plot.bar(x = "Dog Group", y = "Percentage", legend=None)
    plt.xlabel("Dog Breed")
    plt.ylabel("Percentage(%)")
    plt.title("Number of Bite Incidents / Total Number of Dogs in Group (avg 2015-2021)")
    fig = plt.gcf()
    fig.savefig("Bar plot by group(avg 2015-2021)", bbox_inches='tight')

def avg_bar_plot_gender(df):   # plots avg bites vs avg population for 2016-2021 comparing b/t different genders
    df_by_gender = df.groupby(['Gender'])[['Avg Licenses','Avg Bites']].sum().reset_index()
    df_by_gender['Percentage'] = df_by_gender["Avg Bites"] / df_by_gender["Avg Licenses"]*100
    df_by_gender.plot.bar(x = "Gender", y = "Percentage", legend=None)
    plt.xlabel("Gender")
    plt.ylabel("Percentage(%)")
    plt.title("Number of Bite Incidents / Total Number of Dogs in Gender (avg 2015-2021)")
    fig2 = plt.gcf()
    fig2.savefig("Bar plot gender(avg 2015-2021)", bbox_inches='tight')


#############################################################################################################################################################
######################################################### LOGISTIC REGRESSION ###############################################################################
#############################################################################################################################################################

def make_merged_df(file_name): # open csv file and filter out rows/columns
    df = pd.read_csv(file_name)
    df = df.loc[:, ['Dog Group','Gender','Avg Bites','Avg Licenses']]
    return df

# adds a row for each existing license for a given dog breed and gender, then adds x 1's to the Bites (Y/N) column, where x is the number of bite incidents
# for that given breed and gender.
def explode(df):
    groups = pd.get_dummies(df['Dog Group'])
    genders = pd.get_dummies(df['Gender'])

    df = pd.concat([df,groups,genders], axis=1)
    df_list = df.loc[:,['Avg Bites','Avg Licenses']].values.tolist()
    bites_array = []
    for i in range(len(df['Avg Bites'])):
        temp_array = []
        for j in range(df['Avg Bites'][i]):
            temp_array.append(1)
        for k in range(df['Avg Licenses'][i]-df['Avg Bites'][i]):
            temp_array.append(0)
        bites_array.append(temp_array)

    df.loc[:,['Avg Bites','Avg Licenses']] = df.loc[:,['Avg Bites','Avg Licenses']].apply(lambda row: bites_array)
    df = df.drop(columns=['Avg Licenses'])
    df = df.explode(['Avg Bites'])
    df = df.rename(columns={'Avg Bites':'Bites (Y/N)'}).reset_index()
    return df

def fit_linear_regression(df):
    # splitting the data
    x_cols = ['Herding','Hound','Non-Sporting','Sporting','Terrier','Toy','Working','F','M']
    y_cols = 'Bites (Y/N)'
    x_train, x_test, y_train, y_test = train_test_split(df[x_cols], df[y_cols], test_size = 0.2)
    y_train = y_train.astype('int')
    y_test = y_test.astype('int')

    clf = LogisticRegression()
    model = clf.fit(x_train[x_cols], y_train)
    y_pred = model.predict(x_test)

    resid = y_pred - y_test                                     # find the difference b/t the predicted and actual y-values

    resid_plot = pd.DataFrame(resid).reset_index()              # convert resid to Dataframe for sns plotting
    resid_plot = resid_plot.drop(columns=['index'])
    resid_plot.index.names = ['Index']
    resid_plot = resid_plot.reset_index()

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    control_acc = np.count_nonzero(y_test == 0) / len(y_test)   # accuracy gotten just by guessing 0 on everything
    coefficients = model.coef_
    print("MSE:", mse)
    print("r2:", r2)
    print("Accuracy:",accuracy)
    print("Accuracy just guessing 0 on everything:", control_acc)
    print("Coefficients:", coefficients)
    sns.lmplot(x='Index', y='Bites (Y/N)',
           data=resid_plot,
           fit_reg=False, ci=False,
           y_jitter=0.05,
           scatter_kws={'alpha': 0.3})
    plt.title('Residuals (actual - predicted)')
    plt.xlabel('Index of row in data')
    plt.ylabel('Bite (Y/N)')
    plt.show()
    fig = plt.gcf()
    fig.savefig('Logistic_Regression.png', bbox_inches='tight')

#############################################################################################################################################################
######################################################### MAIN ##############################################################################################
#############################################################################################################################################################

def main():

######################################################### INITIAL CLEANING ##################################################################################

    # Adds a Dog Group column in the license dataset and 2015-2021 columns indicating whether a dog exists in that year or not
    licenses_lvl_1 = make_license_df_lvl_1("NYC_Dog_Licensing_Dataset.csv")
    groups = make_group_dict("Dog_Groups.csv")
    licenses_lvl_1 = add_dog_column(licenses_lvl_1, groups)
    licenses_lvl_1 = dogs_in_20xx(licenses_lvl_1)
    licenses_lvl_1.to_csv("Dog_Licenses_with_Groups_Level_1.csv")
    
    # Adds a Dog Group column in the bites dataset and 2015-2021 columns indicating whether a bite occurred in that year or not
    bites_lvl_1 = make_bite_df_lvl_1("DOHMH_Dog_Bite_Data.csv")
    bites_lvl_1 = bites_in_20xx(bites_lvl_1)
    bites_lvl_1 = add_dog_column(bites_lvl_1, groups)
    bites_lvl_1.to_csv("Dog_Bites_with_Groups_Level_1.csv")

######################################################### SECONDARY CLEANING ################################################################################

    licenses_lvl_2 = make_license_df_lvl_2("Dog_Licenses_with_Groups_Level_1.csv")
    bites_lvl_2 = make_bite_df_lvl_2("Dog_Bites_with_Groups_Level_1.csv")
    licenses_lvl_2.to_csv("licenses_level_2.csv")
    bites_lvl_2.to_csv("bites_level_2.csv")

######################################################### CHOROPLETH MAPS ###################################################################################

    licenses_lvl_3 = make_license_df_lvl_3('licenses_level_2.csv')
    licenses_lvl_3 = avg_dogs(licenses_lvl_3)
    choropeth_map_licenses(licenses_lvl_3)                  # creates a map showing the distribution of dog licenses per zip code in a given year

    bites_lvl_3 = make_bite_df_lvl_3('bites_level_2.csv')
    bites_lvl_3 = avg_bites(bites_lvl_3)
    choropeth_map_bites(bites_lvl_3)                        # creates a map showing the distribution of dog bites per zip code across all years

######################################################### BAR GRAPHS ########################################################################################

    license_df_lvl_4 = make_license_df_lvl_4("licenses_level_2.csv")
    bite_df_lvl_4 = make_bite_df_lvl_4("bites_level_2.csv")
    bite_df_lvl_4 = replace_unknown_gender(bite_df_lvl_4)
    merged_df = merge_df(license_df_lvl_4, bite_df_lvl_4)
    avg_bar_plot_group(merged_df)
    avg_bar_plot_gender(merged_df)
    merged_df.to_csv("Merged_license_and_bites_level 3.csv")

######################################################### LOGISTIC REGRESSION ###############################################################################

    merged = make_merged_df("Merged_license_and_bites_level 3.csv")
    merged = explode(merged)
    fit_linear_regression(merged)


if __name__ == "__main__":
    main()

