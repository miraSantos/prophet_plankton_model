import pandas as pd


dfbiovol = pd.read_csv("/dos/MIT-WHOI/github_repos/preliminary_data/data/mvco/biovol_concentration_by_class_time_series_CNN_daily10Jan2022.csv")#download locally hosted data (consider adding the rest api)
dfbiovol.rename(columns = {'milliliters_analyzed':'ml_analyzed'},inplace=True) #rename sample time column to datetime
dfbiovol["datetime"] = pd.to_datetime(dfbiovol["datetime"].str[:-9],format='%d-%b-%Y')#get rid of extra characters off the edge of the script and convert to pandas datetime format

print(dfbiovol.shape)
dfbiovol.head()

# converting from wide to long
dflong = dfbiovol.melt(id_vars = ['datetime','ml_analyzed'],var_name = "species",value_name="biovol")#converting from wide to long format, using datetime and ml_analyzed as id values
dflong["datetime"] = pd.to_datetime(dflong["datetime"],format='%d-%b-%Y')#reconverting to date column to datetime format
print(dflong.head())

clist = pd.read_csv('/dos/MIT-WHOI/github_repos/preliminary_data/data/IFCB_classlist_type.csv') #importing classlist type maps species to functional groups e.g. diatoms, dinoflagellates
clist.long = pd.melt(clist,id_vars = ['CNN_classlist'], var_name = 'group',value_name='present') #converting to long version
clist.long = clist.long.rename(columns = {"CNN_classlist":'species','in_out':'present'}) #renaming columns to species and presense
clist.long = clist.long[clist.long['present']==1] #keeping only rows that have 1 in the present column (other rows are uneccesary)
print("class list long")
print(clist.long.head())

dflong = dflong.join(clist.long.set_index('species'),on = 'species') #joining classlist with the dataframe of biovolumes according to the species

dflong.head()
dfgrouped = dflong.groupby(['datetime','group']).agg({'biovol':'sum','ml_analyzed':'sum'}).reset_index() #grouping by functional group and summing up biovolume and ml analyzed, reset_index turns into back into normal pandas dataframe


dfdiatom = dflong[dflong.group == "Diatom"].groupby("datetime").agg({'biovol':"sum"})
plt.scatter(dfdiatom.index,dfdiatom.biovol)

dfdiatom.reset_index(inplace=True)
dfdiatom.rename(columns={"biovol": "diatomBiovol","datetime":"date"},inplace=True)

dfdiatom["doy_numeric"] = dfdiatom["date"].dt.dayofyear  # extracting day of year

print(dfdiatom.head())

dfdiatom.to_csv("/home/mira/PycharmProjects/prophet_plankton_model/datasets/diatoms_conc.csv")
