# EVALUATE CNN EFFICIENCY BASED ON NUMBER OF DETECTIONS AND RATIO

# basic import
import numpy as np
import pandas as pd
from pandas import Series,  DataFrame

# plotting modules and libraries
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

#set ggplot style
plt.style.use('ggplot')

#read dataframe
dframe = pd.read_csv('test_df.csv')
print(dframe)

#delete mask column
del dframe['Unnamed: 0']
print(dframe.columns)

#sort by image name
df = dframe.sort_values(by=['Img_Name'])
df = df.reset_index(drop=True)
#print(df)

#group by pictures subplots
fig, ax = plt.subplots(figsize=(14,7))
df_group = df.groupby(['Img_Name','Log']).mean()['Scores'].unstack().reset_index()
print(df_group)


plt.xticks(np.arange(len(df_group)),df_group['Img_Name'], rotation=0)
ax.set_xlabel('Images')
ax.set_ylabel('Scores')
df_group.plot(ax=ax,legend=True)
plt.show()


# #plot with reset index!
# df_group = df.groupby(['Log','Img_Name'])[['Scores']].mean().reset_index()
# df_group.plot(ax=ax,legend=True)
# print(df_group)
#
# plt.xticks(np.arange(len(df_group)),df_group['Img_Name'],rotation='vertical')
# ax.set_xlabel('Img_Name')
# ax.set_ylabel('Scores')
# plt.show()

