#!/usr/bin/env python
# coding: utf-8

# The research question that is being proposed by this analysis is "Is there a significant difference in the types of play that end in touchdowns versus those that don't?". The dataset that is being analyzed here is a dataset that details every single play of the 2022 NFL season. There are a variety of features in this dataset with the most important being 'Down','ToGo','YardLine','Yards','Formation','PlayType','IsRush','IsPass','IsIncomplete','PassType','IsSack' and 'RushDirection'. These variables allow us to check all of the important statistics about each play and compare the plays that result in touchdowns against those that do not. In this analysis we will also be comparing how data can be filtered and cleaned using pandas versus using spark. We will begin our analysis with pandas analysis.

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[545]:


df=pd.read_csv(r"C:\Users\Owner\Downloads\PBPData.csv",sep=',')


# In[546]:


df['Yards']=pd.to_numeric(df['Yards'])


# In[547]:


df1=df[df['IsFumble']==0] 
df1=df1[df1['IsInterception']==0]
df1=df1[df1['IsNoPlay']==0]
df1=df1[df1['IsPenaltyAccepted']==0]
df1=df1[df1['IsTouchdown']==1]
df2=df[df['IsTouchdown']==0]
df2=df2[df2['IsFumble']==0] 
df2=df2[df2['IsInterception']==0]
df2=df2[df2['IsNoPlay']==0]
df2=df2[df2['IsPenaltyAccepted']==0]


# The above text box filters our dataset by making sure that each play in df1 is a touchdown and each play in df2 is not a touchdown. Thus, we can examine how each of the other variables changes when a play works and is a touchdown, versus when a play is not a touchdown. The lines of code in the above box also filter the dataset to not have any fumbles, interceptions, dead plays, and accepted penalties. This makes sure that the touchdown plays were by the offense and not off of turnovers or negated by penalties.

# In[548]:


touchdown=df1[['Down','ToGo','YardLine','Yards','Formation','PlayType','IsRush','IsPass','IsIncomplete','PassType','IsSack','RushDirection']]
alldata=df[['Down','ToGo','YardLine','Yards','Formation','PlayType','IsRush','IsPass','IsIncomplete','PassType','IsSack','RushDirection']]
notouchdown=df2[['Down','ToGo','YardLine','Yards','Formation','PlayType','IsRush','IsPass','IsIncomplete','PassType','IsSack','RushDirection']]


# The above text box filters the dataframes by making each of them only contain the columns that are specified. These columns are the only that have been deemed necessary for this analysis.

# In[550]:


x1=list(df[df['IsTouchdown']==1]['Yards'])
x2=list(df[df['IsTouchdown']==0]['Yards'])
plt.hist([x1,x2], bins=int(180/15),density=True,color=colors,label=names)
plt.title("Density of Yards for Touchdown vs No Touchdown")
plt.xlabel("Yards")
plt.ylabel("Density of Yard Amounts ")


# The above text box plots a histogram with the yards gained on a play being blue for a touchdown and yellow for being not a touchdown. The highest density of yards gained is around 0 for both touchdown AND non-touchdown plays. The figures plotted below show the rush direction percentages for each play that resulted in a touchdown and not a touchdown respectively.

# In[551]:


touchdown.groupby('RushDirection').size().plot(kind='pie', autopct='%.2f',title='Rush Direction Breakdown'


# In[552]:


notouchdown.groupby('RushDirection').size().plot(kind='pie', autopct='%.2f',title='Rush Direction Breakdown')


# These two plots only have one major difference, which is that plays that result in a touchdown have a higher percentage of runs up the middle than those that do not result in a touchdown. This is because whenever a team gets within the 0-10 yard lines of the field, they often run the ball up the middle in order to secure easy points in the game.

# In[554]:


notouchdown.groupby('Formation').size().plot(kind='pie', autopct='%.2f',title='Formation Breakdown')


# In[555]:


touchdown.groupby('Formation').size().plot(kind='pie', autopct='%.2f',title='Formation Breakdown')


# The above two plots major difference is that the shotgun percentage for touchdown plays is much higher than non-touchdown plays. This is likely because touchdown plays are often the result of passing plays, with passing plays being often run out of shotgun.

# We will now move into using Spark for this dataset's analysis. Overall, using a pandas dataframe to analyze this data was quite easy, and there were not many drawbacks that I could see to using it even when using such a large dataset.

# In[556]:


from pyspark.sql import SparkSession
spark = SparkSession.builder.appName('ml-bank').getOrCreate()


# In[557]:


sparkdf = spark.read.csv(r"C:\Users\Owner\Downloads\PBPData.csv",sep=',', header = True, inferSchema = True)


# In[558]:


sparkdf.printSchema()


# The above lines create our spark session, import our csv as a spark dataframe, and then print its "schema" which states each of the variables along with their data type.

# In[559]:


sparkdf.crosstab('IsTouchdown', 'Formation').show()


# In[560]:


sparkdf.crosstab('IsTouchdown', 'RushDirection').show()


# The above charts show the breakdown of the the "formation" and "rushdirection" variables using crosstab. This function allows us to see how these variables are distributed among whether the play was a touchdown or not. I feel that this is a much more digestible way of taking in this information in comparison to when the pandas dataframe was used. Instead of having to make four pie charts, the variables relationships to touchdowns can instead be shown in just two charts.

# In[561]:


df1=sparkdf[sparkdf['IsFumble']==0] 
df1=df1[df1['IsInterception']==0]
df1=df1[df1['IsNoPlay']==0]
df1=df1[df1['IsPenaltyAccepted']==0]
tsparkdf=df1[df1['IsTouchdown']==1]


# The above cell filters our spark dataframe to not include the same thing that the previous dataframe included either. In order to examine how running the ball up the middle effects how many yards gained and how many yards were needed, the dataframe was filtered below.

# In[562]:


tsparkdf=tsparkdf[['Down','ToGo','YardLine','Yards','Formation','PlayType','IsRush','IsPass','IsIncomplete','PassType','IsSack','RushDirection']]
sparkdf=sparkdf[['Down','ToGo','YardLine','Yards','Formation','PlayType','IsRush','IsPass','IsIncomplete','PassType','IsSack','RushDirection']]
tsparkdf.filter(sparkdf.RushDirection=='CENTER').show(40)


# This filter shows how running up the middle most often results in gains of 0-5 yards, and they usually occur when the team has less than 5 yards needed for another first down. 

# In[563]:


tsparkdf.orderBy(tsparkdf.Yards.asc()).show(10)


# In[564]:


sparkdf.orderBy(sparkdf.Yards.asc()).show(10)


# When looking at these two charts, it is obvious that the first of them is the chart that has only touchdowns on it, as there are no negative plays that result in touchdowns in the chart. Hence, when looking at the second chart we can see that there are multiple plays that lose over 15 yards, which would be impossible to do on an offensive touchdown play.

# Overall, there is a lot of interesting differences between plays with touchdowns and plays without touchdowns, and both Spark and Pandas dataframes are able to show this. The plays with touchdowns had generally higher rush percentage up the middle and higher percentage of shogun passes than normal plays. However, these facts are not likely to impact how football teams operate, as runs up the middle result in touchdowns so often because they are often used when a team only needs 1-2 yards, and the shotgun formations are usually passes and thus touchdowns come easier. The spark dataframe filtering does not have many overarching conclusions, as many of the comparisons are within the graphs, with one being that obviously plays that score touchdowns on average gain more yards than the typical play overall. I would say from my work in this class that filtering data with Spark is much easier than in Pandas, but graphing is much easier when using Pandas. Thus, when working with massive amounts of data, it feels as though Spark would be generally the better option as filtering the data is one of the most important parts of data analysis.

# In[ ]:




