
# coding: utf-8

#     Data Story on the IMDB Dataset

# First we explore the dataset and clean it to make it usable for analysis

# In[1]:

#import packages


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from IPython.core.display import HTML
from scipy.stats import stats
import seaborn as sns
sns.set(color_codes=True)
from datetime import datetime
css = open('style-table.css').read() + open('style-notebook.css').read()
HTML('<style>{}</style>'.format(css))
get_ipython().magic('matplotlib inline')
get_ipython().magic('pylab inline')


# In[2]:

import bokeh.plotting as bkp
from mpl_toolkits.axes_grid1 import make_axes_locatable


# Load the data

# In[3]:

data = pd.read_csv("movie_metadata.csv")
data.head(5)


# In[4]:

#Keep relevent columns from the table
data = data[['movie_imdb_link', 'imdb_score', 'movie_title', 'title_year', 'duration', 'language', 'country', 'genres', 'director_name', 'director_facebook_likes', 'movie_facebook_likes', 'actor_1_name', 'actor_2_name', 'actor_3_name', 'actor_1_facebook_likes', 'actor_2_facebook_likes', 'actor_3_facebook_likes', 'cast_total_facebook_likes', 'num_critic_for_reviews', 'num_voted_users', 'num_user_for_reviews', 'aspect_ratio', 'facenumber_in_poster', 'plot_keywords','content_rating']]


# In[5]:

len(data)


# In[7]:

#check info
data.info() 


# Check for missing values

# In[8]:

data.isnull().values.any()


# In[9]:

#Check how many values are null in each column
data[data.columns[:]].isnull().sum()


# In[10]:

prop_missing = round((data[data.columns[:]].isnull().sum()/data[data.columns[:]].count())*100,2)
prop_missing


# In[11]:

#Remove the missing data with title year missing
clean_data = data[data.title_year.notnull() & data.duration.notnull()]
len(clean_data)


# In[12]:

clean_data.loc[:, 'title_year'] = clean_data['title_year'].astype(int).astype(str)
clean_data.loc[:, 'year'] = pd.to_datetime(clean_data['title_year'], format='%Y')


# In[13]:

clean_data.describe()


# Q1. What are the total number of movies reviewed by decade?

# In[14]:

temp = clean_data[['year', 'title_year','movie_title', 'imdb_score']]
#solutio
temp1 = temp[['title_year', 'movie_title']]
temp1.groupby(temp1.title_year.astype(int) // 10 * 10).size().plot(kind='bar')
plt.tight_layout()
#need help on adding labels to below


# This shows a growing trend of movies created every decade. The amount of movies created is growing exponentially. The last decade data is only available for 4 years (2010-2014) so it obviously shows a drop in movies created in the last decade

# Q2. What has been the trend of imdb rating?

# In[15]:

#years = mdates.YearLocator()
temp2 = clean_data[['title_year', 'imdb_score']]
temp2 = temp2.groupby(temp2.title_year.astype(int)).imdb_score.mean().plot(kind ='line', title ='IMDB Mean score trend line', xlim=((1950, 2016)))
temp2.xaxis.set_ticks(np.arange(1950, 2016, 7))
plt.tight_layout()
#fig.suptitle('test title', fontsize=20)
plt.xlabel('year', fontsize=12)
plt.ylabel('mean score', fontsize=12)
#add plot title


# It looks like the average score of movies is decreasing over time, However, this could be due to an increase in no of movies being created over time

# Below plot shows comparison of movies created vs average imdb score over time

# In[16]:

#create new table with grouped information
temp = clean_data[['title_year', 'imdb_score', 'movie_imdb_link']]
temp = temp[temp.title_year.astype(int)>1949]
res = temp.groupby(temp.title_year.astype(int)).agg({'imdb_score': 'mean', 'movie_imdb_link': 'count'}).reset_index()
res.columns = ['title_year', 'avg_imdb_score', 'movies_created']
rows = res.title_year

fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.plot(res.title_year, res.avg_imdb_score, color = 'red')
ax1.set_ylabel('avg_imdb_score', color = 'red')

ax2 = ax1.twinx()
ax2.plot(res.title_year, res.movies_created, color='green')
ax2.set_ylabel('movies_created', color='green')
for tl in ax2.get_yticklabels():
    tl.set_color('r')
plt.show()

#plt.savefig('images/two-scales-5.png')


# As we can see, the decrease in average movie scores can be attributed to the increase in amount of movies created in recent times, the increase in amount of movies will lead to more outliers affecting the mean for the duration.

# Q3. How do the score change for each type of content rating? (Are the content ratings tied to the imdb scores?)
# 

# In[17]:

temp = data[['content_rating', 'imdb_score']]
temp.groupby(temp.content_rating).imdb_score.mean().plot(kind='bar')
plt.show()


# Overall, TV-MA content has highest score but there isnt much difference in score between content

# Q4. Which director has the highest IMDB average score? Display the top 10
# 

# In[15]:

#len(data['director_name'])
temp = data[['director_name', 'imdb_score']]
t1 = temp.groupby(temp.director_name).mean().sort_values(by='imdb_score', ascending=[False])
t1.head()


# In[17]:

t1 = t1.head(10).plot(kind='barh', title='Top 10 high scoring directors', legend=[True])
# set labels for both axes
t1 = t1.set( xlabel='Average IMDB Score', ylabel='Director')
plt.show()


# Q5. Plot the relation between the variables movie facebook likes and imdb_score using a scatterplot
# 

# In[40]:

#scatterplot average_imdb_score vs movie_facebook_likes


# In[41]:

#plot
temp = clean_data[['movie_facebook_likes', 'imdb_score']]
temp = temp[temp.imdb_score > 0]
x = temp.plot(x='movie_facebook_likes', y = 'imdb_score',kind='scatter', xlim = (0, 100000), title='IMDB Score VS Movie facebook likes', legend=[True])


# In[42]:

print( temp[['imdb_score','movie_facebook_likes']].corr())


# These is weak positive correlation between imdb score and movie facebook likes

# Q6. Plot average imdb score vs durartion of movie

# In[32]:

temp = clean_data[['duration', 'imdb_score']]
temp = temp.plot('duration', 'imdb_score', kind ='scatter', title ='Duration VS Mean IMDB Score')
plt.xlabel('imdb_score', fontsize=12)
plt.ylabel('duration', fontsize=12)
plt.tight_layout()

#add plot title


# In[33]:

# np.correlate(clean_data["imdb_score"], clean_data["duration"])
print( clean_data[['imdb_score','duration']].corr())


# There is weak positive correlation between imdb_score and duration of movie

# In[34]:

temp = clean_data[['duration', 'imdb_score']]
sns.regplot(x="duration", y="imdb_score", data=temp);


# This shows that there is weak positive correlation between imdb_score and duration, we can use this to find an optimum range of duration that gives the best scores.

# Q7. Check if movies with language 'hindi' have more duration on average than with language 'english'

# In[43]:

temp = clean_data[['language', 'duration', 'title_year']]
temp1 = temp[temp.title_year.astype(int) >= 2000]
temp1 = temp1.loc[temp1['language'].isin(['English','Hindi'])]
# temp1 = temp1[temp1.language == 'English' | temp1.language == 'Hindi']
temp1.groupby(temp1.language).duration.mean().plot(kind='bar')
plt.show()


# In[36]:

hindi = temp1[temp1.language == 'Hindi']
english = temp1[temp1.language == 'English'] 


# In[37]:

print("The dataset has {} Hindi and {} English movies".format(len(hindi), len(english)) )


# As we can see from the above plot, Hindi movies are longer than English movies, however the Hindi movie dataset only has 26 movies while the English movie dataset has 3308 movies (all realeased since year 2000). We need more data for Hindi movies in order to draw a comparison. Below we plot the histogram for these two datasets:

# In[38]:

#plot histogram for hindi movie durations
hindi.duration.plot(kind='hist',color='0.5', bins = 10, title = 'Histogram for duration of Hindi movies').set_xlabel('Duration')
hindi_mean = round(mean(hindi["duration"]),2)
hindi_sd = round((hindi["duration"]).std(),2)
print("The mean duration of the Hindi movies is {} and standard deviation is {}".format(hindi_mean, hindi_sd) )


# In[39]:

#plot histogram for english movie durations
english.duration.plot(kind='hist',color='0.5', bins = 10, title = 'Histogram for duration of English movies').set_xlabel('duration')
english_mean = round(mean(english["duration"]),2)
english_sd = round((english["duration"]).std(),2)
print("The mean duration of the English movies is {} and standard duration is {}".format(english_mean, english_sd) )


# Based on these plots we can analyze the hypothesis that mean duration of Hindi movies and English movies are same, we can use the t test to test
# $H_0: \mu_h=\mu_e $ VS $H_0: \mu_h != \mu_e$ at alpha = 0.05

# Q. Having made these plots, what are some insights you get from them? Do you see any correlations? Is there a hypothesis you would like to investigate further? What other questions do they lead you to ask?
# 
# A. 1) We see that the amount of movies created over time is increasing exponentially while the average IMDB score for the movies is decreasing over time
# 2) We see a positive correlation between IMDB Score and Movie facebook likes and between Duration of movie and its IMDB Score
# 3) We see a difference in mean duration of movies created since 2000 between groups of Hindi and English movies, this leads of us formulate a Hypothesis test to test if the durations are similar between the groups or not
# 

# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:



