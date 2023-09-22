#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Exploratory Analysis, generating plots and looking at linear models

import pandas as pd
import numpy as np
import datetime as dt
import statsmodels.api as sm
import matplotlib.pyplot as plt

main_data = pd.read_csv('ted-talks/ted_main.csv')
transcript_data = pd.read_csv('ted-talks/transcripts.csv')

# main_data and transcript_data also share the 'url' column. We will
# aggregate the data by joining on this column
join_data = pd.merge(main_data, transcript_data, how = "left", on = "url")

# Exploring the data, I think a good first step is to convert 'film_date' and
# 'published_date' from UNIX timestamps (type int) to readable dates.
# Conversion inspired by:
# https://www.w3resource.com/python-exercises/date-time-exercise/python-date-time-exercise-6.php
join_data['film_date'] = [dt.datetime.fromtimestamp(i).strftime('%Y-%m-%d') for i in join_data['film_date']]
join_data['published_date'] = [dt.datetime.fromtimestamp(i).strftime('%Y-%m-%d') for i in join_data['published_date']]

# For use in quantitative analysis, we may wish to use the individual year and month
# as separate columns. Since date string has standardized YYYY-MM-DD format,
# we can make separate columns from substrings of the date.
join_data['film_year'] = [str(i[:4]) for i in join_data['film_date']]
join_data['film_month'] = [str(i[5:7]) for i in join_data['film_date']]
join_data['published_year'] = [str(i[:4]) for i in join_data['published_date']]
join_data['published_month'] = [str(i[5:7]) for i in join_data['published_date']]

# Add an 'engagement' column to measure comments/views ratio. Then add a flag
# for "is_engaging" for the top 10% or so of the talks
engagement = []
for i in range(join_data.shape[0]):
    engagement.append(join_data['comments'][i] / join_data['views'][i])
join_data['engagement'] = engagement

# Add an "is_viral" flag. We can define viral viewership as a number of views,
# or a percentage of the total views
viral_viewership = np.percentile(join_data['views'], 99)
join_data['is_viral'] = np.where(join_data['views'] > viral_viewership, 1, 0)

# Maybe we want more than just two categories. Why not 10? 100? Split the 'views'
# or 'engagement' in buckets equal to max(value) / num_classes. This way we can still
# classify into the top %, but every other percent as well without the majority
# class dominating accuracy
#####################################
#####################################
num_classes = 5
#####################################
#####################################
binsize = max(join_data['engagement']) / num_classes
join_data['engagement_class'] = '000000'   # Placeholder, will add into it

for i in range(join_data.shape[0]):
    
    engagement = join_data['engagement'][i]
    
    for j in range(num_classes):
        if (engagement > binsize*(j+1) - binsize and engagement < binsize*(j+1)):
            join_data['engagement_class'][i] = str(j)
    
    # When loop reaches the last class, the max value is not actually
    # included - the bin limits stop at the max value. We will need
    # to specify explicitly that the max value belongs to the last class,
    # in this case, num_classes - 1
    if (engagement == max(join_data['engagement'])):
        join_data['engagement_class'][i] = str(num_classes - 1)



# Check out the engagement distribution
plt.figure(figsize = (10, 7))
plt.hist(join_data['engagement'], bins = 50, log = True)
plt.title('TED Engagement Distribution')
plt.xlabel("Engagement (Comments/Views)")
plt.ylabel("Count")
plt.show()

# Check out the most popular months
plt.hist([int(i) for i in join_data['published_month']])
plt.title("Distribution of TED Talks by Month")
plt.xlabel("Month")
plt.ylabel("Count")
plt.show()


# Which videos are the most engaged? Grab the top 10
plt.figure(figsize = (10, 5))
most_engaged = join_data.sort_values(by = "engagement", ascending = False).iloc[0:11,:]
plt.bar(most_engaged['name'], most_engaged['engagement'], color = 'red')
plt.title('Top 10 Highest-Engagement Videos, Engagement', weight = 'bold')
plt.xlabel('Title', weight = 'bold')
plt.ylabel('Engagement (comments/views)', weight = 'bold')
plt.xticks(rotation = 45, ha = 'right')
plt.ylim([0, 0.0025])
plt.show()

plt.figure(figsize = (10, 5))
plt.bar(most_engaged['name'], most_engaged['views'], color = 'red')
plt.title('Top 10 Highest-Engagement Videos, Views', weight = 'bold')
plt.xlabel('Title', weight = 'bold')
plt.ylabel('Views', weight = 'bold')
plt.xticks(rotation = 45, ha = 'right')
plt.ylim([0, 5e7])
plt.show()

# Interesting... top 10 most viewed?
plt.figure(figsize = (10, 5))
most_viewed = join_data.sort_values(by = "views", ascending = False).iloc[0:11,:]
plt.bar(most_viewed['name'], most_viewed['views'], color = 'green')
plt.title('Top 10 Most-Viewed Videos, Views', weight = 'bold')
plt.xlabel('Title', weight = 'bold')
plt.ylabel('Views', weight = 'bold')
plt.xticks(rotation = 45, ha = 'right')
plt.ylim([0, 5e7])
plt.show()

plt.figure(figsize = (10, 5))
plt.bar(most_viewed['name'], most_viewed['engagement'], color = 'green')
plt.title('Top 10 Most-Viewed Videos, Engagement', weight = 'bold')
plt.xlabel('Title', weight = 'bold')
plt.ylabel('Engagement (comments/views)', weight = 'bold')
plt.xticks(rotation = 45, ha = 'right')
plt.ylim([0, 0.0025])
plt.show()

# Look at the distribution of our engagement class labels. There still
# seems to be a majority class
plt.hist([int(i) for i in join_data['engagement_class']], log = True)
plt.title('TED Engagement Distribution')
plt.xlabel("Engagement Class")
plt.ylabel("Count")
plt.show()

print("Class proportions: ")
for i in range(num_classes):
    num_in_class = len(join_data[join_data['engagement_class'] == str(i)])
    total_num = join_data.shape[0]
    proportion = num_in_class / total_num
    print("Proportion of sample in engagement class", i,":", proportion)

# Curious about linear regression model

#X = join_data[['duration', 'languages', 'num_speaker', 'views']]   #Rsq = 0.326
#X = join_data[['languages', 'num_speaker', 'views']]               #Rsq = 0.298
#X = join_data[['duration', 'num_speaker', 'views']]                #Rsq = 0.296
#X = join_data[['duration', 'languages','views']]                   #Rsq = 0.326
#X = join_data[['duration', 'languages', 'num_speaker']]            #Rsq = 0.162
#X = join_data[['duration', 'views']]                               #Rsq = 0.295
#X = join_data[['languages', 'views']]                              #Rsq = 0.298
#X = join_data[['num_speaker', 'views']]                            #Rsq = 0.283
X = join_data[['views']]                                           #Rsq = 0.282
X = sm.add_constant(X) # Adds constant/intercept term
y = join_data['comments']

# y = x + mX
# Now building a model is easy. We will use Ordinary Least Squares
lr_model = sm.OLS(y, X).fit()
print(lr_model.summary())

#####


import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import matplotlib.ticker as ticker
import seaborn as sns

#Read separate data files
main = pd.read_csv('/Users/nikhiln/Desktop/Final/ted-talks/ted_main.csv')
transcript = pd.read_csv('/Users/nikhiln/Desktop/Final/ted-talks/transcripts.csv')

#Merge main and transcript data
merged = pd.merge(main, transcript)


#LinReg model to predict comments
X = main[['views']]
y = main[['comments']]

#Set X-axis in millions for views
#https://stackoverflow.com/questions/61330427/set-y-axis-in-millions
def millions(x, pos):
    'The two args are the value and tick position'
    return '%1.1fM' % (x * 1e-6)

formatter = FuncFormatter(millions)

#Plot entire dataset to visualize linear regression line
fig, ax = plt.subplots()
ax.xaxis.set_major_formatter(formatter)
plt.xlabel("Number of Views")
plt.title("Comments and Views on TED Talk Videos", fontsize=16, fontweight='bold')
plt.xlabel("Number of Views")
plt.ylabel("Number of Comments")

main_plot = plt.scatter(X, y)

sns.regplot(data=main, x='views', y='comments')

lr_model = sm.OLS(y, X).fit()
print(lr_model.summary())

#With Views as the sole predictor, the model returns an R-squared value of 0.461.
#With views, duration, languages, num_speakers, R-squared jumps to 0.535. However, the plot shows many outliers. 
#We can filter our data to learn more about the majority of the TED talk videos concentrated toward the lower-left corner of our plot.

#Filter data to examine largest concentration of videos

main_filtered = main[(main.views <= 15000000) & (main.comments <= 1500)]


#By filtering, we lose 30 outliers. Our dataset falls from 2,550 to 2,520 points. However, with views as sole predictor R-squared rises to 0.566.
#With the four predictors, the R-squared rises further to 0.65


#Apply regression and plot

X_filtered = main_filtered[['views']]
y_filtered = main_filtered[['comments']]

main_filtered_plot = plt.plot(X_filtered, y_filtered, 'o')

def millions(x, pos):
    'The two args are the value and tick position'
    return '%1.1fM' % (x * 1e-6)

formatter = FuncFormatter(millions)

fig, ax = plt.subplots()
ax.xaxis.set_major_formatter(formatter)
plt.title("Comments and Views on TED Talk Videos", fontsize=16, fontweight='bold')
plt.xlabel("Number of Views")
plt.ylabel("Number of Comments")

main_filtered_plot = plt.plot(X_filtered, y_filtered, 'o')
sns.regplot(data=main_filtered,
            x='views',
            y='comments')


filtered_lr_model = sm.OLS(y_filtered, X_filtered).fit()
print(filtered_lr_model.summary())


#We can likely increase the model's accuracy by further cutting down on outliers.
#However, this is a design choice and would require domain knowledge to best configure bounds for outliers.
#Furthermore, we do no not want to cut outliers to the extent that the model falls victim to overfitting. 

#Examine Engagement -------


import pandas as pd
import numpy as np
import time
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

main_data = pd.read_csv('ted-talks/ted_main.csv')
transcript_data = pd.read_csv('ted-talks/transcripts.csv')

# main_data and transcript_data also share the 'url' column. We will
# aggregate the data by joining on this column.
# Also note that this will result in some "NaN" values for the transcript. We
# will also eliminate those rows - this can be done by simply selecting
# all data where "transcript" does not equal NaN.
# This will delete indices which are needed for later iteration, so we must
# reset them
join_data = pd.merge(main_data, transcript_data, how = "left", on = "url")
join_data = join_data[join_data['transcript'].notna()]
join_data = join_data.reset_index(drop = True) # True removes original index column

# Add an 'engagement' column to measure comments/views ratio
engagement = []
for i in range(join_data.shape[0]):
    engagement.append(join_data['comments'][i] / join_data['views'][i])
join_data['engagement'] = engagement

# Maybe we want more than just two categories. Why not 10? 100? Split the 'views'
# or 'engagement' in buckets equal to max(value) / num_classes. This way we can still
# classify into the top %, but every other percent as well without the majority
# class dominating accuracy
#####################################
#####################################
num_classes = 5
#####################################
#####################################

binsize = max(join_data['views']) / num_classes
join_data['views_class'] = '000000'   # Placeholder, will add into it

for i in range(join_data.shape[0]):
    
    views = join_data['views'][i]
    
    for j in range(num_classes):
        if (views > binsize*(j+1) - binsize and views < binsize*(j+1)):
            join_data['views_class'][i] = str(j)
    
    # When loop reaches the last class, the max value is not actually
    # included - the bin limits stop at the max value. We will need
    # to specify explicitly that the max value belongs to the last class,
    # in this case, num_classes - 1
    if (views == max(join_data['views'])):
        join_data['views_class'][i] = str(num_classes - 1)



# We can do the same thing with engagement. Then we can compare how certain
# string variables affect viewer engagement classes with Naive Bayes:
binsize = max(join_data['engagement']) / num_classes
join_data['engagement_class'] = '000000'   # Placeholder, will add into it

for i in range(join_data.shape[0]):
    
    engagement = join_data['engagement'][i]
    
    for j in range(num_classes):
        if (engagement > binsize*(j+1) - binsize and engagement < binsize*(j+1)):
            join_data['engagement_class'][i] = str(j)
    
    # When loop reaches the last class, the max value is not actually
    # included - the bin limits stop at the max value. We will need
    # to specify explicitly that the max value belongs to the last class,
    # in this case, num_classes - 1
    if (engagement == max(join_data['engagement'])):
        join_data['engagement_class'][i] = str(num_classes - 1)


# Naive Bayes Section -----

# For use in Naive Bayes Classifier, create separate training and testing
# datasets. Randomize the dataframe and take the top 70% for training, 
# the rest for testing.
randomized_data = join_data.sample(frac = 1)
num_rows = randomized_data.shape[0]
train_proportion = 0.8
training_cutoff = int(num_rows * train_proportion)

train_df = randomized_data.iloc[0:training_cutoff+1]
test_df = randomized_data.iloc[training_cutoff+1:num_rows]

# Construct a classifier pipeline to be used in the Naive Bayes method
classifier = Pipeline([
    ('bow', CountVectorizer()), # strings to token integer counts
    ('classifier', MultinomialNB())
    ])


# Naive Bayes, supply with the string feature name for the predictor and
# string feature name for the response. Also supply the number of models
# to average an accuracy over num_models models. IMPORTANT: The strings
# must be entered with double quotes!
def naive_bayes_classifier(X_string, y_string, num_models):
    
    start = time.time()
    accuracies = []
    
    for i in range(num_models):
        X_train = train_df[X_string]
        y_train = train_df[y_string]
        X_test = test_df[X_string]
        y_test = test_df[y_string]
        
        classifier.fit(X_train, y_train)
        predictions = classifier.predict(X_test)
        accuracies.append(accuracy_score(y_test, predictions))
        
    mean_accuracy = np.mean(accuracies)
    std_accuracy = np.std(accuracies)
    end = time.time()
    process_time = end - start
    
    print("---------------------------------------------------------------")
    print("Naive Bayes averaged over", num_models, "models:")
    print('Predictor used:', X_string)
    print('Response used:', y_string)
    print("Mean accuracy: ", mean_accuracy)
    print("Standard Deviation accuracy: ", std_accuracy)
    print("Processing Time:", process_time, "seconds")
    print("---------------------------------------------------------------")
    print(" ")


    
# Now run our function while testing different predictors and responses
#naive_bayes_classifier("tags", "is_viral", 100)
#naive_bayes_classifier("title", "is_viral", 100)
#naive_bayes_classifier("description", "is_viral", 100)
#naive_bayes_classifier("transcript", "is_viral", 1)

#naive_bayes_classifier("title", "most_engaged", 10)
#naive_bayes_classifier("title", "most_engaged", 10)
#naive_bayes_classifier("description", "most_engaged", 10)
#naive_bayes_classifier("transcript", "most_engaged", 3)

naive_bayes_classifier("tags", "views_class", 10)
naive_bayes_classifier("title", "views_class", 10)
naive_bayes_classifier("description", "views_class", 10)
naive_bayes_classifier("transcript", "views_class", 3)

naive_bayes_classifier("tags", "engagement_class", 10)
naive_bayes_classifier("title", "engagement_class", 10)
naive_bayes_classifier("description", "engagement_class", 10)
naive_bayes_classifier("transcript", "engagement_class", 10)

print("Class proportions: ")
for i in range(num_classes):
    num_in_class = len(join_data[join_data['engagement_class'] == str(i)])
    total_num = join_data.shape[0]
    proportion = num_in_class / total_num
    print("Proportion of sample in engagement class", i,":", proportion)

