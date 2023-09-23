#!/usr/bin/env python
# coding: utf-8

# # Importing Libraries and Functions

# In[29]:


import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import warnings
warnings.filterwarnings("ignore")
from factor_analyzer import FactorAnalyzer
from factor_analyzer.factor_analyzer import calculate_kmo
from factor_analyzer.factor_analyzer import calculate_bartlett_sphericity
from prince import MCA
import scipy.stats as stats
from kmodes.kmodes import KModes
from sklearn.metrics import silhouette_score
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from sklearn.metrics import calinski_harabasz_score
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics.pairwise import cosine_distances
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit


# In[3]:


# Loading data (responses)
df= pd.read_excel("E:\\collage\\Semester 6\\survey\\Survey Methodology Project\\Sleep Habits (Responses).xlsx")


# In[30]:


print("The shape of the dataset is",df.shape)


# In[5]:


#print the first 5 rows of the dataset
df.head()


# # Exploratory Data Analysis (EDA) and Data Cleaning

# In[6]:


# get the column names
df.columns.tolist()


# In[7]:


# Rename the features of the data 
df = df.rename(columns={
    "What's your occupation?": 'occupation',
    'What is your marital status?': 'marital_status',
    'Do you share a bed with a partner or spouse?   ':'sharing_bed',
    'How many hours of sleep you get each night?':'sleeping_hours',
    'How long does it typically take you to fall asleep?':'time_fallingasleep',
    'How often do you experience difficulty falling or staying asleep?':'difficulty_fallingasleep',
    'Do you wake up feeling rested and refreshed most mornings?':'wakeup_rested',
    'How often do you wake up during the night?':'wakeup_night',
    'Which range of time do you prefer to sleep in? ':'sleep_time',
    'Do you have a set bedtime and wake-up time?  \n':'set_timesleep',
    ' Do you use any sleep aids (e.g., medication, herbal supplements) to help you sleep?   \n':'sleep_aids',
    ' How often do you exercise or engage in physical activity?  \n':'physical_activity',
    'How often do you consume caffeine per day ? \n"cup of caffeine "':'caffeine_perday',
    'How often do you consume caffeine before bedtime?   \n':'caffeine_B_bedtime',
    'Do you have any nighttime habits or routines that help you fall asleep?  \n':'night_routine',
    'If yes, please specify which habits or routines you have:':'which_night_routine',
    'How often do you use electronic devices (e.g., phone, tablet) in bed before sleep?  ':'using_elec_devices',
    'How comfortable is your sleep environment (e.g., temperature, noise level, lighting)?   \n':'sleep_environment',
    'How often do you nap during the day on weekly basis?   \n':'nap_times',
    'Do you feel you get enough sleep on a regular basis?   ':'enough_sleep',
    'How often do you feel focused and alert during the day?  ':'focused_duringday',
    'Have you ever been diagnosed with a sleep disorder (e.g., insomnia, sleep apnea, restless leg syndrome or other)?  \n':'sleep_disorder',                                    
    'If yes, please specify which sleep disorder(s) you have been diagnosed with\n':'which_disorder',
    'Do you have any medical conditions that affect your sleep?   ':'medical_condition',
    'If yes, please specify which medical condition(s) affect your sleep:  \n':'which_condition',
    'Do you feel that stress or anxiety affects your sleep quality?   \n':'anxiety',                                                                                             
    'Have you ever had a sleep study or evaluation?   ':'sleep study',                                                                                                             
    'If yes, please specify the reason for the sleep study or evaluation:  \n':'sleepstudy_reason',  
    'How often do you dream during sleep?   ': 'dreaming',                                                                                                                      
    'How often do you have vivid or disturbing dreams?   \n': 'disturbing_dreams',                                                                                               
    'How often do you snore during sleep?  \n':'snoring',
    'Is there anything else you would like to share about your sleep habits or experiences?\n': 'sleep_habits',
    'Have you tried any remedies or strategies to improve your sleep? If so, what has worked for you? Are there any suggestions for improving sleep quality?': 'suggestions',
    'Have you had an accident due to lack of sleep?': 'accident',    
    'Do you usually eat anything before bed?': 'eat_beforebed',
    'If so, how long before bedtime do you usually have your last meal or snack?': 'lastmeal'})


# In[8]:


# Cheking column names after editing
df.columns.tolist()


# In[9]:


# get info of the data 
df.info()


# ## Notice that almost all the data features are of type object (except for 'sleep_environment' variable which indicates the respondent's rating to his/her sleep environment).

# In[10]:


# Drop irrelevant columns (according to pre-testing and validation phases) 
df.drop(["Timestamp","What is your household income?", "Do you have any suggestions for improving sleep quality or addressing sleep-related issues?"], axis=1, inplace=True)


# ### Dealing with Null Values

# In[11]:


# check for null values
df.isnull().sum()


# In[12]:


# fill null values in specific columns with the word 'No'
# these null values represent that there are no responses to the optional questions; since the respondent doesnot have any specific routine, disorder, medical comdition, etc.
df[['which_night_routine', 'which_disorder','which_condition','sleepstudy_reason','sleep_habits','suggestions']] = df[['which_night_routine', 'which_disorder','which_condition','sleepstudy_reason','sleep_habits','suggestions']].fillna('None')


# In[13]:


#Dealing with null values by most frequent values (mode)
df['accident'].fillna(df['accident'].mode()[0], inplace=True)
df['eat_beforebed'].fillna(df['eat_beforebed'].mode()[0], inplace=True)
df['lastmeal'].fillna( df['lastmeal'].mode()[0], inplace=True)


# In[14]:


# check for null values again
df.isnull().sum()


# ### Dealing with Careless Responses

# In[15]:


# Checking 'Age' values 
# to ensure that all of the respondents (participants) are from our specified target population (aging >= 18 years)
df['Age'].value_counts()


# In[16]:


# Dropping 2 rows (responses) where the 'Age' value is 'Under 18' since  they are out of our target population
df = df.drop(df[df['Age'] == 'Under 18'].index)


# In[17]:


# Checking 'occupation' values
df['occupation'].value_counts()


# In[18]:


df['occupation'] = df['occupation'].replace('Student, Unemployed', 'Student')
df['occupation'] = df['occupation'].replace('Student, Employed', 'Student, Employed or freelancer')


# In[19]:


df['occupation'].value_counts()


# In[20]:


# Checking 'sleep_time' values
df['sleep_time'].value_counts()


# In[21]:


df['sleep_time'] = df['sleep_time'].replace(["I don't have a favorite time to sleep, I sleep at any time", "I don't have a preferred time :)"],
                                              'No specific time range')
df['sleep_time'] = df['sleep_time'].replace('3-3', '3 - 5 am')
df['sleep_time'] = df['sleep_time'].replace('7 or 8', '7-8 pm')
df['sleep_time'] = df['sleep_time'].replace('5-8', '5-8 am')

# Checking 'sleep_time' values after treating careless responses
df['sleep_time'].value_counts()


# In[22]:


# Checking 'caffeine_perday' values
df['caffeine_perday'].value_counts()


# In[23]:


df['caffeine_perday'] = df['caffeine_perday'].replace(["على حسب الزنقه 🐕😂","Depending on the time","Usually 1 cup but it's not daily","maybe 2 per week"],
                                                      'It depends')
# Checking 'caffeine_perday' values after treating careless responses
df['caffeine_perday'].value_counts()


# In[33]:


# Checking 'which_disorder' values
df['which_disorder'].value_counts()


# In[34]:


df['which_disorder'] = df['which_disorder'].replace(['None ','Non','Never been diagnosed by a professional/doctor before ',
                                                     '.','High loud & light room ','Nothing ','none','Nothin','I havnt'],
                                                      'None')
# Checking 'which_disorder' values after treating careless responses
df['which_disorder'].value_counts()


# Note that 'Depression' value is not a sleep disorder so it should be replaced by 'None' but it should be taken into consideration in 'which_condition' feature

# In[32]:


df['sleep_disorder'][df['which_disorder']=='Depression ']


# In[28]:


df['which_condition'][df['which_disorder']=='Depression ']


# In[35]:


df['which_condition'][df['which_disorder']=='Depression ']= df['which_condition'][df['which_disorder']=='Depression '].replace('Anxiety', 'Depression, Anxiety')


# In[36]:


df['which_condition'][df['which_disorder']=='Depression ']


# In[37]:


df['which_disorder'] = df['which_disorder'].replace('Depression ', 'None')

# Final check of 'which_disorder' values after treating careless responses
df['which_disorder'].value_counts()


# In[38]:


# Checking 'which_condition' values
df['which_condition'].value_counts()


# In[39]:


df['which_condition'] = df['which_condition'].replace(['None ','No medical condition ', 'Nothing ', 'No'],
                                                      'None')
df['which_condition'] = df['which_condition'].replace('صعوبة في التنفس ', 'Difficulty in breathing')
df['which_condition'] = df['which_condition'].replace('Stress?', 'Stress')

df['which_condition'] = df['which_condition'].replace('Depression, Sleeping from 15 to 20 hours a day', 'Depression')
df['which_condition'] = df['which_condition'].replace('Depression, Anxiety, ', 'Depression, Anxiety')
df['which_condition'].value_counts()


# In[40]:


df['which_condition'].value_counts()


# In[41]:


df['sleep_habits'].value_counts()


# In[42]:


df['sleep_habits'] = df['sleep_habits'].replace(['None','No','no','.','No',' No','Nope','No thankyouu i liked it','No thanx','No , GOOD LUCk🥰'
                                                ,'No, thanks',"I'd say that one doesn't have to sleep more than it should, a decent 7 hour sleep is just fine..",'Oversleeping is the mean reason of 5ebti El t2eela'
                                                ,'Anxiety and depression make me sleep more','يعم حبيبي','during exam days i rarely get any kind of sleep maximum 3 hours of sleep a day as i never sleep comfortably or in comfortable conditions at least, and also i suffer from breathing problems that effects my sleep',
                                                'No🤷‍♀️','Nope','No ','No 🥰','Going to bed late (3 am for instance) makes it more difficult falling asleep','.. ',' ','Nope ','no thanks ','No, thanks ','I dream of nightmares alot'], 'None')
df['sleep_habits'] = df['sleep_habits'].replace("Sleep habits and experiences can vary greatly from person to person. Here are some general facts and information about sleep:\n\n- The average adult needs 7-9 hours of sleep per night, although some people may require more or less.\n- Quality sleep is important for physical and mental health, as well as cognitive function.\n- Consistency in sleep schedule and creating a relaxing sleep environment can help improve sleep quality.\n- Sleep disorders such as insomnia, sleep apnea, and restless leg syndrome can interfere with sleep and require medical attention.\n- Dreams occur during the REM (rapid eye movement) stage of sleep, which usually happens several times throughout the night.\n- Certain medications, lifestyle factors (such as caffeine and alcohol intake), and medical conditions can affect sleep quality and quantity.\n- Chronic sleep deprivation can lead to negative consequences such as fatigue, difficulty concentrating, and increased risk of accidents and health problems.","None")
df['sleep_habits'] = df['sleep_habits'].replace("نقرأ اية الكرسي","Read Quran")
df['sleep_habits'] = df['sleep_habits'].replace("to sleep early ","sleep early")
df['sleep_habits'] = df['sleep_habits'].replace(["It gets affected too much by mobile addiction","not using  app like facebook ,instgram at all , that  help me , it may be a good idea to talk to a doctor."],"Take phone away")
df['sleep_habits'] = df['sleep_habits'].replace(["I can't sleep except when I'm reading a book or watching a movie","use mobile phone until sleeping "],"Reaing, or watching videos")
df['sleep_habits'] = df['sleep_habits'].replace(["to remove the cover from the foot ","I prefer to sleep in cold temperature than normal to get sleep rapidly "],"Cold sleep environment")
df['sleep_habits'] = df['sleep_habits'].replace(["Leave the phone outside the room, dim the room lighting, open the room window to allow as much oxygen as possible, and cover my head with a cotton cover.","Darkness and staying away from smart phones are factors affecting on my sleep quality"],"Darkness and taking phone away")
df['sleep_habits'] = df['sleep_habits'].replace("Stick to a consistent sleep schedule: Our bodies have a natural sleep-wake cycle, also known as the circadian rhythm, that is regulated by our internal clock. Going to bed and waking up at the same time every day helps regulate this rhythm, which can make it easier to fall asleep and wake up feeling refreshed.","consistent sleep schedule")
df['sleep_habits'] = df['sleep_habits'].replace(["I fall asleep fast and wish I dream of more beautiful dreams ","I fall asleep fast and wish I dream of more beautiful dreams"],"None")


# In[43]:


df['sleep_habits'].value_counts()


# In[44]:


counts = df['sleep_habits'].value_counts()
counts_habits = pd.DataFrame({'sleep_habits': counts.index, 'count': counts.values})
counts_habits


# In[45]:


df['suggestions'].value_counts()


# In[46]:


df['suggestions'] = df['suggestions'].replace(["No",'no','No ','.','Nope','Yes i tried',' ',"No , it's 3la Allah 😂😂",'No🤷‍♀️',"No i haven’t",'.. ','Yes , it haven’t ','Noo','no ,ana m7taga eli es3dni hehe'
                                              ,"Bo",'No🥰','Nope ','no',"No, I haven't tried anything",'Yes',"No I haven't",'no ',"No, I haven't tried anything ","Yes i tried.","No i haven’t ",
                                               "No, but this article helped me \nhttps://www.bulknutrients.com.au/blog/wellness/flexible-sleep-for-recovery-the-90-minute-sleep-cycle-method",
                                              "I haven't tried any treatment or strategic method",
                                               "Yes, there are several remedies and strategies that can help improve sleep quality. Here are some suggestions:\n\n1. Stick to a consistent sleep schedule: Go to bed and wake up at the same time every day, even on weekends. This helps regulate your body's natural sleep-wake cycle.\n\n2. Create a relaxing sleep environment: Make sure your bedroom is cool, dark, and quiet. Use comfortable bedding and pillows, and consider using white noise or earplugs if needed.\n\n3. Limit exposure to screens before bed: The blue light emitted by electronic devices can disrupt your sleep-wake cycle. Try to avoid using electronic devices for at least an hour before bedtime.\n\n4. Avoid caffeine and alcohol: Both caffeine and alcohol can interfere with sleep quality. Try to avoid consuming them in the hours leading up to bedtime.\n\n5. Exercise regularly: Regular physical activity can help improve sleep quality, but try to avoid exercising too close to bedtime.\n\n6. Practice relaxation techniques: Techniques such as deep breathing, progressive muscle relaxation, and meditation can help calm your mind and relax your body before bed.\n\n7. Consider seeking medical attention: If you are experiencing chronic sleep problems or suspect you may have a sleep disorder, it's important to seek medical attention. A healthcare professional can help diagnose and treat sleep disorders.\n\nIn terms of what has worked for me personally, sticking to a consistent sleep schedule and avoiding screens before bed have been helpful strategies. I also find that practicing deep breathing or meditation before bed helps me relax and fall asleep more easily."],"None")
df['suggestions'] = df['suggestions'].replace("No I haven't ","None")

df['suggestions'] = df['suggestions'].replace(['read Quran before sleeping ','Reading quaran makes me feel more comfortable while sleeping '],"Read Quran")

df['suggestions'] = df['suggestions'].replace(["haven't tried any remedies, however I've tried breathing excercises and meditation techniques to help me sleep and it was quite beneficial."],'breathing excercises and meditation techniques')

df['suggestions'] = df['suggestions'].replace(['sleep cycle application helped me to know if my sleep was comfort or not','Using the sleep calculator which calculate the sleep period to cycles each of 90 minutes so you wakeup more active and feel less tired it worked with me sometimes'],'sleep cycle applications')

df['suggestions'] = df['suggestions'].replace(["Getting tired all day makes me come back and sleep like a dead body",'Working out before bed '],"Working out before bed")

df['suggestions'] = df['suggestions'].replace(["Not accessing devices prior to sleep for a couple hours and not eating before sleep","Establishing a rule of at least two hours of quietness and technology usage elimination before bedtime",
                                               "Stop using facebook and instagram or any things like that because i think it’s getting thing more worse  maybe watch memes or  movies ","Keep your phone away while sleeping."],"Keep phone away")

df['suggestions'] = df['suggestions'].replace(["Sleeping at the same time everyday ","sleeping and waking up at a specific time everyday","I'm currently trying sleeping and waking up at a specific time everyday, waking up early and avoiding eating or drinking caffeine before bed time, and also meditation and reading before sleeping , and it's helping a lot. ",
                                               "To wakeup and sleep everyday at the same time","Must sleep early and wake up early","Monitoring my sleep cycle to help me know when it's better to wake up at (there are some apps where you enter the desired time to wake up at and based on it, it determines what's the best time to sleep at so that waking up doesn't interfere in the middle of the sleep cycle which's about 90 minutes , as it's preferred to wake up whether at the beginning or at the end of the cycle in order not to have any headache or feeling grumpy )"
                                              ,"I try to get 8 hours sleep to be better and I think it is a good thing "," sleep everyday in same time and wake up in same time .organize your day and not sleep too much morning . look after pray time it helped "],
                                              "sleeping and waking up at a specific time")


df['suggestions'] = df['suggestions'].replace(["Reading, drinking hot peppermint tea or warm milk before bed , breathing exercises if I’m too anxious ","Reading before going to sleep helps immensely."],"Reading before sleeping")

df['suggestions'] = df['suggestions'].replace([" drinking green tea and Going to bed early","Warm cup of milk ","Royal detox "],"Taking hot drinks or detox")

df['suggestions'] = df['suggestions'].replace(["i sing myself to sleep, it can help","Iistening to calm music before sleeping and getting enough amount of sleeping"],"Music")

df['suggestions'] = df['suggestions'].replace('I watch "Friends" and I fall asleep while watching.\nI used to watch DIY videos also.',"Watching videos")
df['suggestions'] = df['suggestions'].replace('I sometimes read the Qur’an before bed, and it helps me sleep, and I stay away from my phone as much as possible, and I stay away from anxiety and overthinking','Read Quran ,stay away from phone ,anxity, and overthinking')
df['suggestions'] = df['suggestions'].replace('Leave the phone outside the room, dim the room lighting, open the room window to allow as much oxygen as possible, and cover my head with a cotton cover. And it works sometimes ',"Take away phone,Dim the lighting,Open window,cover your head")
df['suggestions'] = df['suggestions'].replace('Having a set nighttime routine (showering, doing skincare, etc) kind of helps me sleep better ','Having a set nighttime routine (showering, doing skincare, etc)')

df['suggestions'] = df['suggestions'].replace('I tried for a while to sleep for 6 hours from 10 pm to 4 am. It was an amazing routine that made me more focused, productive, and in a good mood.\nقال النبي صلى الله عليه وسلم: "اللَّهمَّ بارِكْ لأمَّتي في بُكورِها."', "sleeping and waking up at a specific time")

df['suggestions'] = df['suggestions'].replace("Yes, there are several remedies and strategies that can help improve sleep quality. Here are some suggestions:\n\n1. Stick to a consistent sleep schedule: Go to bed and wake up at the same time every day, even on weekends. This helps regulate your body's natural sleep-wake cycle.\n\n2. Create a relaxing sleep environment: Make sure your bedroom is cool, dark, and quiet. Use comfortable bedding and pillows, and consider using white noise or earplugs if needed.\n\n3. Limit exposure to screens before bed: The blue light emitted by electronic devices can disrupt your sleep-wake cycle. Try to avoid using electronic devices for at least an hour before bedtime.\n\n4. Avoid caffeine and alcohol: Both caffeine and alcohol can interfere with sleep quality. Try to avoid consuming them in the hours leading up to bedtime.\n\n5. Exercise regularly: Regular physical activity can help improve sleep quality, but try to avoid exercising too close to bedtime.\n\n6. Practice relaxation techniques: Techniques such as deep breathing, progressive muscle relaxation, and meditation can help calm your mind and relax your body before bed.\n\n7. Consider seeking medical attention: If you are experiencing chronic sleep problems or suspect you may have a sleep disorder, it's important to seek medical attention. A healthcare professional can help diagnose and treat sleep disorders.\n\nIn terms of what has worked for me personally, sticking to a consistent sleep schedule and avoiding screens before bed have been helpful strategies. I also find that practicing deep breathing or meditation before bed helps me relax and fall asleep more easily.","None")

df['suggestions'] = df['suggestions'].replace("Alarm that dismissed with mession\nIt work good for 2 days\nThen sleep again ","None")


# In[47]:


df['suggestions'].value_counts()


# In[48]:


count = df['suggestions'].value_counts()
counts_suggestions = pd.DataFrame({'suggestions': count.index, 'count': count.values})
counts_suggestions


# In[49]:


df['which_night_routine'].value_counts()


# In[50]:


df['which_night_routine'] = df['which_night_routine'].replace(['None ','Nothing','nothing','.','I','nothing '],"None")
df['which_night_routine'] = df['which_night_routine'].replace(['Open app like facet instagram ','Mobile ','games','Use phone until sleep sometimes '
                                                               ,'Watching tiktoks and FB watch','Mobile games or scrolling','tiktok','Scroll on mobile '],'Using Mobile phone')
df['which_night_routine'] = df['which_night_routine'].replace(['Reading before bed, ','Reading athkar 😌🔝','تسبيح و اذكار المساء'],"Reading before bed")
df['which_night_routine'] = df['which_night_routine'].replace(['Watching a movie ','Watching movie or YouTube video','Watching youtube videos while falling asleep',
                                                               'Watching movies and youtube','watching youtube videos and asmr ','Watching movies ',
                                                               'Whatch certain tv series '],"Watching movies")
df['which_night_routine'] = df['which_night_routine'].replace(['Reading before bed, Reading or watching a movie']," Reading or watching a movie")
df['which_night_routine'] = df['which_night_routine'].replace(['Listen to The Holy Quran❤'],"Listen to The Holy Quran")
df['which_night_routine'] = df['which_night_routine'].replace(['listening to ASMR','Asmr'],'Listening to calming music or white noise')
df['which_night_routine'] = df['which_night_routine'].replace('Taking a warm bath or shower, ','Taking a warm bath or shower')
df['which_night_routine'] = df['which_night_routine'].replace('Pray','Praying')


# In[51]:


df['which_night_routine'].value_counts()


# In[52]:


total = df['which_night_routine'].value_counts()
counts_which_night_routine = pd.DataFrame({'which_night_routine': total.index , 'count': total.values})
counts_which_night_routine


# ## Descriptive Summary Statistics

# In[53]:


# Descriptive Summary Statistics for 'sleep_environment' numeric feature 
df['sleep_environment'].describe()


# In[54]:


# Calculating the outliers of 'sleep_environment' numeric feature using IQR rule

# Calculate the first quartile (Q1)
Q1 = df['sleep_environment'].quantile(0.25)

# Calculate the third quartile (Q3)
Q3 =df['sleep_environment'].quantile(0.75)

# Calculate the IQR
IQR = Q3 - Q1

# Calculate the lower and upper bounds for outliers
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Identify outliers using the bounds
outliers = (df['sleep_environment'] < lower_bound) | (df['sleep_environment'] > upper_bound)

# Print the outliers
print('The number of outliers = ', outliers.sum(), '\n')
print('Outlier Data:')
df[outliers]['sleep_environment']


# In[55]:


# Droping optional questions
df2= df.drop(['sleep_habits', 'suggestions','which_night_routine'], axis=1)


# In[56]:


#Encoding for the object features (1-1 Correspondence)
object_columns = df2.select_dtypes(include=['object']).columns
encoded_data=df2.copy()
for i in object_columns:
    encoded_data[i] = encoded_data[i].astype('category').cat.codes
encoded_data.head()


# In[57]:


for column in df2.columns:
    frequency_dist = encoded_data[column].value_counts()
    print(f"Frequency Distribution of {column}:")
    print(frequency_dist)
    print()


# In[58]:


columns= ['Age', 'Gender','occupation','marital_status','sharing_bed','sleeping_hours',
          'time_fallingasleep','difficulty_fallingasleep','wakeup_rested','wakeup_night','sleep_time',
          'set_timesleep','sleep_aids','physical_activity','caffeine_perday','caffeine_B_bedtime',
          'night_routine','using_elec_devices','sleep_environment','nap_times','enough_sleep',
          'focused_duringday','sleep_disorder','which_disorder','medical_condition','which_condition',
          'anxiety','sleep study','sleepstudy_reason','dreaming','disturbing_dreams','snoring',
          'accident','eat_beforebed','lastmeal']
for column in columns:
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.hist(df[column].astype(str))
    ax.set_xlabel(column)
    ax.set_ylabel('Frequency')
    ax.set_title('Histogram of {}'.format(column))
    ax.tick_params(axis='x', labelrotation=45)  # set rotation to 45 degrees
    plt.show()


# In[59]:


fig = px.histogram(df2, x='dreaming'
             ,color='Gender',color_discrete_sequence=['black','blue']
             ,text_auto=True,
                    animation_frame="Age",barmode="group",title='relation between gender and dreaming and age').update_xaxes(categoryorder="total descending")
fig.layout.updatemenus[0].buttons[0].args[1]["frame"]["duration"] = 2000
fig.show()


# In[60]:


fig = px.histogram(df2, x="wakeup_night", barmode="group", facet_col="Gender",
   color="lastmeal"
   , title="2 graphs  of how many times of waking up night according to on gender and time of last meal ")
fig.show(height=1000, width=1000)
# fig.show()


# In[61]:


df2_males=df2[df2['Gender']=='Male']
fig = px.pie(df2_males,hole=.4,names='sleeping_hours',title='percentage of sleeping hour based on males')
fig.show()


# In[62]:


df2_females=df2[df2['Gender']=='Female']
fig = px.pie(df2_females, hole=0.4,names='sleeping_hours',title='percentage of sleeping hour based on females',color_discrete_sequence=px.colors.sequential.RdBu_r)
fig.show()


# In[63]:


fig = px.histogram(df2, x="lastmeal", color="snoring", barmode="group", facet_col="Gender",
    
    title="relation between snoring and last meal based on gender")
fig.update_traces(textposition='inside')
fig.show()


# In[64]:


fig = px.histogram(df2, x="difficulty_fallingasleep", barmode="group",
  color='lastmeal',color_discrete_sequence= px.colors.sequential.Aggrnyl,
    title="relation between last meal and dificulty in sleeping ")

fig.show()


# In[65]:


fig = px.histogram(df2, x="time_fallingasleep", barmode="group",
  color='lastmeal',color_discrete_sequence= px.colors.sequential.RdBu,
    title="relation between last meal and time falling asleep ")
fig.update_layout(plot_bgcolor='grey')
fig.show()


# In[66]:


df2_males=df2[df2['Gender']=='Male']
fig = px.pie(df2_males,names='eat_beforebed',title='percentage of eating before sleep based on males',color_discrete_sequence=px.colors.sequential.Electric)
fig.show()


# In[67]:


df2_females=df2[df2['Gender']=='Female']
fig = px.pie(df2_females,names='eat_beforebed',title='percentage of eating before sleep based on females',color_discrete_sequence=px.colors.sequential.Viridis)
fig.show()


# In[68]:


fig = px.histogram(df2, x="sleeping_hours", barmode="group",
  color='wakeup_night',animation_frame='eat_beforebed',color_discrete_sequence= px.colors.sequential.Agsunset,
    title="relation between sleeping hours and eating before sleep and waking up night ")

fig.show()


# # Assessing Construct Validity

# Assessing the construct validity of the survey by examining the relationships between the survey items and other related constructs. This can be done through statistical techniques such as factor analysis, correlation analysis, or hypothesis testing to establish convergent validity and discriminant validity.

# ### Factor Analysis

# ### Adequacy Test
# Before performing factor analysis, we need to evaluate the “factorability” of our dataset. Factorability means "can we found the factors in the dataset?". There are two methods to check the factorability or sampling adequacy: Bartlett’s Test and Kaiser-Meyer-Olkin Test

# In[240]:


# Kaiser-Meyer-Olkin (KMO) Test
kmo_all,kmo_model=calculate_kmo(encoded_data)
kmo_model


# Value of KMO less than 0.6 is considered inadequate.

# In[241]:


# Bartlett’s test of sphericity
chi_square_value,p_value=calculate_bartlett_sphericity(encoded_data)
chi_square_value, p_value


# - The decimal representation of the value 8.463940998407733e-86 is a very small number close to zero.
# - In this Bartlett ’s test, the p-value is almost 0. The test was statistically significant, indicating that the observed correlation matrix is not an identity matrix.

# In[242]:


# Initialize the factor analysis object
factor_analyzer = FactorAnalyzer()

# Fit the factor analysis model on the data
factor_analyzer.fit(encoded_data)


# In[243]:


# Obtain the eigenvalues
eigenvalues, _ = factor_analyzer.get_eigenvalues()

# Plot the scree plot to visualize the eigenvalues
plt.figure(figsize=(8, 6))
plt.plot(range(1, len(eigenvalues) + 1), eigenvalues, marker='o')
plt.axhline(y=1, color='r', linestyle='--')  # Add a horizontal line at y=1
plt.xlabel('Number of Factors')
plt.ylabel('Eigenvalues')
plt.title('Scree Plot')
plt.grid(True)
plt.show()


# According to the Scree Plot, the optimal number of factors is equal to 15

# In[244]:


# Applying factor analysis
factor_analyzer = FactorAnalyzer(n_factors=15) 
factor_analyzer.fit(encoded_data)

# Calculate factor loadings
factor_loadings = factor_analyzer.loadings_
print("Factor Loadings:", factor_loadings)


# In[245]:


# Calculate communalities
communalities = factor_analyzer.get_communalities()
print("Communalities:")
communalities


# In[246]:


# Create x-axis values (indices)
x = list(range(len(communalities)))

# Plot the proportion of explained variance
plt.plot(x, communalities, marker='o')
plt.xlabel('Index')
plt.ylabel('Value')
plt.title('Communalities Visualization')
plt.show()


# In[247]:


# Access eigenvalues and explained variance
eigenvalues = factor_analyzer.get_eigenvalues()
explained_variance = factor_analyzer.get_factor_variance()
print("Eigenvalues:")
eigenvalues


# In[69]:


# Create x-axis values (indices)
x = list(range(len(eigenvalues)))

# Plot the eigen values
plt.plot(x, eigenvalues, marker='o')
plt.xlabel('Index')
plt.ylabel('Value')
plt.title('Eigen Values Visualization')
plt.show()


# In[70]:


print("Explained Variance:")
explained_variance


# In[71]:


# Create x-axis values (indices)
x = list(range(len(explained_variance)))

# Plot the proportion of explained variance
plt.plot(x, explained_variance, marker='o')
plt.xlabel('Index')
plt.ylabel('Value')
plt.title('Explained Variance Visualization')
plt.show()


# In[178]:


x


# ### Multiple Correspondance Analysis

# In[72]:


# Initialize the MCA object and fit the model
mca = MCA(n_components=len(df2.columns))
mca = mca.fit(df2)

# Extract the eigenvalues
eigenvalues = mca.eigenvalues_

# Calculate the proportion of variance explained by each component
explained_var_ratio = eigenvalues / np.sum(eigenvalues)

# Plot the scree plot
plt.figure(figsize=(8, 6))
plt.plot(range(1, len(explained_var_ratio) + 1), explained_var_ratio, marker='o')
plt.xlabel('Number of Factors')
plt.ylabel('Proportion of Variance Explained')
plt.title('Scree Plot')
plt.grid(True)
plt.show()


# In[73]:


# Initialize the MCA object and fit the model
mca = MCA(n_components=len(encoded_data.columns))
mca = mca.fit(encoded_data)

# Extract the eigenvalues
eigenvalues = mca.eigenvalues_

# Calculate the proportion of variance explained by each component
explained_var_ratio = eigenvalues / np.sum(eigenvalues)

# Plot the scree plot
plt.figure(figsize=(8, 6))
plt.plot(range(1, len(explained_var_ratio) + 1), explained_var_ratio, marker='o')
plt.xlabel('Number of Factors')
plt.ylabel('Proportion of Variance Explained')
plt.title('Scree Plot')
plt.grid(True)
plt.show()


# - Notice that it is the same curve, but different Scale of 'Proportion of Variance Explained'; 'encoded_data' gives better scale, so it will be the choice to apply the model on.
# - According to the Scree Plot, the optimal number of factors is equal to 5

# In[74]:


# Initialize the MCA object and fit the model
mca = MCA(n_components=5)
mca = mca.fit(encoded_data)

# Get the factor scores and loadings

loadings = mca.row_coordinates(encoded_data)
eigenvalues = mca.eigenvalues_


# In[75]:


print("Eigen Values:", eigenvalues)


# In[76]:


#Calculate the proportion of variance explained by each component
explained_var_ratio = eigenvalues / np.sum(eigenvalues)
print("The Proportion of Variance Explained:",explained_var_ratio)


# In[77]:


# Create x-axis values (indices)
x = list(range(len(explained_var_ratio)))

# Plot the proportion of explained variance
plt.plot(x, explained_var_ratio, marker='o')
plt.xlabel('Index')
plt.ylabel('Value')
plt.title('The Proportion of Explained Variance Visualization')
plt.show()


# In[78]:


print("New Coordinates (Loadings):")
loadings


# In[79]:


# Create a larger figure size
fig, ax = plt.subplots(figsize=(10, 6))

# Plot the factor loadings
loadings.plot(kind='line', ax=ax)

# Modify the plot
ax.set_title('Line Plot')
ax.set_xlabel('Rows')
ax.set_ylabel('Loadings')

plt.show()


# In[80]:


sns.set(style='whitegrid')
plt.figure(figsize=(10, 5))

# Define a list of colors for the factors
colors = ['b', 'g', 'r', 'c', 'm']

# Iterate over the factor scores and colors
for i in range(loadings.shape[1]):
    sns.barplot(x=loadings.columns, y=loadings.values[i], color=colors[i])

plt.xlabel('Variable')
plt.ylabel('Factor Loadings')
plt.title('Factor Loadings Plot')

# Create a legend with factor colors
legend_labels = ['Factor 1', 'Factor 2', 'Factor 3', 'Factor 4', 'Factor 5']
legend_handles = [plt.Rectangle((0, 0), 1, 1, color=colors[i]) for i in range(len(legend_labels))]
plt.legend(legend_handles, legend_labels)

plt.show()


# # Correlation Analysis (Measures of Association)

# ### Cramer's V and Chi-Squared Test
# Cramer's V is a measure of association between categorical variables. It ranges from 0 to 1, where 0 indicates no association and 1 indicates a strong association. 

# In[81]:


# Calculate Cramer's V for each pair of columns
correlation_matrix = pd.DataFrame(index=df2.columns, columns=df2.columns)

for col1 in df2.columns:
    for col2 in df2.columns:
        crosstab = pd.crosstab(df2[col1], df2[col2])
        chi2, p, dof, expected = stats.chi2_contingency(crosstab)
        n = crosstab.sum().sum()
        phi_c = chi2 / n
        cramers_v = (phi_c / min(crosstab.shape[0] - 1, crosstab.shape[1] - 1)) ** 0.5
        correlation_matrix.loc[col1, col2] = cramers_v


# In[82]:


# Print the correlation matrix
correlation_matrix


# In[83]:


correlation_matrix = correlation_matrix.astype(float)

# Create a heatmap using seaborn
plt.figure(figsize=(20, 20))
sns.heatmap(correlation_matrix, cmap='coolwarm', annot=True, fmt=".2f", square=True)
plt.title('Correlation Matrix Heatmap')
plt.show()


# ### Chi-Squared Test

# - H0 : The categorical variables have no relationship (independent)
# - H1 : There is a relationship (dependency) between categorical variables

# In[84]:


# Print the results
print("Chi-Squared Statistic:", chi2)
print("P-value:", p)
print("Degrees of Freedom:", dof)
print("Expected Frequencies:\n", expected)


# Based on the given output, the large chi-squared statistic, very small p-value, and the presence of expected frequencies that deviate from the observed frequencies, it indicates strong evidence to reject the null hypothesis of independence between the variables. Therefore, there is likely a significant relationship or association between the variables being tested.

# # Assessing Internal Consistency and Reliability
# - For assessing internal consistency of a categorical dataset, One measure is the Kuder-Richardson Formula 20 (KR-20) for binary (yes/no) items or the Kuder-Richardson Formula 21 (KR-21) for categorical items with multiple response options.
# - These measures are similar to Cronbach's Alpha but are suitable for categorical variables.

# In[85]:


# Define a function to get binary and categorical columns separately.

def get_binary_categorical_columns(data):
    
    """
    Get binary and categorical columns.
    
    Parameters:
    - data (DataFrame): The input DataFrame.
    
    Returns:
    - binary_cols (list): List of column names with binary categorical values.
    """
    binary_cols = []
    categorical_cols= []
    
    for column in data.select_dtypes(include=['object']):
        
        unique_values = data[column].nunique()
        if unique_values == 2:
            binary_cols.append(column)
        else:
            categorical_cols.append(column)
    
    return binary_cols, categorical_cols


# In[86]:


# Get binary categorical columns
binary_categorical_columns = get_binary_categorical_columns(df2)

#binary_categorical_columns
binary_cols = binary_categorical_columns[0]
categorical_cols = binary_categorical_columns[1]

# binary data
binary_data = encoded_data[binary_cols]

# categorical data
categorical_data = encoded_data[categorical_cols]


# In[87]:


# The Kuder-Richardson Formula 20 (KR-20)
# Compute the item-total correlations
item_total_corr = binary_data.corr().sum() / (len(binary_data) - 1)

# Compute the variance of the total scores
total_score_var = binary_data.sum(axis=1).var()

# Compute the KR-20 coefficient
kr20 = (len(binary_data) / (len(binary_data) - 1)) * (1 - (item_total_corr.sum() / total_score_var))

print('KR-20:', kr20)


#  The KR-20 coefficient suggests that there is a strong relationship and consistency among the binary items in your dataset, indicating that they are measuring a similar construct or attribute. This high level of reliability suggests that the items are reliable and consistent in their measurement.

# In[88]:


# The Kuder-Richardson Formula 21 (KR-21)
# Convert categorical data into binary data using dummy variables
binary_data = pd.get_dummies(categorical_data)

# Compute the item-total correlations
item_total_corr = binary_data.corr().sum() / (len(binary_data) - 1)

# Compute the variance of the total scores
total_score_var = binary_data.sum(axis=1).var()

# Compute the KR-21 coefficient
kr21 = (len(binary_data) / (len(binary_data) - 1)) * (1 - ((item_total_corr.sum() - item_total_corr.shape[0]) / (total_score_var - binary_data.shape[0])))

print('KR-21:', kr21)


# The KR-21 coefficient suggests that there is a relatively high level of internal consistency among the items in your test. This indicates that the items in your questionnaire, with their multiple response options, tend to measure a similar construct or dimension consistently.

# In[89]:


# Columns with high correlation (association)
df3= df2.drop(['sleep_disorder', 'medical_condition', 'sleep study'], axis=1)


# In[90]:


df3.shape


# In[91]:


#Encoding for the object features (1-1 Correspondence)
object_columns3 = df3.select_dtypes(include=['object']).columns
encoded_data3=df3.copy()
for i in object_columns3:
    encoded_data3[i] = encoded_data3[i].astype('category').cat.codes
encoded_data3.head()


# # Inferential Statistics 

# In[92]:


test_df=df3[['Gender','sleeping_hours','sleep_environment']]
test_df


# In[93]:


def randNumbers(hours):
    # Set the bounds of the range 
    if hours=='7-9 hours':
        lower_bound = 7
        upper_bound = 9
    elif hours=='4-6 hours':
        lower_bound=4
        upper_bound=6
    elif hours=='more than 9 hours':
        lower_bound=9
        upper_bound=13
    else:
        lower_bound=1
        upper_bound=3     
    # Set the step value
    step = 0.5

    # Calculate the number of steps in the range
    num_steps = int((upper_bound - lower_bound) / step)

    # Generate a random integer between 0 and num_steps
    random_step = random.randint(0, num_steps)

    # Calculate the random float
    random_number = lower_bound + (random_step * step)

    # Print the random number
    return random_number


# In[94]:


random.seed(42)
len(test_df)
for i in range(0,len(test_df)):
    hour=randNumbers(test_df.iloc[i,1])
    hour=str(hour)
    test_df.iloc[i,1]=test_df.iloc[i,1].replace( test_df.iloc[i,1],hour)
test_df    


# In[95]:


test_df['sleeping_hours']=test_df['sleeping_hours'].astype(float)


# First we need to check whether the data in that column follows a normal distribution or not
# as if it doesn't, then we can't perform t-testing (parametric statistical tests) and then we can perform non-parametric statistical tests 

# In[96]:


from scipy import stats
# Test for normality using the Shapiro-Wilk test
stat, p = stats.shapiro(test_df['sleeping_hours'])
print("Shapiro-Wilk test:")
print(f"Statistic: {stat}")
print(f"P-value: {p}")


# Since p-value is less than the significance level then we reject the null hypothesis which states that the data is normal.
# Therefore, data is not normally distributed so we'll use non-parametric tests 

# In[97]:


female_hours=test_df.loc[test_df['Gender']=='Female','sleeping_hours']
male_hours=test_df.loc[test_df['Gender']=='Male','sleeping_hours']


# In[98]:


plt.hist(female_hours)
plt.show()


# In[99]:


plt.hist(male_hours)
plt.show()


# We're not sure that the distributions of both of them are the same so we'll use Kolmogorov- smirnov test to make sure of it.
# H0: The distributions of the female sleeping hours is the same as that of the males.
# Ha: The distribution of the female sleeping hours differ than that of the males.

# In[100]:


from scipy.stats import ks_2samp
# Perform the KS test
statistic, pvalue = ks_2samp(female_hours, male_hours)

# Print the test statistic and p-value
print('KS statistic:', statistic)
print('P-value:', pvalue)


# We noticed that the p-value is greater than the significance level so we can't reject the null hypothesis. Thus , both of them have the same distribution

# Since the distribution of the 2 independent samples is the same and the values are continous therefore we can use Mann-Whitney U test,which is a non-parametric test that is used to compare two sample means that come from the same population, and used to test whether two sample means are equal or not. 

# In[ ]:





# ### The Mann-Whitney U Test
# 
# - The null hypothesis (HO) for the Mann-Whitney U test is no significant difference between the distributions of groups being compared. In other words, the mean two groups are equal. 
# - The alternative hypothesis (Ha) for the Mann-Whitney U test there is a significant difference between the distributions of two groups being compared. In other words, the mean the two groups are not equal.

# In[101]:


# We'll use Mann-Whitney U test to test whether the mean of the sleeping hours of females is equal to that of the males or not
# Perform the Mann-Whitney U test
stat, p = stats.mannwhitneyu(female_hours, male_hours)
print("Mann-Whitney U test:")
print(f"Statistic: {stat}")
print(f"P-value: {p}")


# Since significance level (alpha) by default is equal to 0.05 therefore p-value is smaller than alpha which shows that the mean of the sleeping hours of the females is not equal to that of the males.

# Then, we'd like to perform statistical test also on the "sleep_environment" to see whether males and females have the same sleeping environment or not (through the median).
# we'll use non-parametric test as the values in this column range from 1 till 5, hence it doesn't follow the normal distribution.

# In[102]:


female_env=test_df.loc[test_df['Gender']=='Female','sleep_environment']
male_env=test_df.loc[test_df['Gender']=='Male','sleep_environment']


# In[103]:


plt.hist(female_env)
plt.show()


# In[104]:


plt.hist(male_env)
plt.show()


# We'll use Man Whitney U test as its assumptions are met to see whether the medians of the samples are equal or not.
# if the medians are not the same, it indicates that one group tends to have higher or lower values than the other group.

# In[105]:


stat2, p2 = stats.mannwhitneyu(female_env, male_env)
print("Mann-Whitney U test:")
print(f"Statistic: {stat2}")
print(f"P-value: {p2}")


# P-value is greater than alpha therefore we can't reject the null hypothesis.Thus, the medians of the sleeping environment of the females and males are the same showing that there is no much great difference in the sleeping environment of females and males

# # Clustering Analysis

# ### KModes Clustering Algorithm

# In[106]:


# Set the seed value
np.random.seed(123)

# Determine the optimal number of clusters using the elbow method
cost=[]

# Iterate over a range of values for k (from 1 to 10) and calculate the cost for each number of clusters. 
# The cost represents the sum of distances between each sample and its cluster centroid.
for num_clusters in range(1, 11):
    kmode=KModes(n_clusters=num_clusters,init='Huang',n_init=5,verbose=0)
    kmode.fit_predict(encoded_data3)
    cost.append(kmode.cost_)


# In[107]:


# Plot the cost against the number of clusters
plt.plot(range(1, 11), cost, marker='o')
plt.axhline(y=cost[4], color='r', linestyle='--') 
plt.title('Elbow Method')
plt.xlabel('Number of Clusters')
plt.ylabel('Cost')
plt.grid(True)
plt.show()


# In[108]:


np.random.seed(123)
kmode=KModes(n_clusters=5, init='Huang', n_init=5, verbose=0)
cluster_labels=kmode.fit_predict(encoded_data3)
cluster_labels


# In[109]:


kmode.cost_


# In[110]:


# Calculate the silhouette score
silhouette_avg = silhouette_score(encoded_data3, cluster_labels)
print("Silhouette Score:", silhouette_avg)


# The silhouette score is used to evaluate the compactness and separation of clusters. The silhouette score ranges from -1 to 1, with higher values indicating better clustering. 

# In[111]:


# Clusters Centroids
cluster_centroids = pd.DataFrame(kmode.cluster_centroids_)
cluster_centroids.columns = df3.columns
cluster_centroids


# In[112]:


# Reset the index of the DataFrame
df3.reset_index(drop=True, inplace=True)
data_clustered = pd.concat([df3, pd.DataFrame({'Cluster': cluster_labels})], axis=1)


# In[113]:


data_clustered.shape


# In[114]:


data_clustered.head()


# In[115]:


data_clustered.isnull().sum()


# In[116]:


data_clustered['Cluster']=data_clustered['Cluster'].astype(int)
data_clustered.info()


# In[117]:


for i in range(5):
    print('Cluster {}:'.format(i))
    print(data_clustered[data_clustered['Cluster'] == i].iloc[:, :-1].head())


# In[118]:


fig = px.histogram(x=data_clustered["Cluster"].astype(str), labels={'x':'Clusters'})
fig.show()


# In[119]:


# Plot the clusters
fig, ax = plt.subplots(figsize=(10, 6))
for i in range(5):
    ax.scatter(data_clustered.iloc[cluster_labels == i, 0], data_clustered.iloc[cluster_labels == i, 5], label=f'Cluster {i+1}')
ax.set_xlabel('Age')
ax.set_ylabel('Sleeping Hours')
ax.set_title('K-Modes Clustering')

# Move the legend outside the plot
legend = ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

# Adjust the plot layout to make room for the legend
plt.tight_layout(rect=[0, 0, 1, 1])

plt.show()


# ### Hierarchical Clustering

# In[120]:


#to calculate the distance between all of the mapped features 
distances = pdist((encoded_data3.iloc[:,:]).values, metric='jaccard') 
#to have it in a matrix form instead of the upper triangle only
dist_matrix = squareform(distances)
print(dist_matrix)


# In[121]:


linkage_data = linkage(dist_matrix, method='complete', metric='euclidean')
""""to make sure that all of the elements in this matrix are of type float64 
we use this function to cast them all into float64 for drawing the dendrogram
"""
linkage_data=linkage_data.astype('float64')


# In[122]:


# First method to choose the optimal number of clusters (k)
fig, ax = plt.subplots(figsize=(30,15))
dendrogram(linkage_data)
plt.axhline(y=3.2, color='b', linestyle='-')
plt.show()


# In[123]:


labels=fcluster(linkage_data,3,criterion='maxclust')


# In[124]:


silhouette_avgPp=silhouette_score(dist_matrix,labels)
silhouette_avgPp


# In[159]:


# By taking a random sample from 'encoded_data3' dataset 
# of size equals to 175 observations (75% of the size of the original dataset)
random.seed(1234)
sample = encoded_data3.sample(175)


# In[160]:


distances2 = pdist(sample.values, metric='jaccard')

#to have it in a matrix form instead of the upper triangle only
dist_matrix2 = squareform(distances2)
print(dist_matrix2)


# In[161]:


linkage_data2 = linkage(dist_matrix2, method='complete', metric='euclidean')
""""to make sure that all of the elements in this matrix are of type float64 
we use this function to cast them all into float64 for drawing the dendrogram
"""
linkage_data2=linkage_data2.astype('float64')


# In[162]:


fig, ax = plt.subplots(figsize=(30,15))
dendrogram(linkage_data2)
plt.axhline(y=3,color='b',linestyle='-')
plt.show()
#this shows that the optimal number of clusters is 3


# In[163]:


#assigning cluster labels to data using linkage matrix
k=3
labels2=fcluster(linkage_data2,k,criterion='maxclust')


# In[164]:


silhouette_avg2=silhouette_score(dist_matrix2, labels2)
silhouette_avg2


# In[165]:


sample['cluster_label']=labels2
sample


# In[166]:


cluster_means=sample.groupby('cluster_label').mean()
cluster_means


# In[167]:


fig,ax =plt.subplots(figsize=(25,12))
im=ax.imshow(cluster_means, cmap='YlGnBu', aspect='auto')
ax.set_xticks(np.arange(len(sample.columns)))
ax.set_xticklabels(sample.columns)
ax.set_yticks(np.arange(k))
ax.set_yticklabels(['Cluster {}'.format(i) for i in range(1, k+1)])
ax.grid(False)
plt.colorbar(im)
plt.show()


# In[168]:


# Another method to find the optimal number of clusters (according to Silhouette Score)
# For the whole dataset
# Calculate silhouette scores for different numbers of clusters

max_clusters = 10  # Maximum number of clusters to consider
silhouette_scores = []

for n_clusters in range(2, max_clusters+1):
    labels = fcluster(linkage_data, n_clusters, criterion='maxclust')
    silhouette_scores.append(silhouette_score(dist_matrix, labels))
    
# Find the optimal number of clusters based on silhouette scores
optimal_clusters = silhouette_scores.index(max(silhouette_scores)) + 2  # +2 because range started from 2
print("Silhouette Scores: ",silhouette_scores, "\n")
print("Optimal number of clusters:", optimal_clusters)


# In[169]:


labels_final =fcluster(linkage_data, optimal_clusters, criterion='maxclust')
# optimal_clusters = 2


# #### Silhouette Score:
# 
# The silhouette score measures the average distance between each sample and samples in its own cluster compared to samples in other clusters. Higher silhouette scores indicate well-separated clusters, with values ranging from -1 to 1.

# In[170]:


silhouette_score(dist_matrix,labels_final)


# #### Cophenetic Correlation Coefficient (CC):
# 
# The cophenetic correlation coefficient measures the correlation between the pairwise distances of original data points and the distances obtained from hierarchical clustering. A higher value close to 1 indicates a better clustering solution.

# In[137]:


# Calculate the Cophenetic Correlation Coefficient (CC)
coph_corr_coeff = dist_matrix.ravel().tolist()
coph_corr = linkage_data[:, 2].tolist()
cc = pd.Series(coph_corr).corr(pd.Series(coph_corr_coeff))

print("Cophenetic Correlation Coefficient:", cc)


# #### Dunn Index:
# 
# The Dunn index calculates the ratio of the minimum inter-cluster distance to the maximum intra-cluster distance. A higher Dunn index indicates better clustering, with larger inter-cluster distances and smaller intra-cluster distances.

# In[138]:


# Defining a function that finds the minimum inter-cluster and the maximum intra-cluster distances
# then calculates the Dunn Index
def dunn_index(labels, distance_matrix):
    min_inter_cluster = float('inf')
    max_intra_cluster = 0.0

    for i in range(len(labels) - 1):
        for j in range(i + 1, len(labels)):
            if labels[i] != labels[j]:
                inter_cluster = distance_matrix[i, j]
                if inter_cluster < min_inter_cluster:
                    min_inter_cluster = inter_cluster
            else:
                intra_cluster = distance_matrix[i, j]
                if intra_cluster > max_intra_cluster:
                    max_intra_cluster = intra_cluster

    dunn = min_inter_cluster / max_intra_cluster
    return dunn

# calling the function
dunn = dunn_index(labels_final, dist_matrix)
print("Dunn Index:", dunn)


# #### Calinski-Harabasz Index:
# 
# The Calinski-Harabasz index evaluates the ratio of between-cluster dispersion to within-cluster dispersion. Higher index values indicate better-defined clusters.

# In[139]:


# Calculate the Calinski-Harabasz Index
calinski_harabasz = calinski_harabasz_score(dist_matrix, labels_final)
print("Calinski-Harabasz Index:", calinski_harabasz)


# In[171]:


# For the sample
# Calculate silhouette scores for different numbers of clusters
max_clusters = 10  # Maximum number of clusters to consider
silhouette_scores = []

for n_clusters in range(2, max_clusters+1):
    labelss = fcluster(linkage_data2, n_clusters, criterion='maxclust')
    silhouette_scores.append(silhouette_score(dist_matrix2, labelss))
    
# Find the optimal number of clusters based on silhouette scores
optimal_clusters2 = silhouette_scores.index(max(silhouette_scores)) + 2  # +2 because range started from 2
print("Silhouette Scores: ",silhouette_scores, "\n")
print("Optimal number of clusters:", optimal_clusters2)


# In[172]:


labelss = fcluster(linkage_data2, optimal_clusters2, criterion='maxclust')
# optimal_clusters2 = 2


# In[173]:


silhouette_score(dist_matrix2,labelss)


# In[174]:


# Calculate the Cophenetic Correlation Coefficient (CC)
coph_corr_coeff = dist_matrix2.ravel().tolist()
coph_corr = linkage_data2[:, 2].tolist()
cc = pd.Series(coph_corr).corr(pd.Series(coph_corr_coeff))

print("Cophenetic Correlation Coefficient:", cc)


# In[175]:


dunn = dunn_index(labelss, dist_matrix2)
print("Dunn Index:", dunn)


# In[176]:


# Calculate the Calinski-Harabasz Index
calinski_harabasz = calinski_harabasz_score(dist_matrix2, labelss)
print("Calinski-Harabasz Index:", calinski_harabasz)


# In[177]:


# appending clusters' labels (those of n_clusters = 2) to the original dataset
encoded_data3['cluster_label']=labels_final
encoded_data3


# ### Agglomerative Clustering

# In[147]:


#Casting this column to be of type string as we're going to perform one-hot encoding on the whole datatset to use cosine distance
df3.iloc[:,14]=df3.iloc[:,14].astype(str)


# In[148]:


encoder=OneHotEncoder(handle_unknown='ignore')
encoded_data4=encoder.fit_transform(df3)
dis_matrix=cosine_distances(encoded_data4)
dis_matrix


# In[149]:


pairwise_clustering3 = AgglomerativeClustering(affinity='precomputed',linkage='complete').fit(dis_matrix)
pairwise_clustering3


# In[150]:


silhouette_score(dis_matrix, pairwise_clustering3.labels_)


# In[151]:


#Creating a dataframe of the one-hot encoded data to deal with it instead of the sparse matrix (encoded_data4)
encoded_df=pd.DataFrame(encoded_data4.toarray(),columns=encoder.get_feature_names(df3.columns))
encoded_df


# In[152]:


encoded_df['cluster']=pairwise_clustering3.labels_
cluster_means3=encoded_df.groupby('cluster').mean()
cluster_means3


# In[153]:


fig,ax =plt.subplots(figsize=(25,12))
im=ax.imshow(cluster_means3, cmap='YlGnBu', aspect='auto')
ax.set_xticks(np.arange(len(encoded_df.columns)))
ax.set_xticklabels(encoded_df.columns)
ax.set_yticks(np.arange(2))
ax.set_yticklabels(['Cluster {}'.format(i) for i in range(1, 3)])
ax.grid(False)
plt.colorbar(im)
plt.show()


# # Visualizations 

# In[154]:


#Related to the clusters produced from the first hierarchical method 
"""Transforming data from wide format to long format:
In wide format, each row represents a single observation and each column represents a different
variable while in Long format, each row represents a single observation and each column represents
a different variable value"""

cluster_means_melted = pd.melt(cluster_means.reset_index(),id_vars=['cluster_label'], var_name='variable', value_name='mean')
cluster_means_melted


# In[155]:


# Plot the cluster profile plot that shows the average value of each variable for each cluster
fig,ax =plt.subplots(figsize=(14,10))
sns.set_style('whitegrid')
sns.set_palette('colorblind')
sns.barplot(x='variable', y='mean', hue='cluster_label', data=cluster_means_melted, palette='Set2')
plt.xticks(rotation=90)
plt.xlabel('Variable')
plt.ylabel('Mean')
plt.show()


# In[156]:


subset_df = df3[['Gender','Age','sleeping_hours', 'nap_times','physical_activity','disturbing_dreams','eat_beforebed','lastmeal']]
fig = px.parallel_categories(subset_df)
fig


# In[157]:


split=StratifiedShuffleSplit(n_splits=1,test_size=0.3,random_state=42)
for train_index, test_index in split.split(encoded_data3,encoded_data3['cluster_label']):
    train_data=encoded_data3.iloc[train_index]
    test_data=encoded_data3.iloc[test_index]

print(f"Train set size: {len(train_data)}")
print(f"Test set size: {len(test_data)}")


# In[158]:


fig=px.parallel_coordinates(train_data[['Age', 'Gender', 'occupation', 'physical_activity','enough_sleep','cluster_label']],color='cluster_label',width=1400,height=600)
fig.show()
#we notice that the lines are not parallel to each other which shows that the variables are not independent of each other

