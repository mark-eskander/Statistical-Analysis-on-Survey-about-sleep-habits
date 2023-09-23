# Statistical-Analysis-on-Survey-about-sleep-habits
we conducted a survey about sleep habits among people and their habits before sleeping ,demographic data, data about health...,etc <br/>and we did some statistical tests ,model validation , pre-test,<i>internal consistency<i/> <br/>
# Data collection
we collected responses from people through <b>Google forms<b/> 
# REQUIRMENTS
knowledge about :<br/>
-Data preprocessing & Cleaning <br/>
-Statistical  Background [inferential & descriptive]<br/>
-Hypothesis test<br/>
-clustring techniques<br/>
-construct validity <br/>
-internal consistency and Reliability<br/>

# PREPROCESSING Stage
1- rename the features of data as it was quesions<br/>
2- handle null values wrong responses<br/>
3- put constrains on the age of the response and drop who is out of the constrain ( who is under 18 years old )<br/>
4- mapping & transform values<br/>
5- dealing with careless responses<br/>
6- drop optional questions <br/>
7- drop irrelevant questions  according to <b><i> PRETEST & VALIDATION PHASE <i/> <b/><br/>
8- Calculate outliers<br/>
# STATISTICAL ANALYSIS stage
1- <b>Assessing construct validty :<b/><br/>
     -factor analysis<br/>
     - adequacy test<br/>
     - multible correspondance analysis<br/>
2-<b>Correlation anaysis (Measures of Association): <b/><br/>
     - Cramer's V <br/>
     - Chi-Squared Test<br/>
3- <b> Assessing Internal Consistency and Reliability : <b/>
     - Kuder-Richardson Formula 20 (KR-20) for binary  <br/>
     - Kuder-Richardson Formula 21 (KR-20) for categorical <br/>
4- <b>inferential statistics:<b/>
we want to determine if the mean of sleeping hours of Males and Females are the same or not  so we did :<br/>
     - check the normality of the data of specific column [sleeping hours] for Females & Males if exists use parametric test ,else use 
       non parametric test ( in our case we used non parametric test using <b><i> Shapiro-Wilk test<b/><i>) <br/>
     
     -  check if the <b>distribution<b/> of sleeping hours of Males & Females the same or not using <b><i> Kolmogorov- smirnov test <b/><i/><br/>
     -  to compare two sample means that come from the same population  and to test whether two sample means are equal or not , as <u>distribution of 2 independent samples is the same<u/> and the <u> values are continous<u/> therefore we can use <i><b>Mann-Whitney U test<b/><i/><br/>
5- <b> clustreing analysis: <b/>
     - KModes Clustering Algorithm<br/>
     - Hierarchical Clustering<br/>
     - Agglomerative Clustering <br/>
     - Silhouette Score <br/>
     - Cophenetic Correlation Coefficient<br/>
     - Dunn Index<br/>
     - Calinski-Harabasz Index<br/>
     - Agglomerative Clustering<br/>
# Visualitions
we used  animation graphs  & fixed visualizations at preprocessing stage and statistical analysis stage


