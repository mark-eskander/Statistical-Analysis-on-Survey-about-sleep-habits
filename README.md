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
     &emsp;-factor analysis<br/>
     &emsp;- adequacy test<br/>
     &emsp;- multible correspondance analysis<br/>
     <hr>
2-<b>Correlation anaysis (Measures of Association): <b/><br/>
   &emsp;  - Cramer's V <br/>
   &emsp;  - Chi-Squared Test<br/>
   <hr>
3- <b> Assessing Internal Consistency and Reliability : <b/><br/>
    &emsp; - Kuder-Richardson Formula 20 (KR-20) for binary  <br/>
    &emsp; - Kuder-Richardson Formula 21 (KR-20) for categorical <br/>
    <hr>
4- <b>inferential statistics:<b/><br/>
we want to determine if the mean of sleeping hours of Males and Females are the same or not  so we did :<br/>
    &emsp; - check the normality of the data of specific column [sleeping hours] for Females & Males if exists use parametric test ,else 
             use non parametric test ( in our case we used non parametric test using  Shapiro-Wilk test) <br/>
    &emsp; - check if the <b>distribution<b/> of sleeping hours of Males & Females the same or not using Kolmogorov- smirnov test <br/>
    &emsp; - to compare two sample means that come from the same population  and to test whether two sample means are equal or not , as 
             distribution of 2 independent samples is the same and the values are continous therefore we can use Mann-Whitney U test<br/>
           <hr>
5- <b> clustreing analysis: <b/> <br/>
   &emsp;  - KModes Clustering Algorithm<br/>
   &emsp;  - Hierarchical Clustering<br/>
   &emsp;  - Agglomerative Clustering <br/>
   &emsp;  - Silhouette Score <br/>
   &emsp;  - Cophenetic Correlation Coefficient<br/>
   &emsp;  - Dunn Index<br/>
    &emsp; - Calinski-Harabasz Index<br/>
    &emsp; - Agglomerative Clustering<br/>
# Visualitions 
we used  animation graphs  & fixed visualizations at preprocessing stage and statistical analysis stage


