---
layout: post
title: Can Streaming More Increase Your Follower Count? 
subtitle: Lamda School Unit 1 Portfolio Project
cover-img: /assets/img/Increase_Sales.jpg
thumbnail-img: /assets/img/Increase_Sales.jpg
share-img: /assets/img/Increase_Sales.jpg
tags: [Lambda-School, portfolio]
---

# One Way to Increase Follower Count

One common tip that streamers get when trying to gain more Twitch followers is to stream a lot and frequently. Some sources* suggest streaming 5 hours a day is a faster way to gain followers. With data of the top 1000 Twitch streamers in the world being available on Kaggle, I wanted to check to see if there truly is a relationship between streaming more frequently and an increase in follower count.  

---

## Examining the Data

The data provided by Kaggle gives us basic data from the top 100 Twitch streamers, updated as recently as of February 2020. As the image below shows, we get data ranging from their channel name to what language they use.

<img src="/assets/img/Twitch_Data_Intro.png" width="2000" height="200">

Here is a key to see exactly what each column header(variable) is:
- **Channel:** The streamer's channel name
- **Watch time(Minutes):** The total amount of time that the streamer has been watched on stream (in minutes)
- **Stream time(minutes):** The total amount of time the streamer streamed to their audience (in minutes)
- **Peak viewers:** The max amount of viewers they had at once during a stream
- **Average viewers:** The average amount of viewers they had during a stream
- **Followers:** The total amount of followers they currently have (as of February 2021)
- **Followers gained** The amount of followers gained during February 2020 - February 2021
- **Views gained:** The amount of views gained during February 2020 - February 2021
- **Partnered:** If they are an official Twitch partner
- **Mature:** If the streamer has voluntarily (usually) set their channel to mature, meaning they tell users that there could be profanity, etc
- **Language:** What language the streamer mainly streams in  

In order to work with the data, I cleaned the data by replacing spaces in the column headers with underscores. This is to prevent any bugs within my code when creating linear regression models down the line. I also feature engineered new variables, _"Stream_time_hr"_ and _"Stream_time_days"_, in order to make reading the stream time data more digestible. The variables that were most essential to the analysis was _"Stream_time_hr"_ and _"Followers gained"_.

## So Now What?

After I cleaned up my data, I started to think of what the best methods were to prove that there was a relationship between streaming often and gaining followers. I decided to take a step-by-step process, using the more simplistic methods first, and then expanding to more elaborative methods at the end. This would hopefully allow me to craft a better picture of the relationship between the variables while also showcasing multiple methods in the process. As a teaser, the methods that I used in order were:
- 2-independent-sample t-test
- Simple linear regression model
- Logistic regression model
- Multiple regression model

I will go into more detail of each method as we proceed.

## Step 1: Using a 2-Independent-Sample T-Test

The first method, and perhaps the most simplest method I used was a **2-independent-sample t-test**. This method checks to see if two population means(averages) are equal. Or specific to this case, it checks to see if the mean of followers gained for _streamers who stream a lot_ is equal to the mean of followers gained for _streamers who dont stream a lot_. This is called the null hypothesis. There is also an alternate hypothesis where it says the population means aren't equal. We test to see if we can reject or fail to reject the null hypothesis or accept the alternate hypothesis. You can find the null and alternate hypothesis below:

- H0:μ1=μ2

- Ha:μ1≠μ2

However in order to conduct this test, we would need to classify how many hours is considered a lot for streamers. Using the baseline provided by Lifewire of 5 hours a day, we can estimate that _streamers who stream a lot_ probably stream 5 hours a day, or more than 1303.57 hours a year. So I feature engineered another variable _"Streams_a_lot"_, which assigned a _0_ to streamers who didn't stream more than 1303.57 hours a year and a _1_ to streamers who streamed more than 1303.57 hours a year. 

I then conducted the t-test using the variables _"Streams_a_lot"_ and _"Followers_gained"_ to see if there is a relationship betwen them. Using Python, I calculated the **P-value to be 1.28e-07**, which was below the standard significance level of 0.05. Therefore we could reject the null hypothesis and conclude that there is indeed a relationship between streaming a lot and gaining followers.

## Step 2: Diving Deeper with a Simple Linear Regression Model   

One of the limits of using a T-Test only, is that it only tells us that there is a relationship between two variables. If we wanted to learn how statistically related two variables are we could use a **Simple Linear Regression Model**. This model compares the linear relationship between two continous variables. So in this model, we would use the quantitative variable _"Stream_time_hr"_ as our independent variable, instead of _"Streams_a_lot"_ since _"Streams_a_lot"_ is a categorical variable. Our dependent variable would still be _"Followers_gained"_. And our null hypothesis will be slightly different as although it still tests that the two variables are unrelated, it also mathmatically tests that the slope between these two variables are equal to zero.

- Ho:  β1=0
- Ha:  β1≠0

In order to see the simple linear regression model, I plotted our two variables against each other in Python

![Simple Linear Regression Model](/assets/img/Simple_linear_reg_graph_port1.png)

Just by looking at the graph, I could see that there is a negative correlation between the two variables. However since the axis for _"Followers_gained"_ is in the millions, it hard to say how strong the correlation, but it still looks a litte bit on the weaker side. We could also see a bit of outliers in the graph, which could skew the correlation to be more negative. The data points also seem pretty spread, however since there are 1,000 data points, it could be deceptively more linear than it looks.

Now that we know how the model looks like, we could use Pyhton to calculate the P-value and R-Coefficient(linear correlation coefficient). The R coefficent measures the strength and direction of the linear relationship between two quantitative variables.

```python
from scipy import stats

r_st_fg, p_val_st_fg = stats.pearsonr(df['Stream_time_hr'], df['Followers_gained'])

print('R Coefficient: ', r_st_fg)
print('P-Value: ', p_val_st_fg)
```
The resulting code shows us that the P-value is 4.98e-07 and the R coefficent to be -0.158. So since this P-value is less than 0.05, we would reject our null hypothesis that there is no relationship between the two variables. This confirm what we already know from the t-test before. Additionally the R coefficent shows us that there is a weak, negative correlation between the variables. This confirms what we saw in our graph bit it also shows contrary of what Lifewire said about the relationship between streaming time and gaining followers. Instead of seeing an increase of followers with streaming time, we see a decrease of followers with streaming time. However, there are still a few more models to execute before we make a conclusion.

We could also use Python use the ols model from statsmodels to find us the details of the linear regression

![Simple Linear Regression OLS](/assets/img/Simple_linear_reg_model.png)

The main components to look at in this model are
- _Intercept_: This shows the Y-intercep of the graph
- _Stream_time_hr_: This shows the slope of the graph. For every one unit increase in _Stream_time_hr_, _Followers_gained_ decreases by -37.78
- _R-Squared_: the percent of the variability in the y variable that is explained by differences in the x variable 

The most important component to get from this model is _R-Squared_. It basically says that 2.5% of the variability seen in _Followers_gained_ is explained by _Stream_time_hr_. This value is pretty low and shows that there are either a better model to fit the data, or there are other factors affecting the data, whether it be outliers or other data points that we aren't looking at. This confirms that we should make changes in our model to find a more fitting model.

## Lets Remove Some Outliers

---



