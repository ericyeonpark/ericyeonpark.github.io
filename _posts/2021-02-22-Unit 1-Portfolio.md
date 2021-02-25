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

![data image](/assets/img/Twitch_Data_Intro.png | width=100)

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

## Step 1: Using a 2-independent-sample t-test

The first method, and perhaps the most simplest method I used was a **2-independent-sample t-test**. This method checks to see if two population means(averages) are equal. Or specific to this case, it checks to see if the mean of followers gained for _streamers who stream a lot_ is equal to the mean of followers gained for _streamers who dont stream a lot_. This is called the null hypothesis. There is also an alternate hypothesis where it says the population means aren't equal. We test to see if we can reject or fail to reject the null hypothesis or accept the alternate hypothesis. You can find the null and alternate hypothesis below:

$H_0: \mu_1 = \mu_2$

$H_a: \mu_1 \neq \mu_2$

However in order to conduct this test, we would need to classify how many hours is considered a lot for streamers. Using the baseline provided by Lifewire of 5 hours a day, we can estimate that _streamers who stream a lot_ probably stream 5 hours a day, or more than 1303.57 hours a year. So I feature engineered another variable _"Streams_a_lot"_, which assigned a _0_ to streamers who didn't stream more than 1303.57 hours a year and a _1_ to streamers who streamed more than 1303.57 hours a year. 

I then conducted the t-test using the variables _"Streams_a_lot"_ and _"Followers_gained"_ to see if there is a relationship betwen them. Using Python, I calculated the **P-value to be 1.28e-07**, which was below the standard significance level of 0.05. Therefore we could reject the null hypothesis and conclude that there is indeed a relationship between streaming a lot and gaining followers.




---




