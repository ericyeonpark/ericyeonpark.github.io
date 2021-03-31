---
layout: post
title: How to Make a Movie that Gets Featured on Netflix
subtitle: Lamda School Unit 2 Portfolio Project
cover-img: /assets/img/port2/Netflix_Cover.jpeg
thumbnail-img: /assets/img/port2/Netflix_Thumbnail.jpeg
share-img: /assets/img/port2/Netflix_Thumbnail.jpeg
tags: [Lambda-School, portfolio]
---

# How to Get On Netflix

Are you an aspring film maker? Are you intrigued by algorithms? Are you curious what aspects that companies and consumers look for in a successful movie. Then you're in the right place. Today, I'm going to dive in to what features of a movie make it most likely to be featured on Netflix. In order to figure this out, I conducted a binary classification model using features I found on Kaggle(data updated as of May 2020). And what are these features. You can see exactly what they are below:

- **Netflix:** Whether the movie is found on Netflix
- **ID:** Arbitrary ID number for each movie
- **Title:** Title of the movie
- **Year:** The year in which the movie was produced
- **IMDb:** IMDb rating of the movie
- **Rotten Tomatoes:** Rotten Tomatoes % rating of the movie
- **Hulu:** Whether the movie is found on Hulu
- **Prime Video** Whether the movie is found on Amazon Prime Video
- **Disney+** Whether the movie is found on Disney+
- **Type:** Is it movie or a TV series
- **Directors:** Director of the movie
- **Genres:** Genre of the movie
- **Country:** Country origin of the movie
- **Language:** Primary language spoken in the movie
- **Runtime:** The runtime of the movie

---

# Proofreading the Script

With 17,000 observations, one could imagine the amount of cleaning I had to do just to prepare the data for my model. So I took a nice and easy systematic approach to skimming some of the fat of my data.

#### Dropping High Null Count Features
My first step was to look at the null counts for all the features. To my dismay, I found that some of the more interesting features had high null counts. Both the _Rotten Tomatoes_ (~11,500 null values) and _Age_(~9390 null values) had over half of their observations as null and had to be removed.

#### Dropping High-Cardinality and Low-Cardinality Features
Next I looked into which features had high-cardinality (had many unique values) and found 4 features that made the cut. _Title_, _Directors_, and _ID_. I wasn't too sad about dropping these features as in terms of the greater picture of finding what movies/shows get on Netflix, these features would have little importance for prediction in our model. However, unfortuntely, I found that _Type_ seemed to be a broken feature in that in only had one value throughout (value = 0) and therefore had to be dropped since it would give no value to our model.

#### Dropping Irrelevant Features to Prevent Data Leakage
When I first created my model, I found that it was performing at absurd levels (>94% accuracy) and I had wondered why my model was overfitting. After taking a closer look at the features, I realized that some of my features were most likely causing data leakage and would need to be removed. I decided to remove _Hulu_, _Prime Video_, and _Disney+_ as it occured to me that whether a movie/show was on Hulu, Amazon Prime, or Disney+ logically should have no bearing on whether a movie/show goes onto Netflix. It was also likely that a lot of movies/shows overlapped with Netflix, so there data leakage that made my model overfit. However, I think it would be interesting to see what features matter most for the other streaming platforms and compare and contrast them in a future project.

Now that I dropped a lot of the fat, the features that would be incorporated in my model are the following:
- **Netflix:** Whether the movie is found on Netflix
- **Year:** The year in which the movie was produced
- **IMDb:** IMDb rating of the movie
- **Genres:** Genre of the movie
- **Country:** Country origin of the movie
- **Language:** Languages spoken in the movie
- **Runtime:** The runtime of the movie

# Expand the Interesting Scenarios
While exploring the data, I found that for three of my features, that some of their observations had multiple values that could be seperated into new variables. For example, in the _Language_ column, one observation could be 'English, Japanese, Spanish', which shows what languages were spoken in the film. 

![Clean Dataframe](/assets/img/port2/cleaned_data_frame.png)

However, in order to tune our data so that our binary classification model performs better, it is better to convert these observations with multiple languages so that each language receives its own value. If you compare the image above to the image below, you can see now that the original _Language_ column has been deleted, and replaced by each of the language values that it previously contained. So now, for the previous observation of 'English, Japanese, Spanish', there would be an int value of '1' for the _English_, _Japanese_, and _Spanish_ columns. I applied the same process for the _Genres_ and _Country_ columns as well, which expanded our dataframe to 201 columns(which still works since we still had 16,744 observations)

![Expanded Dataframe](/assets/img/port2/expanded_data_frame.png)

# Hire the Crew to Get the Job Done

Now that we have our script written, it's time to hire the producers, actors, and crew to create the movie. In our case, its finally time to build our model. Since our goal is to see what features help a movie get on Netflix, our target feature is _Netflix_. After using our target feature to create our train, val, and test sets using a train_test_split, I wanted to use a random forest classification model and XGBoost model and see which model would create the best predictions and interesting feature importances. In order to check to see which model would be viable, I created a baseline accuracy (using the normalizrd max value of our y_train set) that showed that the models had to beat a baseline accuracy of 78.9&. I would also look at other metrics of the models such as ROC AUC, precision, and AUC scores to compare the efficacies of the models.

#### Random Forest Model

I first created a Random Forest model using randomized search and grid search methods to tune my hyper-parameters.
```python
#Random Forest Model
model = make_pipeline(OrdinalEncoder(),
                      SimpleImputer(strategy = 'median'),
                      StandardScaler(),
                      RandomForestClassifier(random_state=42,
                                             max_depth = 60,
                                             n_estimators = 90,
                                             class_weight = 'balanced_subsample')
                      )
                                       
model.fit(X_train, y_train)
```
One hyper-parameter that I want to highlight is class_weight, which helps balance our unbalanced target _Netflix_. We previously saw in our baseline that the "0" value was far more heavily weighted (~78.9% of our data showed movies that weren't on Netflix). Using the class_weight hyperparameter helps balance those weights so that we could increase our recall and precision values for our models down the line.

#### XGBoost Model

I also created an XGBoost Model and tuned the hyperparameters with randomized search methods. One different hyper-parameter is learning_rate, where a higher value helps increase the model fit(range is from 0 to 1)

```python
#XGBoost Model
model_xgb = make_pipeline(OrdinalEncoder(),
                          SimpleImputer(strategy = 'median'),
                          XGBClassifier(random_state = 42,
                                        n_estimators=20,
                                        max_depth = 70,
                                        learning_rate= 0.00010526315789473685,
                                        n_jobs=-1))

model_xgb.fit(X_train, y_train)
```

# Host a Screening of the Movie and Recieve Feedback

Now that we directed and produced the movie, it's now time to hold a pre-release screening and see the initial reaction to our movie. Or in more technical terms, we fit our models and see what metrics we get from each model.

#### Accuracy of the Models
The first metric that we checked for both models is accuracy. We checked the accuracy of the Random Forest and XGBoost models for our training, validation, and test sets.

- Baseline Accuarcy : .789

- Random Forest Training Accuracy: 0.997
- Random Forest Validation Accuracy: 0.825
- Random Forest Testing Accuracy: 0.824

- XGBoost Accuracy Train:  0.894
- XGBoost Accuracy Val:  0.793
- XGBoost Accuracy Test:  0.807

We can see that both our model outperform our baseline which is a good sign. The main categories that concern ourselves with is how the models perform within the validation and testing sets, so I'm not too concerned that the Random Forest model seems to be over-fitting based on the training accuracy. We also see that the Random Forest model performs better, and this is likely due to how we balanced the weights for the random forest model and not for the XGBoost model. Also to note, I had higher accuracy scores for my XGBoost model with other hyperparameters, however kept this version of the model as it had more balanced recall and precision scores as we will see soon.

#### Precision and Recall Values

Next, using the classification_report function from sklearn, I found the precision and recall values of the two models.

![Classification report](/assets/img/port2/classification_report.png)

At first glance, we can see that the "Not On Netflix" precision and recall values are pretty high (>84%). This shows that our model was generally good at predicting movies that weren't featured on Netflix. However, more importantly, is seems that our "On Netflix" precision and recall values are much lower (35% - 64%). The precision values shows that of all the movies our model predicted to be on Netflix, what percentage of those predictions were correct (i.e., out of all the predictions our random forest model predicted to be on Netflix, it was correct for 64% of its guesses). The recall values shows that out of the total movies that were on Netflix, what percentage of those movies did our model correctly predict to be on Netflix (i.e., our random forest model predicted 35% of all the Netflix movies to be on Netflix). We can see that our XGBoost model slightly outperformed our random forest model, however, I think there is still a way to help increase our values.

# Back to the Editing Room

---

## So, Streaming More Often Doesn't Help Gain More Followers?

Therefore, it seems that Lifewire and other sources claims were wrong and that streaming more often doesn't help streamers gain more followers. However, before we make any definite conclusions, we still have to look at the context of the data that was used in all our models. Firstly, the data from Twitch that we received only contained data from the Top 1000 streamers on Twitch. And although the data encompasses streamers who aren't the most impressive holistically (the smallest streamer in our dataset averages 235 viewers), it is still a very small and biased minority of the streamer community. These streamers are still at the top of the streaming ecosystem, and it is likely that different trends occur at the top compared to streamers just starting out. Also, there are many more smaller streamers in the streaming ecosystem, who not only are on Twitch, but on other platforms like Facebook, YouTube, and more. If we wanted to make less biased models, we would need a random sampling of all streamers on all streaming platforms (or better data for our models include all data of all streamers on all streaming platforms). 

Also, it seemed that Lifewire's advice was primarily targeting aspiring streamers. So if our data included only all streamers who started in the past year, we might be able to get different results that supports Lifewire's claim that streaming more helps streamers gain more followers. In the future, if that data is available, I would re-run the models with this data to see if there is a difference.

We also have to see awknowledge that there are many unknown factors that help a streamer grow, and some of those factors may not even be able to be recorded into data as of yet. For instance, some streamers blow up depending on their network connections. Or other streamers grow significantly because they excelled at streaming the newest gaming trend. As more variables of data are able to become recorded, we could create more accurate models to see how correlated certain variables are to gaining followers.

However, in conclusion, we found that within the top 1000 streamers on Twitch, streaming more often is correlated negatively with gaining followers. Or top Twitch streamers who streamed more often, would often not gain as much followers as top Twitch streamers who streamed less often. 


