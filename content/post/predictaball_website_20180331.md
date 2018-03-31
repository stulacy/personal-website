+++
date = 2018-03-31
draft = false
tags = ["Predictaball", "football", "machine learning"]
title = "Predictaball has its own website!"
math = false
+++

I've created a website for Predictaball with team ratings and match predictions for all 4 main European leagues, at [thepredictaball.com](https://thepredictaball.com). It has each team's current rating and plots showing the change over the course of season, along with match outcome forecasts. Various statistics are also included, such as the biggest upset, worst teams in history, as well as this season's predictive accuracy. Previously only Premiership match predictions were made available (via Twitter) and so I'm happy that I've finally got this website released.

For example, the _Ratings_ tab shows the league ranked by their current rating, alongside an interactive plot showing the change in ratings for each team over the course of the season. In the screenshot below we can observe the clear separation between the top 6 teams in the Premiership and the remaining 14.
![ratings as shown on the website](/img/predictaballwebsite_20180331/ratings.png)

The _Predictions_ tab shows the predicted probabilities for each day's matches. For completed games, the probability of the actual outcome is displayed in green if it was the most likely outcome, and red if it wasn't.
![match predictions](/img/predictaballwebsite_20180331/predictions.png)

I have to admit it isn't the best looking site, but I'm happy with it since it conveys all the information I want to. I'm not a great UI designer and would rather keep it working and functional than tinker with the appearance.
However, one thing I would like to change about it is the framework I've used, which is [Shiny](https://shiny.rstudio.com/).
I built it with Shiny as I have experience in using it for small web-apps and knew I could get something up and running rather quickly.

However, I quickly realised this site wouldn't need any of the interactivity that Shiny excels at, instead it's just reading content straight from the database. As a result, I get the drawbacks of Shiny's slow page loading time (not helped by using the smallest EC2 instance), without the benefits of a true interactive dashboard.

At some point in the summer I'd like to get round to rebuilding it with a more suitable lightweight framework, probably whatever Javascript framework is flavour of the month by then. This would result in a smoother browsing experience, potentially a simpler codebase, as well as providing me with experience in Javascript web-dev, which I've been itching for a while.

In the meantime, please enjoy the website and let me know if you have any feedback.


