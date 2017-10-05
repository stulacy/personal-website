+++
date = 2016-05-29
draft = false
tags = ["football", "machine learning", "Predictaball"]
title = "Predictaball end of season review for 2015-2016"
math = false
+++

This post summarises Predictaball's performance in the 2015-2016 season. I'll look at overall performance, accuracy per week, how it fared in terms of making profit, and finally the annual comparison with Lawro.

Compared to [last year](http://stuartlacy.co.uk/27102015-predictaballendseason) when it achieved 48% overall, Predictaball has fared less well this season with 43%. This isn't largely surprising since this season has been full of surprises to say the least, with Leicester beating out the traditional top four for the title, and Spurs doing their best to break the monopoly (despite failing in typical Spurs fashion). Yet it could be expected that since the algorithm is basing its predictions solely on the past 5 performances, it should be relatively free of bias (unlike all the pundits at the start of the season who forecast Leicester to get relegated) and so should be able to predict matches relatively easily. 

It would be useful to come up with a statistic to measure a season's _predictability_ to test this hypothesis. Such a measure would also identify which leagues are most easily predictable, enabling more accurate tips and also just for interest. However, I'm unlucky to incorporate this until I roll out Predictaball to cover more leagues and sports, which shouldn't be too challenging now that the main infrastructure is in place but as always lots of minor unforeseen issues will undoubtedly crop up. 

The plot below details the weekly performance of Predictaball. 

![Weekly average](/img/endofseason_2016/weekly_performance.png)

Interestingly, it did better in the second half of the season, this was when [I added the bookies' odds to the feature set](http://stuartlacy.co.uk/25112015-predictaball-odds)so perhaps this result isn't largely surprising. However, ideally my prediction algorithm won't need the bookie's odds, so that it can recommend which bets are likely to produce profit. Incidentally, betting £1 on each match this season with William Hill following the outcome estimated by Predictaball would have ended you up losing £36.40, so there's still some work to go to reach the goal of an automated profitable prediction algorithm. This is especially challenging given that I'm just doing this in my free time, compared to the experts working full time on this employed by the bookies.

Lawro got 47% this year, once again beating me. This is definitely one of my main goals for next season!

For further work, I'd like to look at different ways of modeling the data, including:

  - Utilise more predictors
  - Frame the problem as a forecasting problem
  - Build a Bayesian model

After that, when I've got a reliable and more accurate prediction model I can look into voting strategies, i.e. do I vote on every match, or just those where the difference between the predicted outcome as estimated by Predictaball and the odds is above a certain threshold, or some other method?

I hope to get models built on additional predictors and forecasting networks built within the next month or so, a Bayesian approach may take a fair bit longer as I need to get to grips with the theory before going blindly head first into an application.

