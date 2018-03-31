+++
# Date this page was created.
date = "2017-10-05"

# Project title.
title = "Predictaball"

# Project summary to display on homepage.
summary = "A machine learning sports prediction bot"

# Optional image to display on homepage (relative to `static/img/` folder).
image_preview = "predictaball_logo.png"

# Tags: can be used for filtering projects.
tags = ["machine learning", "Predictaball", "football"]

# Optional external URL for project (replaces project detail page).
external_link = ""

# Does the project detail page use math formatting?
math = false

# Optional featured image (relative to `static/img/` folder).
[header]
image = "headers/predictaball_twitter_banner.png"
#caption = "My caption :smile:"

+++

[Predictaball](http://stuartlacy.co.uk/tags/predictaball/) is a Sports prediction bot, currently providing outcome predictions for football (soccer) matches. Each day it automatically scrapes the fixtures for that day and calculates outcome probabilities using a [Bayesian hierarchical model](http://stuartlacy.co.uk/2017/08/31/implementing-an-elo-rating-system-for-european-football/). It is currently implemented for the four main European leagues (La Liga, Premiership, Serie A and the Bundesliga), with the Premiership predictions being [tweeted daily](https://twitter.com/thepredictaball). New for the 2017-2018 season, it rates teams in these leagues using a [modified Elo system](http://stuartlacy.co.uk/2017/08/31/implementing-an-elo-rating-system-for-european-football/). I've also implemented an [betting model](http://localhost:1313/2017/06/28/predicting-football-results-in-2016-2017-with-machine-learning---automated-betting-system/).

Football team ratings and predictions can be found online at [thepredictaball.com](https://www.thepredictaball.com), which will hopefully be expanded to include additional leagues and sports in the future.

