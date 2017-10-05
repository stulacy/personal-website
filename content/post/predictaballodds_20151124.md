+++
date = 2015-11-24
draft = false
tags = ["football", "Predictaball", "machine learning"]
title = "Incorporating odds into Predictaball"
math = false
+++

I've tinkered around with Predictaball a bit recently in an effort to increase its accuracy, with the overall goal of beating Paul Merson and Lawro so that I can claim 'human competitiveness'. I've mentioned in previous posts that I envisage 2 potential ways to achieve this.

  1. Include more player data
  2. Incorporate bookies odds

Adding more player data (such as a variable for each player indicating whether they are in the squad or not) would allow the model to account for situations when a player who is strongly associated with the team winning is now injured - for an example see City's abysmal record when Kompany isn't playing. However having variables for the thirty members of a club football squad would prove very hard to interpret analytically, in addition to having a large number of parameters. A logistic regression model for instance wouldn't fare so well here. 

In a somewhat controversial 2002 paper, Leo Breiman discusses the difference between data driven approaches (typically linear models formed using conventional statistical techniques) and algorithmic methods (black box models such as SVMs and random forests which are highly accurate but very hard to interpret). A crucial part of model design is knowing when each approach is appropriate, and for a data set comprising a large number of predictors for each player a black box would definitely be more useful rather than trying to manually tweak the model. As an aside I'd definitely recommend reading Breiman's paper in full, it can be found here [Breiman, Leo. _Statistical modeling: The two cultures (with comments and a rejoinder by the author)._ Statistical Science 16.3 (2001): 199-231.](http://projecteuclid.org/euclid.ss/1009213726) complete with responses from big names in the statistical community such as Sir David Cox and Brad Efron. In fact I may even devote a future post to it, it's that important.

### Large model

One idea on how to develop Predictaball is to incorporate a huge amount of information about each match, mostly focused around player availability but also potentially including factors such as whether they have a new manager, if they're close to relegation zone, resting time since last match and so on.

### Playing the bookies

The second idea I had was to exploit the large amount of already available predictions, i.e. bookies odds. While eventually I'd like to be able to have my own predictions more accurate than the bookies give (using the above technique), in the meantime it seems a shame to not make use of their predictions. The main idea is to compare the odds offered by different websites to find firstly the most likely outcome, and secondly, identify bookies which are offering good bets. Obviously this is something that some people do currently but it's time consuming to do manually; an automated system would provide numerous benefits! I'd need to fine tune this idea a bit, but a manual data driven approach would work well here, trading off different bookies (along with Predictaball's predicted outcomes) to highlight good bets.

### Incorporating odds into Predictaball

To this extent I've started having a play with the available data and have downloaded a set of historical odds from a particular bookies in order to train a model. For now I've just extended my current predictor set of W/L/D for each team to include the W/L/D outcome probabilities as provided by the bookies. Straight away this has helped Predictaball achieve a 15% increase in accuracy, from an estimated 48% correct predictions to 55%, which is better than Merson but still significantly worse than Lawro. However note that by using bookies odds, even from multiple sources, it's unlikely that I'll ever be able to largely improve upon them. There's more scope for improving accuracy by obtaining more data to reflect true signal in the data.

Incidentally, the bookies achieved an accuracy of 56% on the same unseen test data, so Predictaball is still slightly worse than just betting with the favourite. This is where incorporating _big data_ would be helpful to be able to beat the bookies, by including such a large range of information in my model I'd hope to be able to offer greater predictive power than the bookies have, and therefore be able to identify good bets.

### Modeling profit

Another idea I had was to form a secondary modeling stage, combining the predicted probabilities of each outcome from Predictaball (just using W/L/D input features) with the bookies odds for these outcomes into a model which aims to provide a binary "Yes you should place this bet" or "Avoid this one!" outcome. Using the current rather naive Predictaball implementation I was able to make a profit of £4, from 100 matches (only about 40 or so were deemed as good bets), which obviously is far from satisfactory in its current state but lays the groundwork for a good system when Predictaball is more capable.

## Future directions

To an extent the two approaches I've bee describing have contrasting objectives, the first method aims to guess the most likely outcome of a match, while the second will be more concerned with maximising profit. A hierarchical system may be most appropriate here, inputting Predictaball's objective guesses (from assessing which players are available etc...) into a secondary model with various bookies odds to identify which bets I should take to optimise my income.

My main focus is thereby to improve Predictaball's accuracy, either by incorporating a vast range of match meta data, or by adopting a forecasting approach, to then form the secondary model to establish whether the odds offered by a bookies represent a good bet or not. This could later be extended to sweep multiple bookies to find multiple good bets.

