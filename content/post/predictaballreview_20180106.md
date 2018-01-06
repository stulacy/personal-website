+++
date = 2018-01-06
draft = false
tags = ["football", "elo", "machine learning", "Predictaball"]
title = "Predictaball: mid-season review"
math = true
+++

This post continues on from the mid-season review [of the Elo
system](http://www.stuartlacy.co.uk/2017/12/29/elo-ratings-of-the-premier-league-mid-season-review/)
and looks at my Bayesian football prediction model,
[Predictaball](http://www.stuartlacy.co.uk/project/predictaball/), up to
and including matchday 20 of the Premier League (29th December). I'll go
over the overall predictive accuracy and compare my model to others,
including bookies, expected goals (xG), and a compilation of football
models.

Overall accuracy
----------------

So far, across the top 4 European leagues, there have been 696 matches
with 379 (54%) of these outcomes being correctly predicted. While this
figure is immediately interpretable to us, accuracy as an evaluation of
a statistical model has a number of drawbacks, the chief one being that
it doesn't take the predicted probabilities into account.

A proper scoring measure for ordinal data (i.e. takes probabilities and
the order of the 3 outcomes into account) is the Ranked Probability
Score (RPS), introduced by [Constantinou & Fenton,
2012](http://constantinou.info/downloads/papers/solvingtheproblem.pdf).
Predictaball's current RPS across all 4 leagues is 0.193, which may be
directly less interpretable than accuracy (essentially a **lower** value
is better) but provides more information about the forecasting ability
of the model.

The RPS and accuracies for Predictaball are displayed in the table
below, along with those for William Hill as a comparison. It shows
several trends, firstly, that there is a certain amount of variation
between the leagues' predictability, which I've identified before
([here](http://www.stuartlacy.co.uk/2017/12/29/elo-ratings-of-the-premier-league-mid-season-review/)
and
[here](http://stuartlacy.co.uk/2016/07/23/is-la-liga-the-most-predictable-european-football-league/)).
La Liga and the Bundesliga are a lot less easy to forecast than the
Premier League and Serie A, both by Predictaball and William Hill (and
have lower accuracies too). However, this highlights another interesting
result, that RPS and accuracy aren't necessarily well correlated. The
RPS for the Bundesliga is very similar to that of La Liga for both
models, yet its accuracies are far lower than La Liga's.

Finally, it shows that my model is less accurate than that used by
William Hill. However, I'd expect that to be the case, given that my
model is a very simple model that only uses the team's [Elo
rating](http://stuartlacy.co.uk/2017/08/31/implementing-an-elo-rating-system-for-european-football/)
(which is solely based on match results), rather than including any
finer detail, such as xG, injuries, and other factors, which I'd fully
expect to be included in a model used by William Hill in order to
maintain their edge over punters who are starting to use their own
models to influence their betting (such as Predictaball).

<table class="table table-striped table-hover table-condensed table-responsive" style="width: auto !important; margin-left: auto; margin-right: auto;">
<thead>
<tr>
<th style="border-bottom:hidden" colspan="1">
</th>
<th style="text-align:center; border-bottom:hidden; padding-bottom:0; padding-left:3px;padding-right:3px;" colspan="2">
RPS

</th>
<th style="text-align:center; border-bottom:hidden; padding-bottom:0; padding-left:3px;padding-right:3px;" colspan="2">
Accuracy

</th>
</tr>
<tr>
<th style="text-align:left;">
League
</th>
<th style="text-align:center;">
Predictaball
</th>
<th style="text-align:center;">
William Hill
</th>
<th style="text-align:center;">
Predictaball
</th>
<th style="text-align:center;">
William Hill
</th>
</tr>
</thead>
<tbody>
<tr>
<td style="text-align:left;">
Serie A
</td>
<td style="text-align:center;">
0.183
</td>
<td style="text-align:center;">
0.172
</td>
<td style="text-align:center;">
59.0
</td>
<td style="text-align:center;">
61.2
</td>
</tr>
<tr>
<td style="text-align:left;">
Premier League
</td>
<td style="text-align:center;">
0.185
</td>
<td style="text-align:center;">
0.180
</td>
<td style="text-align:center;">
55.8
</td>
<td style="text-align:center;">
57.3
</td>
</tr>
<tr>
<td style="text-align:left;">
Bundesliga
</td>
<td style="text-align:center;">
0.204
</td>
<td style="text-align:center;">
0.197
</td>
<td style="text-align:center;">
48.0
</td>
<td style="text-align:center;">
49.3
</td>
</tr>
<tr>
<td style="text-align:left;">
La Liga
</td>
<td style="text-align:center;">
0.205
</td>
<td style="text-align:center;">
0.195
</td>
<td style="text-align:center;">
53.9
</td>
<td style="text-align:center;">
56.3
</td>
</tr>
</tbody>
</table>
Comparison with 35 other models
-------------------------------

A user on Twitter by the name of [Alex
B](https://twitter.com/fussbALEXperte) has very kindly collated the
results of 35 Premier League forecasting models and keeps a running
total. See his [page
here](https://cognitivefootball.wordpress.com/2017/12/22/smwdtkdtktlfo3/)
where he discusses the setup, and find the current standings
[here](https://cognitivefootball.wordpress.com/rps-17-18/). Also note
his fantastic [cartoonifed
drawing](https://pbs.twimg.com/media/DSTIzw5WkAEHGW-.jpg:large) of my
profile photo!

As of matchday 20, Predictaball is tenth out of 35 models with an RPS of
0.1775 (note this is different to the value above as Alex starting
collecting predictions from matchday 9). I'm very happy with this
result, as again I'm using a very straight forward model that is solely
based on match results and doesn't include any of the metrics that are
now widely available, such as expected goals (xG). I also don't include
any player-level information. In future, and I'll discuss this at the
end of the post, I'd definitely like to tweak my model to improve a few
steps up the ladder.

Expected points
---------------

In the [Elo mid-season
review](http://www.stuartlacy.co.uk/2017/12/29/elo-ratings-of-the-premier-league-mid-season-review/)
I posted a table comparing a team's current standing to their Elo
ranking in order to identify over and under-performing teams. In this
post I'll do something similar with a slightly different technique: I'll
calculate the expected points for each team by simulating the expected
outcome from Predictaball's forecasts for each game (I used a thousand
simulations). I'll then compare these expected points to those obtained
from an xG model that [Simon Gleave](https://twitter.com/SimonGleave)
has developed and displayed [on
Twitter](https://twitter.com/SimonGleave/status/946775975345508352). I
believe the expected points based from this model are not pre-match
forecasts (**prospective** estimates), but rather **retrospectively**
assigning 3 points each game to the team that had the higheset expected
goals score.

The resultant table is below and shows a lot of information! Where the
expected points differs from the actual by at least 3, the value is
coloured in either **green** (team has more points than expected, team
is **over-performing**), or **red** (team has fewer points than
predicted, team is **under-performing**).

**NB: It is important to note that the two models here are not directly
comparable: an over-performing team as highlighted by Predictaball is
one that is winning games against teams that are higher rated, while an
over-performing team in terms of xG is one that is winning games despite
having poorer scoring opportunities during the match.**

<table class="table table-striped table-hover table-condensed table-responsive" style="width: auto !important; margin-left: auto; margin-right: auto;">
<thead>
<tr>
<th style="border-bottom:hidden" colspan="2">
</th>
<th style="text-align:center; border-bottom:hidden; padding-bottom:0; padding-left:3px;padding-right:3px;" colspan="2">
Predictaball

</th>
<th style="text-align:center; border-bottom:hidden; padding-bottom:0; padding-left:3px;padding-right:3px;" colspan="2">
xG

</th>
<th style="border-bottom:hidden" colspan="1">
</th>
</tr>
<tr>
<th style="text-align:left;">
Team
</th>
<th style="text-align:center;">
Points
</th>
<th style="text-align:center;">
Estimated
</th>
<th style="text-align:center;">
Difference
</th>
<th style="text-align:center;">
Estimated
</th>
<th style="text-align:center;">
Difference
</th>
<th style="text-align:center;">
Elo rank
</th>
</tr>
</thead>
<tbody>
<tr>
<td style="text-align:left;">
Man City
</td>
<td style="text-align:center;">
58
</td>
<td style="text-align:center;">
<span style="color: green;">43</span>
</td>
<td style="text-align:center;">
<span style="color: green;">15</span>
</td>
<td style="text-align:center;">
<span style="color: black;">56</span>
</td>
<td style="text-align:center;">
<span style="color: black;">2</span>
</td>
<td style="text-align:center;">
1
</td>
</tr>
<tr>
<td style="text-align:left;">
Chelsea
</td>
<td style="text-align:center;">
42
</td>
<td style="text-align:center;">
<span style="color: black;">41</span>
</td>
<td style="text-align:center;">
<span style="color: black;">1</span>
</td>
<td style="text-align:center;">
<span style="color: black;">42</span>
</td>
<td style="text-align:center;">
<span style="color: black;">0</span>
</td>
<td style="text-align:center;">
2
</td>
</tr>
<tr>
<td style="text-align:left;">
Tottenham
</td>
<td style="text-align:center;">
37
</td>
<td style="text-align:center;">
<span style="color: red;">41</span>
</td>
<td style="text-align:center;">
<span style="color: red;">-4</span>
</td>
<td style="text-align:center;">
<span style="color: red;">46</span>
</td>
<td style="text-align:center;">
<span style="color: red;">-9</span>
</td>
<td style="text-align:center;">
3
</td>
</tr>
<tr>
<td style="text-align:left;">
Man Utd
</td>
<td style="text-align:center;">
43
</td>
<td style="text-align:center;">
<span style="color: green;">38</span>
</td>
<td style="text-align:center;">
<span style="color: green;">5</span>
</td>
<td style="text-align:center;">
<span style="color: green;">37</span>
</td>
<td style="text-align:center;">
<span style="color: green;">6</span>
</td>
<td style="text-align:center;">
4
</td>
</tr>
<tr>
<td style="text-align:left;">
Liverpool
</td>
<td style="text-align:center;">
38
</td>
<td style="text-align:center;">
<span style="color: black;">36</span>
</td>
<td style="text-align:center;">
<span style="color: black;">2</span>
</td>
<td style="text-align:center;">
<span style="color: red;">47</span>
</td>
<td style="text-align:center;">
<span style="color: red;">-9</span>
</td>
<td style="text-align:center;">
5
</td>
</tr>
<tr>
<td style="text-align:left;">
Arsenal
</td>
<td style="text-align:center;">
37
</td>
<td style="text-align:center;">
<span style="color: black;">36</span>
</td>
<td style="text-align:center;">
<span style="color: black;">1</span>
</td>
<td style="text-align:center;">
<span style="color: red;">46</span>
</td>
<td style="text-align:center;">
<span style="color: red;">-9</span>
</td>
<td style="text-align:center;">
6
</td>
</tr>
<tr>
<td style="text-align:left;">
Everton
</td>
<td style="text-align:center;">
27
</td>
<td style="text-align:center;">
<span style="color: black;">28</span>
</td>
<td style="text-align:center;">
<span style="color: black;">-1</span>
</td>
<td style="text-align:center;">
<span style="color: green;">21</span>
</td>
<td style="text-align:center;">
<span style="color: green;">6</span>
</td>
<td style="text-align:center;">
7
</td>
</tr>
<tr>
<td style="text-align:left;">
Southampton
</td>
<td style="text-align:center;">
19
</td>
<td style="text-align:center;">
<span style="color: red;">26</span>
</td>
<td style="text-align:center;">
<span style="color: red;">-7</span>
</td>
<td style="text-align:center;">
<span style="color: red;">24</span>
</td>
<td style="text-align:center;">
<span style="color: red;">-5</span>
</td>
<td style="text-align:center;">
10
</td>
</tr>
<tr>
<td style="text-align:left;">
Leicester
</td>
<td style="text-align:center;">
27
</td>
<td style="text-align:center;">
<span style="color: black;">26</span>
</td>
<td style="text-align:center;">
<span style="color: black;">1</span>
</td>
<td style="text-align:center;">
<span style="color: green;">24</span>
</td>
<td style="text-align:center;">
<span style="color: green;">3</span>
</td>
<td style="text-align:center;">
8
</td>
</tr>
<tr>
<td style="text-align:left;">
Watford
</td>
<td style="text-align:center;">
25
</td>
<td style="text-align:center;">
<span style="color: black;">24</span>
</td>
<td style="text-align:center;">
<span style="color: black;">1</span>
</td>
<td style="text-align:center;">
<span style="color: black;">25</span>
</td>
<td style="text-align:center;">
<span style="color: black;">0</span>
</td>
<td style="text-align:center;">
13
</td>
</tr>
<tr>
<td style="text-align:left;">
West Ham
</td>
<td style="text-align:center;">
18
</td>
<td style="text-align:center;">
<span style="color: red;">23</span>
</td>
<td style="text-align:center;">
<span style="color: red;">-5</span>
</td>
<td style="text-align:center;">
<span style="color: black;">16</span>
</td>
<td style="text-align:center;">
<span style="color: black;">2</span>
</td>
<td style="text-align:center;">
12
</td>
</tr>
<tr>
<td style="text-align:left;">
Stoke
</td>
<td style="text-align:center;">
20
</td>
<td style="text-align:center;">
<span style="color: red;">23</span>
</td>
<td style="text-align:center;">
<span style="color: red;">-3</span>
</td>
<td style="text-align:center;">
<span style="color: green;">16</span>
</td>
<td style="text-align:center;">
<span style="color: green;">4</span>
</td>
<td style="text-align:center;">
15
</td>
</tr>
<tr>
<td style="text-align:left;">
West Brom
</td>
<td style="text-align:center;">
15
</td>
<td style="text-align:center;">
<span style="color: red;">23</span>
</td>
<td style="text-align:center;">
<span style="color: red;">-8</span>
</td>
<td style="text-align:center;">
<span style="color: red;">19</span>
</td>
<td style="text-align:center;">
<span style="color: red;">-4</span>
</td>
<td style="text-align:center;">
17
</td>
</tr>
<tr>
<td style="text-align:left;">
Bournemouth
</td>
<td style="text-align:center;">
17
</td>
<td style="text-align:center;">
<span style="color: red;">22</span>
</td>
<td style="text-align:center;">
<span style="color: red;">-5</span>
</td>
<td style="text-align:center;">
<span style="color: black;">16</span>
</td>
<td style="text-align:center;">
<span style="color: black;">1</span>
</td>
<td style="text-align:center;">
14
</td>
</tr>
<tr>
<td style="text-align:left;">
Crystal Palace
</td>
<td style="text-align:center;">
18
</td>
<td style="text-align:center;">
<span style="color: red;">22</span>
</td>
<td style="text-align:center;">
<span style="color: red;">-4</span>
</td>
<td style="text-align:center;">
<span style="color: red;">31</span>
</td>
<td style="text-align:center;">
<span style="color: red;">-13</span>
</td>
<td style="text-align:center;">
11
</td>
</tr>
<tr>
<td style="text-align:left;">
Burnley
</td>
<td style="text-align:center;">
30
</td>
<td style="text-align:center;">
<span style="color: green;">21</span>
</td>
<td style="text-align:center;">
<span style="color: green;">9</span>
</td>
<td style="text-align:center;">
<span style="color: green;">18</span>
</td>
<td style="text-align:center;">
<span style="color: green;">12</span>
</td>
<td style="text-align:center;">
9
</td>
</tr>
<tr>
<td style="text-align:left;">
Swansea
</td>
<td style="text-align:center;">
13
</td>
<td style="text-align:center;">
<span style="color: red;">21</span>
</td>
<td style="text-align:center;">
<span style="color: red;">-8</span>
</td>
<td style="text-align:center;">
<span style="color: black;">15</span>
</td>
<td style="text-align:center;">
<span style="color: black;">-2</span>
</td>
<td style="text-align:center;">
19
</td>
</tr>
<tr>
<td style="text-align:left;">
Huddersfield
</td>
<td style="text-align:center;">
23
</td>
<td style="text-align:center;">
<span style="color: green;">20</span>
</td>
<td style="text-align:center;">
<span style="color: green;">3</span>
</td>
<td style="text-align:center;">
<span style="color: green;">14</span>
</td>
<td style="text-align:center;">
<span style="color: green;">9</span>
</td>
<td style="text-align:center;">
15
</td>
</tr>
<tr>
<td style="text-align:left;">
Brighton
</td>
<td style="text-align:center;">
21
</td>
<td style="text-align:center;">
<span style="color: black;">20</span>
</td>
<td style="text-align:center;">
<span style="color: black;">1</span>
</td>
<td style="text-align:center;">
<span style="color: green;">14</span>
</td>
<td style="text-align:center;">
<span style="color: green;">7</span>
</td>
<td style="text-align:center;">
18
</td>
</tr>
<tr>
<td style="text-align:left;">
Newcastle
</td>
<td style="text-align:center;">
18
</td>
<td style="text-align:center;">
<span style="color: black;">18</span>
</td>
<td style="text-align:center;">
<span style="color: black;">0</span>
</td>
<td style="text-align:center;">
<span style="color: black;">19</span>
</td>
<td style="text-align:center;">
<span style="color: black;">-1</span>
</td>
<td style="text-align:center;">
20
</td>
</tr>
</tbody>
</table>
Looking at Predictaball firstly, and there are a few notable outliers,
such as Man City who are 15 points ahead of where they are expected to
be. This isn't that surprising given their record-breaking 18 straight
victories, which few models would predict ahead of time. Since the xG
model is calculating expected points retrospectively based on scoring
opportunities it demonstrates that Man City are successfully generating
better chances than their opponents as they are only 2 points away from
their expected total.

Both models have Spurs identified as under-achievers, with them having 9
fewer points than the xG model predicted and 4 fewer than Predictaball
estimated. The interpretation here is that Predictaball rates Spurs
highly, but they are not obtaining the expected results. However, the
difference in 9 points with the xG model indicates that they are playing
well in these games and generating good scoring opportunities but they
are not being converted.

Southampton, West Brom, and Crystal Palace are other teams marked down
as under-achievers by both models, with Crystal Palace having a massive
13 fewer points than expected by xG. Burnley are well-identified as
over-performing by both models, having 9 and 12 points more than
expected by both models. The teams that both models have predicted most
accurately are Chelsea, Watford, and regrettably for their fans,
Newcastle, all of whom had their points total correctly predicted to
within 1 point.

Future modifications
--------------------

I've got several ways in which I plan to update the Predictaball system
in 2018. I firstly want to look at different methods of generating the
outcome probabilities, which I believe could be done simply within the
Elo framework. Another idea is to fit a model based on the Elo ratings
directly optimising the RPM, as the current Bayesian method models a
multi-nomial outcome and thus doesn't take the ordering into account.
Probabilistic modelling methods, such as
[Edward](http://edwardlib.org/), are starting to gain traction in the
machine learning community and could be used for this purpose. They
combine the probabilistic computing approaches of software such as BUGS,
JAGS, and STAN, with modern deep learning software, such as Tensor Flow
and PyTorch, and thereby allow a probabilistic model with an arbitrary
cost function.

I've been saying for years how I'd like to incorporate player-level
information, but I'm now rather happy with this rating method. It is
conceptually simple, provides an alternative league ranking, easily
interpretable, and doesn't require much effort in obtaining the
information to forecast each match. And most importantly it is rather
accurate; my current model is the most accurate of the 3 incarnations
I've used, and the Euro Club Index that is currently doing very well in
[Alex's standings](https://cognitivefootball.wordpress.com/rps-17-18/)
is a very similar rating method to mine.

One way in which I would like to adapt my model, however, is to predict
the score. This would provide a closed loop between my Elo rating (that
incorporates margin of victory), and my forecasting model, allowing me
to forecast whole seasons in advance. It would also open up access to an
increased range of bets, such as both teams to score. I'll write about
this if I do ever get around to developing it.

And finally, I intend to update the infrastructure behind Predictaball
this year, to provide a web-app with both the current ratings and the
match predictions freely available. I'd then like to broaden the scope
to include more leagues and more sports. Watch this space...
