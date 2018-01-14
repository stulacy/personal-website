+++
date = 2017-12-29
modified = 2018-01-14
draft = false
tags = ["football", "elo", "machine learning", "Predictaball"]
title = "Elo ratings of the Premier League: mid-season review"
math = true
+++

This is going to be the first of 2 posts looking at the mid-season
performance of my football prediction and rating system,
[Predictaball](http://www.stuartlacy.co.uk/project/predictaball/). In
this post I'm going to focus on the [Elo rating
system](http://www.stuartlacy.co.uk/2017/08/31/implementing-an-elo-rating-system-for-european-football/).

Premier league standings
------------------------

I'll firstly look at how the teams in the Premiership stand, both in
terms of their Elo rating and their accumulated points, as displayed in
the table below, ordered by Elo. Over-performing teams, as defined by
being at least 3 ranks higher in points than in Elo, are coloured in
green, while under-performing teams, the opposite, are highlighted in
red.

Man City are dominating the Elo ranking, with 85 more points than
second-placed Chelsea, which is completely expected from their 18
successive (often high-scoring) victories. It can be seen that there is
an asymmetric rating distribution, with 13 teams below the mean. This
emphasises the dominance of the top 6 (Everton in 7th are a long way
behind 6th placed Arsenal). The competitiveness of the top teams is
highlighted by the fact that a mere 34 points separates second placed
Chelsea from 5th placed Liverpool, which is less than half the
difference from Man City to Chelsea.

<table class="table table-striped table-hover table-condensed table-responsive" style="width: auto !important; margin-left: auto; margin-right: auto;">
<thead>
<tr>
<th style="text-align:left;">
Elo rank
</th>
<th style="text-align:left;">
Team
</th>
<th style="text-align:center;">
Elo
</th>
<th style="text-align:center;">
Points
</th>
<th style="text-align:center;">
Points rank
</th>
<th style="text-align:center;">
Rank difference
</th>
<th style="text-align:center;">
Played
</th>
</tr>
</thead>
<tbody>
<tr>
<td style="text-align:left;">
1
</td>
<td style="text-align:left;">
<span style="color: black;">Man City</span>
</td>
<td style="text-align:center;">
1796
</td>
<td style="text-align:center;">
58
</td>
<td style="text-align:center;">
1
</td>
<td style="text-align:center;">
<span style="color: black;">0</span>
</td>
<td style="text-align:center;">
20
</td>
</tr>
<tr>
<td style="text-align:left;">
2
</td>
<td style="text-align:left;">
<span style="color: black;">Chelsea</span>
</td>
<td style="text-align:center;">
1711
</td>
<td style="text-align:center;">
42
</td>
<td style="text-align:center;">
3
</td>
<td style="text-align:center;">
<span style="color: black;">-1</span>
</td>
<td style="text-align:center;">
20
</td>
</tr>
<tr>
<td style="text-align:left;">
3
</td>
<td style="text-align:left;">
<span style="color: black;">Tottenham</span>
</td>
<td style="text-align:center;">
1695
</td>
<td style="text-align:center;">
37
</td>
<td style="text-align:center;">
5
</td>
<td style="text-align:center;">
<span style="color: black;">-2</span>
</td>
<td style="text-align:center;">
20
</td>
</tr>
<tr>
<td style="text-align:left;">
4
</td>
<td style="text-align:left;">
<span style="color: black;">Man Utd</span>
</td>
<td style="text-align:center;">
1687
</td>
<td style="text-align:center;">
43
</td>
<td style="text-align:center;">
2
</td>
<td style="text-align:center;">
<span style="color: black;">2</span>
</td>
<td style="text-align:center;">
20
</td>
</tr>
<tr>
<td style="text-align:left;">
5
</td>
<td style="text-align:left;">
<span style="color: black;">Liverpool</span>
</td>
<td style="text-align:center;">
1677
</td>
<td style="text-align:center;">
38
</td>
<td style="text-align:center;">
4
</td>
<td style="text-align:center;">
<span style="color: black;">1</span>
</td>
<td style="text-align:center;">
20
</td>
</tr>
<tr>
<td style="text-align:left;">
6
</td>
<td style="text-align:left;">
<span style="color: black;">Arsenal</span>
</td>
<td style="text-align:center;">
1633
</td>
<td style="text-align:center;">
37
</td>
<td style="text-align:center;">
5
</td>
<td style="text-align:center;">
<span style="color: black;">1</span>
</td>
<td style="text-align:center;">
20
</td>
</tr>
<tr>
<td style="text-align:left;">
7
</td>
<td style="text-align:left;">
<span style="color: black;">Everton</span>
</td>
<td style="text-align:center;">
1504
</td>
<td style="text-align:center;">
27
</td>
<td style="text-align:center;">
8
</td>
<td style="text-align:center;">
<span style="color: black;">-1</span>
</td>
<td style="text-align:center;">
20
</td>
</tr>
<tr>
<td style="text-align:left;">
8
</td>
<td style="text-align:left;">
<span style="color: black;">Leicester</span>
</td>
<td style="text-align:center;">
1494
</td>
<td style="text-align:center;">
27
</td>
<td style="text-align:center;">
8
</td>
<td style="text-align:center;">
<span style="color: black;">0</span>
</td>
<td style="text-align:center;">
20
</td>
</tr>
<tr>
<td style="text-align:left;">
9
</td>
<td style="text-align:left;">
<span style="color: black;">Burnley</span>
</td>
<td style="text-align:center;">
1470
</td>
<td style="text-align:center;">
30
</td>
<td style="text-align:center;">
7
</td>
<td style="text-align:center;">
<span style="color: black;">2</span>
</td>
<td style="text-align:center;">
19
</td>
</tr>
<tr>
<td style="text-align:left;">
10
</td>
<td style="text-align:left;">
<span style="color: red;">Southampton</span>
</td>
<td style="text-align:center;">
1427
</td>
<td style="text-align:center;">
19
</td>
<td style="text-align:center;">
14
</td>
<td style="text-align:center;">
<span style="color: red;">-4</span>
</td>
<td style="text-align:center;">
20
</td>
</tr>
<tr>
<td style="text-align:left;">
11
</td>
<td style="text-align:left;">
<span style="color: red;">Crystal Palace</span>
</td>
<td style="text-align:center;">
1425
</td>
<td style="text-align:center;">
18
</td>
<td style="text-align:center;">
15
</td>
<td style="text-align:center;">
<span style="color: red;">-4</span>
</td>
<td style="text-align:center;">
20
</td>
</tr>
<tr>
<td style="text-align:left;">
12
</td>
<td style="text-align:left;">
<span style="color: red;">Bournemouth</span>
</td>
<td style="text-align:center;">
1407
</td>
<td style="text-align:center;">
17
</td>
<td style="text-align:center;">
18
</td>
<td style="text-align:center;">
<span style="color: red;">-6</span>
</td>
<td style="text-align:center;">
20
</td>
</tr>
<tr>
<td style="text-align:left;">
13
</td>
<td style="text-align:left;">
<span style="color: green;">Watford</span>
</td>
<td style="text-align:center;">
1406
</td>
<td style="text-align:center;">
25
</td>
<td style="text-align:center;">
10
</td>
<td style="text-align:center;">
<span style="color: green;">3</span>
</td>
<td style="text-align:center;">
20
</td>
</tr>
<tr>
<td style="text-align:left;">
14
</td>
<td style="text-align:left;">
<span style="color: black;">West Ham</span>
</td>
<td style="text-align:center;">
1405
</td>
<td style="text-align:center;">
18
</td>
<td style="text-align:center;">
15
</td>
<td style="text-align:center;">
<span style="color: black;">-1</span>
</td>
<td style="text-align:center;">
20
</td>
</tr>
<tr>
<td style="text-align:left;">
15
</td>
<td style="text-align:left;">
<span style="color: black;">Stoke</span>
</td>
<td style="text-align:center;">
1395
</td>
<td style="text-align:center;">
20
</td>
<td style="text-align:center;">
13
</td>
<td style="text-align:center;">
<span style="color: black;">2</span>
</td>
<td style="text-align:center;">
20
</td>
</tr>
<tr>
<td style="text-align:left;">
16
</td>
<td style="text-align:left;">
<span style="color: green;">Huddersfield</span>
</td>
<td style="text-align:center;">
1387
</td>
<td style="text-align:center;">
23
</td>
<td style="text-align:center;">
11
</td>
<td style="text-align:center;">
<span style="color: green;">5</span>
</td>
<td style="text-align:center;">
20
</td>
</tr>
<tr>
<td style="text-align:left;">
17
</td>
<td style="text-align:left;">
<span style="color: black;">West Brom</span>
</td>
<td style="text-align:center;">
1380
</td>
<td style="text-align:center;">
15
</td>
<td style="text-align:center;">
19
</td>
<td style="text-align:center;">
<span style="color: black;">-2</span>
</td>
<td style="text-align:center;">
20
</td>
</tr>
<tr>
<td style="text-align:left;">
18
</td>
<td style="text-align:left;">
<span style="color: green;">Brighton</span>
</td>
<td style="text-align:center;">
1375
</td>
<td style="text-align:center;">
21
</td>
<td style="text-align:center;">
12
</td>
<td style="text-align:center;">
<span style="color: green;">6</span>
</td>
<td style="text-align:center;">
20
</td>
</tr>
<tr>
<td style="text-align:left;">
19
</td>
<td style="text-align:left;">
<span style="color: black;">Swansea</span>
</td>
<td style="text-align:center;">
1366
</td>
<td style="text-align:center;">
13
</td>
<td style="text-align:center;">
20
</td>
<td style="text-align:center;">
<span style="color: black;">-1</span>
</td>
<td style="text-align:center;">
20
</td>
</tr>
<tr>
<td style="text-align:left;">
20
</td>
<td style="text-align:left;">
<span style="color: green;">Newcastle</span>
</td>
<td style="text-align:center;">
1357
</td>
<td style="text-align:center;">
18
</td>
<td style="text-align:center;">
15
</td>
<td style="text-align:center;">
<span style="color: green;">5</span>
</td>
<td style="text-align:center;">
19
</td>
</tr>
</tbody>
</table>

Overall, Elo and points totals appear to be well correlated, with a few
exceptions. For example, Tottenham are third in terms of Elo, but 5th in
points. Likewise Man Utd are second in the actual league, but only have
the 4th highest Elo. Looking at the tail end of the league and we see
similar phenomena. Brighton are 18th when ranked by Elo, but are 12th in
the actual league, a difference of 6 ranks!

There are a number of reasons for this behaviour: the most obvious being
that points don't take the opponent's strength into consideration, while
Elo does. Winning a game against a team in the top 6th will result in
more Elo points than against a relegation candidate, but both wins would
be awarded with 3 points. A strength of Elo is that by taking opponent
strength into account, it shows a fixture-independent table, while
ranking by points isn't entirely fair if a team has managed to have
fewer games against the top 6 than others.

Due to the inclusion of margin of victory in the Elo update equation,
([see Elo explanation
post](http://www.stuartlacy.co.uk/2017/08/31/implementing-an-elo-rating-system-for-european-football/)),
a win by a larger score results in additional Elo points. This could
partly explain why Stoke are ranked 15th in Elo but 13th by points, as
they have the second worst goal difference in the league (-18). Another
potential explanation for this discrepancy between ranks is how promoted
teams are handled. Currently, only teams in the top 4 European leagues
are tracked, so when a team is promoted up to the Premier League (or La
Liga etc...), it is assigned the average rating of the relegated teams.
So Newcastle, Brighton, and Huddersfield were all given the same rating
(1350) at the start of the season, which isn't entirely accurate and it
may take longer than 20 games for their ratings to converge on their
actual values. This last point is quite important, Elo rating is a
continuous score with only a soft-reset each season, whereas points are
obviously wiped clean each summer. Just because a team has a high rating
doesn't mean they are going to be good now or in the future, but that
they were good enough in the past, and a lot can change in the
off-season.

The correlation between Elo rank and points rank is shown below (with a
high *r* value). Teams **above** the line have worse Elo than their
points suggest (thereby **overperforming** in the real league), while
teams **below** the line are scoring fewer points than their skill level
would suggest (**underperforming**). The difference between the top and
bottom of the league is clear, with teams at the top having less
variation between their 2 ranks, while teams in the lower half are more
dispersed. The 3 promoted teams (Brighton, Newcastle, Huddersfield) are
the 3 most over-performing teams (in that order), which suggests that
setting their Elos to be equal to the averaged rating of the relegated
teams isn't entirely accurate, although in absence of tracking the
rating of the lower leagues I can't see a better way of handling this
that still maintains a zero-sum system.

<img src="/img/eloreview_29122017/rankcorrelation.png" style="display: block; margin: auto;" />

Expected goals
--------------

No football analytics post is complete without mentioning **expected
goals (xG)**, the stat so beloved by analytics and yet so poorly
understood by football 'experts'. I'm using the table [found
here](https://pbs.twimg.com/media/DRzmN8hWkAAKL0C.jpg:large) of expected
points, provided by [Gracenote
sports](https://twitter.com/GracenoteLive) and [Simon
Gleave](https://twitter.com/SimonGleave) **although note that it is one
matchday behind**. Under-performing teams are highlighted in red and
over-performing in black, calculated as a 3 point difference compared to
the expected total. It provides an alternative view of performance
looking at match level data rather than just the result. Importantly,
these two methods of calculating over and under-performance are not
directly comparable, the Elo method purely based on match outcome and
the other comparing outcomes with how the match was played.

There are a number of differences with my Elo rating. Firstly, the top 6
are ordered differently, with Liverpool and Arsenal moving up into
positions 2 and 3 respectively (although it looks like Arsenal are only
ahead of Spurs on goal difference). Their model has identified
Liverpool, Arsenal, and Spurs as under-performing, while according to
Elo both Liverpool and Arsenal are ranked relatively fairly, although it
agrees that Spurs are under-performing. Both systems agree that Man Utd
are doing better than expected.

The biggest under-performer by xG is Crystal Palace, who are the joint
biggest under-performer by Elo, along with Southampton and Bournemouth.
The bottom half of the table doesn't contain any under-performing teams.
The over-performing teams as identified by xG are Burnley (first by
quite some margin), and Huddersfield, both of which are considered to be
over-performing by Elo but to a lesser extent, with Brighton the most
over-performing according to Elo.

Rating change over the season
-----------------------------

I'm interested to see how the team's ratings have changed over the
course of the season. I've plotted the temporal trend below, although it
can be hard to identify which team is which, any suggestions for how to
better visualise this with 20 lines when there isn't much y-separation
would be welcome!

The most immediate finding is the large separation between the top 6 and
the bottom 14. This is slightly worrying as it leads to a sense of
inevitability in games between a top-6 and a bottom-14 team, although
looking at it from a more positive perspective it allows for potentially
exciting games of football whenever the top-6 play each other. Man
City's fantastic season is shown here as they started the season ranked
3rd after Chelsea and Spurs, but overtook of them by the start of
October and never looked back, while Spurs started to slide down the
table. From my own perspective, I'm heartened to see Liverpool's
improvement following their 3-0 away win at Stoke at the end of
November.

Looking at the bottom of the ratings, we can see Newcastle,
Huddersfield, and Brighton starting the season at 1350 Elo, the lowest
of all teams, with Huddersfield immediately jumping up with their 3-0
away win at Crystal Palace, before falling back into the mid-table and
being kept company by Brighton, while Newcastle starting off well before
going on a losing run from mid-November.

<img src="/img/eloreview_29122017/ratingtrend.png" style="display: block; margin: auto;" />

The table below displays the change in Elo across the half-season so
far, once again demonstrating Man City's superiority, having gained more
than double the number of rating points of the second most improved team
(Man Utd). Swansea's dismal season is shown here, having lost 72 points
over the course of the season.

<table class="table table-striped table-hover table-condensed table-responsive" style="width: auto !important; margin-left: auto; margin-right: auto;">
<thead>
<tr>
<th style="text-align:left;">
Team
</th>
<th style="text-align:center;">
Current elo
</th>
<th style="text-align:center;">
$\Delta elo$
</th>
</tr>
</thead>
<tbody>
<tr>
<td style="text-align:left;">
Man City
</td>
<td style="text-align:center;">
1796
</td>
<td style="text-align:center;">
145
</td>
</tr>
<tr>
<td style="text-align:left;">
Man Utd
</td>
<td style="text-align:center;">
1687
</td>
<td style="text-align:center;">
70
</td>
</tr>
<tr>
<td style="text-align:left;">
Liverpool
</td>
<td style="text-align:center;">
1677
</td>
<td style="text-align:center;">
59
</td>
</tr>
<tr>
<td style="text-align:left;">
Burnley
</td>
<td style="text-align:center;">
1470
</td>
<td style="text-align:center;">
56
</td>
</tr>
<tr>
<td style="text-align:left;">
Huddersfield
</td>
<td style="text-align:center;">
1387
</td>
<td style="text-align:center;">
29
</td>
</tr>
<tr>
<td style="text-align:left;">
Chelsea
</td>
<td style="text-align:center;">
1711
</td>
<td style="text-align:center;">
17
</td>
</tr>
<tr>
<td style="text-align:left;">
Brighton
</td>
<td style="text-align:center;">
1375
</td>
<td style="text-align:center;">
17
</td>
</tr>
<tr>
<td style="text-align:left;">
Watford
</td>
<td style="text-align:center;">
1406
</td>
<td style="text-align:center;">
11
</td>
</tr>
<tr>
<td style="text-align:left;">
Arsenal
</td>
<td style="text-align:center;">
1633
</td>
<td style="text-align:center;">
9
</td>
</tr>
<tr>
<td style="text-align:left;">
Leicester
</td>
<td style="text-align:center;">
1494
</td>
<td style="text-align:center;">
8
</td>
</tr>
<tr>
<td style="text-align:left;">
Newcastle
</td>
<td style="text-align:center;">
1357
</td>
<td style="text-align:center;">
-1
</td>
</tr>
<tr>
<td style="text-align:left;">
Tottenham
</td>
<td style="text-align:center;">
1695
</td>
<td style="text-align:center;">
-7
</td>
</tr>
<tr>
<td style="text-align:left;">
Crystal Palace
</td>
<td style="text-align:center;">
1425
</td>
<td style="text-align:center;">
-26
</td>
</tr>
<tr>
<td style="text-align:left;">
Everton
</td>
<td style="text-align:center;">
1504
</td>
<td style="text-align:center;">
-40
</td>
</tr>
<tr>
<td style="text-align:left;">
Bournemouth
</td>
<td style="text-align:center;">
1407
</td>
<td style="text-align:center;">
-42
</td>
</tr>
<tr>
<td style="text-align:left;">
Stoke
</td>
<td style="text-align:center;">
1395
</td>
<td style="text-align:center;">
-54
</td>
</tr>
<tr>
<td style="text-align:left;">
Southampton
</td>
<td style="text-align:center;">
1427
</td>
<td style="text-align:center;">
-56
</td>
</tr>
<tr>
<td style="text-align:left;">
West Ham
</td>
<td style="text-align:center;">
1405
</td>
<td style="text-align:center;">
-61
</td>
</tr>
<tr>
<td style="text-align:left;">
West Brom
</td>
<td style="text-align:center;">
1380
</td>
<td style="text-align:center;">
-62
</td>
</tr>
<tr>
<td style="text-align:left;">
Swansea
</td>
<td style="text-align:center;">
1366
</td>
<td style="text-align:center;">
-72
</td>
</tr>
</tbody>
</table>

European leagues
----------------

Predictaball also tracks the other 3 major European leagues (La Liga,
Serie A, and the Bundesliga) even those these predictions for these
matches aren't tweeted. As I've already made this post longer than I
expected, and also because I know even less about these leagues than I
do the Premiership (I actually don't follow football that much, I just
enjoy predictive modelling), I'm just going to display the Elo tables
below without much commentary. Remember that the Elo systems are
league-dependent and scores from different leagues are not directly
comparable.

### La Liga

As with the Premier League, the league is effectively grouped into 2,
with the team in 3rd separated from the remaining 17 teams by 126
points, and only 255 points separating the 4th place team from last. By
calculating the standard deviation of Elo, we get a measure of the
spread of skill in the league, with a more competitive league having a
smaller skill range. This value is 142 for the Premier League and 133
for La Liga, which isn't a large difference.

There are also some discrepancies between the Elo ranking and the actual
league position, the biggest of which by far is Girona, who lie in 8th
in the league, but only have the 17th best Elo. On the other hand,
Espanyol have the 11th best Elo but are placed in 16th.

<table class="table table-striped table-hover table-condensed table-responsive" style="width: auto !important; margin-left: auto; margin-right: auto;">
<thead>
<tr>
<th style="text-align:left;">
Elo rank
</th>
<th style="text-align:left;">
Team
</th>
<th style="text-align:center;">
Elo
</th>
<th style="text-align:center;">
Points
</th>
<th style="text-align:center;">
Points rank
</th>
<th style="text-align:center;">
Rank difference
</th>
<th style="text-align:center;">
Played
</th>
</tr>
</thead>
<tbody>
<tr>
<td style="text-align:left;">
1
</td>
<td style="text-align:left;">
<span style="color: black;">Barcelona</span>
</td>
<td style="text-align:center;">
1837
</td>
<td style="text-align:center;">
45
</td>
<td style="text-align:center;">
1
</td>
<td style="text-align:center;">
<span style="color: black;">0</span>
</td>
<td style="text-align:center;">
17
</td>
</tr>
<tr>
<td style="text-align:left;">
2
</td>
<td style="text-align:left;">
<span style="color: black;">Real Madrid</span>
</td>
<td style="text-align:center;">
1738
</td>
<td style="text-align:center;">
31
</td>
<td style="text-align:center;">
4
</td>
<td style="text-align:center;">
<span style="color: black;">-2</span>
</td>
<td style="text-align:center;">
16
</td>
</tr>
<tr>
<td style="text-align:left;">
3
</td>
<td style="text-align:left;">
<span style="color: black;">Atletico Madrid</span>
</td>
<td style="text-align:center;">
1707
</td>
<td style="text-align:center;">
36
</td>
<td style="text-align:center;">
2
</td>
<td style="text-align:center;">
<span style="color: black;">1</span>
</td>
<td style="text-align:center;">
17
</td>
</tr>
<tr>
<td style="text-align:left;">
4
</td>
<td style="text-align:left;">
<span style="color: black;">Valencia</span>
</td>
<td style="text-align:center;">
1581
</td>
<td style="text-align:center;">
34
</td>
<td style="text-align:center;">
3
</td>
<td style="text-align:center;">
<span style="color: black;">1</span>
</td>
<td style="text-align:center;">
17
</td>
</tr>
<tr>
<td style="text-align:left;">
5
</td>
<td style="text-align:left;">
<span style="color: black;">Villarreal</span>
</td>
<td style="text-align:center;">
1554
</td>
<td style="text-align:center;">
27
</td>
<td style="text-align:center;">
6
</td>
<td style="text-align:center;">
<span style="color: black;">-1</span>
</td>
<td style="text-align:center;">
17
</td>
</tr>
<tr>
<td style="text-align:left;">
6
</td>
<td style="text-align:left;">
<span style="color: black;">Sevilla</span>
</td>
<td style="text-align:center;">
1544
</td>
<td style="text-align:center;">
29
</td>
<td style="text-align:center;">
5
</td>
<td style="text-align:center;">
<span style="color: black;">1</span>
</td>
<td style="text-align:center;">
17
</td>
</tr>
<tr>
<td style="text-align:left;">
7
</td>
<td style="text-align:left;">
<span style="color: red;">Athletic Bilbao</span>
</td>
<td style="text-align:center;">
1534
</td>
<td style="text-align:center;">
21
</td>
<td style="text-align:center;">
11
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
8
</td>
<td style="text-align:left;">
<span style="color: black;">Real Sociedad</span>
</td>
<td style="text-align:center;">
1508
</td>
<td style="text-align:center;">
23
</td>
<td style="text-align:center;">
8
</td>
<td style="text-align:center;">
<span style="color: black;">0</span>
</td>
<td style="text-align:center;">
17
</td>
</tr>
<tr>
<td style="text-align:left;">
9
</td>
<td style="text-align:left;">
<span style="color: black;">Eibar</span>
</td>
<td style="text-align:center;">
1492
</td>
<td style="text-align:center;">
24
</td>
<td style="text-align:center;">
7
</td>
<td style="text-align:center;">
<span style="color: black;">2</span>
</td>
<td style="text-align:center;">
17
</td>
</tr>
<tr>
<td style="text-align:left;">
10
</td>
<td style="text-align:left;">
<span style="color: black;">Celta Vigo</span>
</td>
<td style="text-align:center;">
1488
</td>
<td style="text-align:center;">
21
</td>
<td style="text-align:center;">
11
</td>
<td style="text-align:center;">
<span style="color: black;">-1</span>
</td>
<td style="text-align:center;">
17
</td>
</tr>
<tr>
<td style="text-align:left;">
11
</td>
<td style="text-align:left;">
<span style="color: red;">Espanyol</span>
</td>
<td style="text-align:center;">
1455
</td>
<td style="text-align:center;">
17
</td>
<td style="text-align:center;">
16
</td>
<td style="text-align:center;">
<span style="color: red;">-5</span>
</td>
<td style="text-align:center;">
16
</td>
</tr>
<tr>
<td style="text-align:left;">
12
</td>
<td style="text-align:left;">
<span style="color: black;">Leganes</span>
</td>
<td style="text-align:center;">
1450
</td>
<td style="text-align:center;">
21
</td>
<td style="text-align:center;">
11
</td>
<td style="text-align:center;">
<span style="color: black;">1</span>
</td>
<td style="text-align:center;">
16
</td>
</tr>
<tr>
<td style="text-align:left;">
13
</td>
<td style="text-align:left;">
<span style="color: green;">Getafe</span>
</td>
<td style="text-align:center;">
1434
</td>
<td style="text-align:center;">
23
</td>
<td style="text-align:center;">
8
</td>
<td style="text-align:center;">
<span style="color: green;">5</span>
</td>
<td style="text-align:center;">
17
</td>
</tr>
<tr>
<td style="text-align:left;">
14
</td>
<td style="text-align:left;">
<span style="color: red;">Alaves</span>
</td>
<td style="text-align:center;">
1423
</td>
<td style="text-align:center;">
15
</td>
<td style="text-align:center;">
17
</td>
<td style="text-align:center;">
<span style="color: red;">-3</span>
</td>
<td style="text-align:center;">
17
</td>
</tr>
<tr>
<td style="text-align:left;">
15
</td>
<td style="text-align:left;">
<span style="color: green;">Real Betis</span>
</td>
<td style="text-align:center;">
1403
</td>
<td style="text-align:center;">
21
</td>
<td style="text-align:center;">
11
</td>
<td style="text-align:center;">
<span style="color: green;">4</span>
</td>
<td style="text-align:center;">
16
</td>
</tr>
<tr>
<td style="text-align:left;">
16
</td>
<td style="text-align:left;">
<span style="color: red;">Malaga</span>
</td>
<td style="text-align:center;">
1400
</td>
<td style="text-align:center;">
11
</td>
<td style="text-align:center;">
19
</td>
<td style="text-align:center;">
<span style="color: red;">-3</span>
</td>
<td style="text-align:center;">
17
</td>
</tr>
<tr>
<td style="text-align:left;">
17
</td>
<td style="text-align:left;">
<span style="color: green;">Girona</span>
</td>
<td style="text-align:center;">
1393
</td>
<td style="text-align:center;">
23
</td>
<td style="text-align:center;">
8
</td>
<td style="text-align:center;">
<span style="color: green;">9</span>
</td>
<td style="text-align:center;">
17
</td>
</tr>
<tr>
<td style="text-align:left;">
18
</td>
<td style="text-align:left;">
<span style="color: green;">Levante</span>
</td>
<td style="text-align:center;">
1369
</td>
<td style="text-align:center;">
18
</td>
<td style="text-align:center;">
15
</td>
<td style="text-align:center;">
<span style="color: green;">3</span>
</td>
<td style="text-align:center;">
17
</td>
</tr>
<tr>
<td style="text-align:left;">
19
</td>
<td style="text-align:left;">
<span style="color: black;">La Coruna</span>
</td>
<td style="text-align:center;">
1363
</td>
<td style="text-align:center;">
12
</td>
<td style="text-align:center;">
18
</td>
<td style="text-align:center;">
<span style="color: black;">1</span>
</td>
<td style="text-align:center;">
16
</td>
</tr>
<tr>
<td style="text-align:left;">
20
</td>
<td style="text-align:left;">
<span style="color: black;">Las Palmas</span>
</td>
<td style="text-align:center;">
1326
</td>
<td style="text-align:center;">
11
</td>
<td style="text-align:center;">
19
</td>
<td style="text-align:center;">
<span style="color: black;">1</span>
</td>
<td style="text-align:center;">
16
</td>
</tr>
</tbody>
</table>

### Serie A

Serie A is characterized by 2 dominant teams, Juventus and Napoli, who
only have 5 Elo points separating them (and 1 point). Roma also look
strong but the gap to the 4th place team is 96 points. The standard
deviation of Elo for Serie A is 150, which is the highest of the
European leagues, suggesting that there is greater variability in team
skill.

There are a number of over-performers, such as Sampdoria who are placed
in 6th but have the 10th highest Elo, but interestingly very few
under-performers, with no team being rated 2 positions better by Elo
than their actual standing. I'm also amazed to see Benevento having
picked up a solitary point from 18 games, well deserving of the lowest
Elo score across all 4 leagues. This doesn't necessarily mean that
Benevento are the worst team in these leagues, but they are the
furtherest from their league's average.

<table class="table table-striped table-hover table-condensed table-responsive" style="width: auto !important; margin-left: auto; margin-right: auto;">
<thead>
<tr>
<th style="text-align:left;">
Elo rank
</th>
<th style="text-align:left;">
Team
</th>
<th style="text-align:center;">
Elo
</th>
<th style="text-align:center;">
Points
</th>
<th style="text-align:center;">
Points rank
</th>
<th style="text-align:center;">
Rank difference
</th>
<th style="text-align:center;">
Played
</th>
</tr>
</thead>
<tbody>
<tr>
<td style="text-align:left;">
1
</td>
<td style="text-align:left;">
<span style="color: black;">Juventus</span>
</td>
<td style="text-align:center;">
1779
</td>
<td style="text-align:center;">
44
</td>
<td style="text-align:center;">
2
</td>
<td style="text-align:center;">
<span style="color: black;">-1</span>
</td>
<td style="text-align:center;">
18
</td>
</tr>
<tr>
<td style="text-align:left;">
2
</td>
<td style="text-align:left;">
<span style="color: black;">Napoli</span>
</td>
<td style="text-align:center;">
1774
</td>
<td style="text-align:center;">
45
</td>
<td style="text-align:center;">
1
</td>
<td style="text-align:center;">
<span style="color: black;">1</span>
</td>
<td style="text-align:center;">
18
</td>
</tr>
<tr>
<td style="text-align:left;">
3
</td>
<td style="text-align:left;">
<span style="color: black;">Roma</span>
</td>
<td style="text-align:center;">
1730
</td>
<td style="text-align:center;">
38
</td>
<td style="text-align:center;">
3
</td>
<td style="text-align:center;">
<span style="color: black;">0</span>
</td>
<td style="text-align:center;">
17
</td>
</tr>
<tr>
<td style="text-align:left;">
4
</td>
<td style="text-align:left;">
<span style="color: black;">Inter</span>
</td>
<td style="text-align:center;">
1634
</td>
<td style="text-align:center;">
37
</td>
<td style="text-align:center;">
4
</td>
<td style="text-align:center;">
<span style="color: black;">0</span>
</td>
<td style="text-align:center;">
17
</td>
</tr>
<tr>
<td style="text-align:left;">
5
</td>
<td style="text-align:left;">
<span style="color: black;">Lazio</span>
</td>
<td style="text-align:center;">
1617
</td>
<td style="text-align:center;">
36
</td>
<td style="text-align:center;">
5
</td>
<td style="text-align:center;">
<span style="color: black;">0</span>
</td>
<td style="text-align:center;">
17
</td>
</tr>
<tr>
<td style="text-align:left;">
6
</td>
<td style="text-align:left;">
<span style="color: black;">Atalanta</span>
</td>
<td style="text-align:center;">
1575
</td>
<td style="text-align:center;">
27
</td>
<td style="text-align:center;">
6
</td>
<td style="text-align:center;">
<span style="color: black;">0</span>
</td>
<td style="text-align:center;">
18
</td>
</tr>
<tr>
<td style="text-align:left;">
7
</td>
<td style="text-align:left;">
<span style="color: black;">Fiorentina</span>
</td>
<td style="text-align:center;">
1553
</td>
<td style="text-align:center;">
26
</td>
<td style="text-align:center;">
8
</td>
<td style="text-align:center;">
<span style="color: black;">-1</span>
</td>
<td style="text-align:center;">
18
</td>
</tr>
<tr>
<td style="text-align:left;">
8
</td>
<td style="text-align:left;">
<span style="color: black;">Torino</span>
</td>
<td style="text-align:center;">
1516
</td>
<td style="text-align:center;">
24
</td>
<td style="text-align:center;">
9
</td>
<td style="text-align:center;">
<span style="color: black;">-1</span>
</td>
<td style="text-align:center;">
18
</td>
</tr>
<tr>
<td style="text-align:left;">
9
</td>
<td style="text-align:left;">
<span style="color: black;">Udinese</span>
</td>
<td style="text-align:center;">
1499
</td>
<td style="text-align:center;">
24
</td>
<td style="text-align:center;">
9
</td>
<td style="text-align:center;">
<span style="color: black;">0</span>
</td>
<td style="text-align:center;">
17
</td>
</tr>
<tr>
<td style="text-align:left;">
10
</td>
<td style="text-align:left;">
<span style="color: green;">Sampdoria</span>
</td>
<td style="text-align:center;">
1492
</td>
<td style="text-align:center;">
27
</td>
<td style="text-align:center;">
6
</td>
<td style="text-align:center;">
<span style="color: green;">4</span>
</td>
<td style="text-align:center;">
17
</td>
</tr>
<tr>
<td style="text-align:left;">
11
</td>
<td style="text-align:left;">
<span style="color: black;">Milan</span>
</td>
<td style="text-align:center;">
1483
</td>
<td style="text-align:center;">
24
</td>
<td style="text-align:center;">
9
</td>
<td style="text-align:center;">
<span style="color: black;">2</span>
</td>
<td style="text-align:center;">
18
</td>
</tr>
<tr>
<td style="text-align:left;">
12
</td>
<td style="text-align:left;">
<span style="color: green;">Bologna</span>
</td>
<td style="text-align:center;">
1454
</td>
<td style="text-align:center;">
24
</td>
<td style="text-align:center;">
9
</td>
<td style="text-align:center;">
<span style="color: green;">3</span>
</td>
<td style="text-align:center;">
18
</td>
</tr>
<tr>
<td style="text-align:left;">
13
</td>
<td style="text-align:left;">
<span style="color: black;">Sassuolo</span>
</td>
<td style="text-align:center;">
1426
</td>
<td style="text-align:center;">
20
</td>
<td style="text-align:center;">
14
</td>
<td style="text-align:center;">
<span style="color: black;">-1</span>
</td>
<td style="text-align:center;">
18
</td>
</tr>
<tr>
<td style="text-align:left;">
14
</td>
<td style="text-align:left;">
<span style="color: black;">Chievo</span>
</td>
<td style="text-align:center;">
1412
</td>
<td style="text-align:center;">
21
</td>
<td style="text-align:center;">
13
</td>
<td style="text-align:center;">
<span style="color: black;">1</span>
</td>
<td style="text-align:center;">
18
</td>
</tr>
<tr>
<td style="text-align:left;">
15
</td>
<td style="text-align:left;">
<span style="color: black;">Cagliari</span>
</td>
<td style="text-align:center;">
1393
</td>
<td style="text-align:center;">
17
</td>
<td style="text-align:center;">
15
</td>
<td style="text-align:center;">
<span style="color: black;">0</span>
</td>
<td style="text-align:center;">
18
</td>
</tr>
<tr>
<td style="text-align:left;">
16
</td>
<td style="text-align:left;">
<span style="color: black;">Genoa</span>
</td>
<td style="text-align:center;">
1391
</td>
<td style="text-align:center;">
17
</td>
<td style="text-align:center;">
15
</td>
<td style="text-align:center;">
<span style="color: black;">1</span>
</td>
<td style="text-align:center;">
18
</td>
</tr>
<tr>
<td style="text-align:left;">
17
</td>
<td style="text-align:left;">
<span style="color: black;">Crotone</span>
</td>
<td style="text-align:center;">
1353
</td>
<td style="text-align:center;">
15
</td>
<td style="text-align:center;">
17
</td>
<td style="text-align:center;">
<span style="color: black;">0</span>
</td>
<td style="text-align:center;">
18
</td>
</tr>
<tr>
<td style="text-align:left;">
18
</td>
<td style="text-align:left;">
<span style="color: black;">SPAL</span>
</td>
<td style="text-align:center;">
1343
</td>
<td style="text-align:center;">
15
</td>
<td style="text-align:center;">
17
</td>
<td style="text-align:center;">
<span style="color: black;">1</span>
</td>
<td style="text-align:center;">
18
</td>
</tr>
<tr>
<td style="text-align:left;">
19
</td>
<td style="text-align:left;">
<span style="color: black;">Verona</span>
</td>
<td style="text-align:center;">
1340
</td>
<td style="text-align:center;">
13
</td>
<td style="text-align:center;">
19
</td>
<td style="text-align:center;">
<span style="color: black;">0</span>
</td>
<td style="text-align:center;">
17
</td>
</tr>
<tr>
<td style="text-align:left;">
20
</td>
<td style="text-align:left;">
<span style="color: black;">Benevento</span>
</td>
<td style="text-align:center;">
1237
</td>
<td style="text-align:center;">
1
</td>
<td style="text-align:center;">
20
</td>
<td style="text-align:center;">
<span style="color: black;">0</span>
</td>
<td style="text-align:center;">
18
</td>
</tr>
</tbody>
</table>

### Bundesliga

The German league looks to be the most competitive, with the leaders
having the lowest Elo of these 4 leagues and bottom-placed team having
the highest. This is reflected in the standard deviation of Elo at 98,
far lower than the other leagues. A similar finding has been [identifed
previously](http://stuartlacy.co.uk/2016/07/23/is-la-liga-the-most-predictable-european-football-league/),
where the bookies were less accurate at predicting Bundesliga matches
than the 3 other leagues.

<table class="table table-striped table-hover table-condensed table-responsive" style="width: auto !important; margin-left: auto; margin-right: auto;">
<thead>
<tr>
<th style="text-align:left;">
Elo rank
</th>
<th style="text-align:left;">
Team
</th>
<th style="text-align:center;">
Elo
</th>
<th style="text-align:center;">
Points
</th>
<th style="text-align:center;">
Points rank
</th>
<th style="text-align:center;">
Rank difference
</th>
<th style="text-align:center;">
Played
</th>
</tr>
</thead>
<tbody>
<tr>
<td style="text-align:left;">
1
</td>
<td style="text-align:left;">
<span style="color: black;">Bayern Munich</span>
</td>
<td style="text-align:center;">
1780
</td>
<td style="text-align:center;">
41
</td>
<td style="text-align:center;">
1
</td>
<td style="text-align:center;">
<span style="color: black;">0</span>
</td>
<td style="text-align:center;">
17
</td>
</tr>
<tr>
<td style="text-align:left;">
2
</td>
<td style="text-align:left;">
<span style="color: black;">Borussia Dortmund</span>
</td>
<td style="text-align:center;">
1623
</td>
<td style="text-align:center;">
28
</td>
<td style="text-align:center;">
3
</td>
<td style="text-align:center;">
<span style="color: black;">-1</span>
</td>
<td style="text-align:center;">
17
</td>
</tr>
<tr>
<td style="text-align:left;">
3
</td>
<td style="text-align:left;">
<span style="color: black;">Bayern Leverkusen</span>
</td>
<td style="text-align:center;">
1573
</td>
<td style="text-align:center;">
28
</td>
<td style="text-align:center;">
3
</td>
<td style="text-align:center;">
<span style="color: black;">0</span>
</td>
<td style="text-align:center;">
17
</td>
</tr>
<tr>
<td style="text-align:left;">
4
</td>
<td style="text-align:left;">
<span style="color: red;">Hoffenheim</span>
</td>
<td style="text-align:center;">
1559
</td>
<td style="text-align:center;">
26
</td>
<td style="text-align:center;">
7
</td>
<td style="text-align:center;">
<span style="color: red;">-3</span>
</td>
<td style="text-align:center;">
17
</td>
</tr>
<tr>
<td style="text-align:left;">
5
</td>
<td style="text-align:left;">
<span style="color: green;">Schalke</span>
</td>
<td style="text-align:center;">
1554
</td>
<td style="text-align:center;">
30
</td>
<td style="text-align:center;">
2
</td>
<td style="text-align:center;">
<span style="color: green;">3</span>
</td>
<td style="text-align:center;">
17
</td>
</tr>
<tr>
<td style="text-align:left;">
6
</td>
<td style="text-align:left;">
<span style="color: green;">Leipzig</span>
</td>
<td style="text-align:center;">
1545
</td>
<td style="text-align:center;">
28
</td>
<td style="text-align:center;">
3
</td>
<td style="text-align:center;">
<span style="color: green;">3</span>
</td>
<td style="text-align:center;">
17
</td>
</tr>
<tr>
<td style="text-align:left;">
7
</td>
<td style="text-align:left;">
<span style="color: green;">Borussia Moenchengladbach</span>
</td>
<td style="text-align:center;">
1522
</td>
<td style="text-align:center;">
28
</td>
<td style="text-align:center;">
3
</td>
<td style="text-align:center;">
<span style="color: green;">4</span>
</td>
<td style="text-align:center;">
17
</td>
</tr>
<tr>
<td style="text-align:left;">
8
</td>
<td style="text-align:left;">
<span style="color: black;">Augsburg</span>
</td>
<td style="text-align:center;">
1504
</td>
<td style="text-align:center;">
24
</td>
<td style="text-align:center;">
9
</td>
<td style="text-align:center;">
<span style="color: black;">-1</span>
</td>
<td style="text-align:center;">
17
</td>
</tr>
<tr>
<td style="text-align:left;">
9
</td>
<td style="text-align:left;">
<span style="color: black;">Hertha</span>
</td>
<td style="text-align:center;">
1480
</td>
<td style="text-align:center;">
24
</td>
<td style="text-align:center;">
9
</td>
<td style="text-align:center;">
<span style="color: black;">0</span>
</td>
<td style="text-align:center;">
17
</td>
</tr>
<tr>
<td style="text-align:left;">
9
</td>
<td style="text-align:left;">
<span style="color: black;">Ein Frankfurt</span>
</td>
<td style="text-align:center;">
1480
</td>
<td style="text-align:center;">
26
</td>
<td style="text-align:center;">
7
</td>
<td style="text-align:center;">
<span style="color: black;">2</span>
</td>
<td style="text-align:center;">
17
</td>
</tr>
<tr>
<td style="text-align:left;">
11
</td>
<td style="text-align:left;">
<span style="color: black;">Wolfsburg</span>
</td>
<td style="text-align:center;">
1475
</td>
<td style="text-align:center;">
19
</td>
<td style="text-align:center;">
12
</td>
<td style="text-align:center;">
<span style="color: black;">-1</span>
</td>
<td style="text-align:center;">
17
</td>
</tr>
<tr>
<td style="text-align:left;">
12
</td>
<td style="text-align:left;">
<span style="color: red;">Werder Bremen</span>
</td>
<td style="text-align:center;">
1463
</td>
<td style="text-align:center;">
15
</td>
<td style="text-align:center;">
16
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
13
</td>
<td style="text-align:left;">
<span style="color: black;">Hannover</span>
</td>
<td style="text-align:center;">
1438
</td>
<td style="text-align:center;">
23
</td>
<td style="text-align:center;">
11
</td>
<td style="text-align:center;">
<span style="color: black;">2</span>
</td>
<td style="text-align:center;">
17
</td>
</tr>
<tr>
<td style="text-align:left;">
14
</td>
<td style="text-align:left;">
<span style="color: black;">Mainz</span>
</td>
<td style="text-align:center;">
1418
</td>
<td style="text-align:center;">
17
</td>
<td style="text-align:center;">
14
</td>
<td style="text-align:center;">
<span style="color: black;">0</span>
</td>
<td style="text-align:center;">
17
</td>
</tr>
<tr>
<td style="text-align:left;">
15
</td>
<td style="text-align:left;">
<span style="color: black;">Hamburg</span>
</td>
<td style="text-align:center;">
1413
</td>
<td style="text-align:center;">
15
</td>
<td style="text-align:center;">
16
</td>
<td style="text-align:center;">
<span style="color: black;">-1</span>
</td>
<td style="text-align:center;">
17
</td>
</tr>
<tr>
<td style="text-align:left;">
15
</td>
<td style="text-align:left;">
<span style="color: green;">Freiburg</span>
</td>
<td style="text-align:center;">
1413
</td>
<td style="text-align:center;">
19
</td>
<td style="text-align:center;">
12
</td>
<td style="text-align:center;">
<span style="color: green;">3</span>
</td>
<td style="text-align:center;">
17
</td>
</tr>
<tr>
<td style="text-align:left;">
17
</td>
<td style="text-align:left;">
<span style="color: black;">Koln</span>
</td>
<td style="text-align:center;">
1382
</td>
<td style="text-align:center;">
6
</td>
<td style="text-align:center;">
18
</td>
<td style="text-align:center;">
<span style="color: black;">-1</span>
</td>
<td style="text-align:center;">
17
</td>
</tr>
<tr>
<td style="text-align:left;">
18
</td>
<td style="text-align:left;">
<span style="color: green;">Stuttgart</span>
</td>
<td style="text-align:center;">
1380
</td>
<td style="text-align:center;">
17
</td>
<td style="text-align:center;">
14
</td>
<td style="text-align:center;">
<span style="color: green;">4</span>
</td>
<td style="text-align:center;">
17
</td>
</tr>
</tbody>
</table>
