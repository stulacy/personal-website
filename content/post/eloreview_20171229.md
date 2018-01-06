+++
date = 2017-12-29
draft = false
tags = ["football", "elo", "machine learning", "Predictaball"]
title = "Elo ratings of the Premier League: mid-season review"
math = true
+++

This is going to be the first of 2 posts looking at the
mid-season performance of my football prediction and rating system,
[Predictaball](http://www.stuartlacy.co.uk/project/predictaball/). In
this post I'm going to focus on the [Elo rating
system](http://www.stuartlacy.co.uk/2017/08/31/implementing-an-elo-rating-system-for-european-football/).

Premier league standings
------------------------

I'll firstly look at how the teams in the Premiership stand, both in
terms of their Elo rating and their accumulated points, as displayed
in the table below, ordered by Elo. Man City are dominating the Elo ranking, with 84
more points than second-placed Chelsea, which is completely expected
from their 18 successive (often high-scoring) victories. Remembering that this
system is designed to have a mean rating of 1500, it can be seen that
there is an asymmetric rating distribution, with 13 teams below
the mean. This emphasises the dominance of the top 6 (Everton in 7th
are a long way behind 6th placed Arsenal). The competitiveness of the
top teams is highlighted by the fact that a mere 36 points separates
second placed Chelsea from 5th placed Liverpool, which is less than half
the difference from Man City to Chelsea.

<table class="table table-striped table-hover table-condensed table-responsive" style="width: auto !important; margin-left: auto; margin-right: auto;">
<thead>
<tr>
<th style="text-align:right;">
Elo rank
</th>
<th style="text-align:left;">
Team
</th>
<th style="text-align:right;">
Elo
</th>
<th style="text-align:right;">
Points
</th>
<th style="text-align:right;">
Points rank
</th>
<th style="text-align:right;">
Rank difference
</th>
<th style="text-align:right;">
Played
</th>
</tr>
</thead>
<tbody>
<tr>
<td style="text-align:right;">
1
</td>
<td style="text-align:left;">
Man City
</td>
<td style="text-align:right;">
1812
</td>
<td style="text-align:right;">
58
</td>
<td style="text-align:right;">
1
</td>
<td style="text-align:right;">
0
</td>
<td style="text-align:right;">
20
</td>
</tr>
<tr>
<td style="text-align:right;">
2
</td>
<td style="text-align:left;">
Chelsea
</td>
<td style="text-align:right;">
1728
</td>
<td style="text-align:right;">
42
</td>
<td style="text-align:right;">
3
</td>
<td style="text-align:right;">
-1
</td>
<td style="text-align:right;">
20
</td>
</tr>
<tr>
<td style="text-align:right;">
3
</td>
<td style="text-align:left;">
Tottenham
</td>
<td style="text-align:right;">
1718
</td>
<td style="text-align:right;">
37
</td>
<td style="text-align:right;">
5
</td>
<td style="text-align:right;">
-2
</td>
<td style="text-align:right;">
20
</td>
</tr>
<tr>
<td style="text-align:right;">
4
</td>
<td style="text-align:left;">
Man Utd
</td>
<td style="text-align:right;">
1701
</td>
<td style="text-align:right;">
43
</td>
<td style="text-align:right;">
2
</td>
<td style="text-align:right;">
2
</td>
<td style="text-align:right;">
20
</td>
</tr>
<tr>
<td style="text-align:right;">
5
</td>
<td style="text-align:left;">
Liverpool
</td>
<td style="text-align:right;">
1692
</td>
<td style="text-align:right;">
38
</td>
<td style="text-align:right;">
4
</td>
<td style="text-align:right;">
1
</td>
<td style="text-align:right;">
20
</td>
</tr>
<tr>
<td style="text-align:right;">
6
</td>
<td style="text-align:left;">
Arsenal
</td>
<td style="text-align:right;">
1650
</td>
<td style="text-align:right;">
37
</td>
<td style="text-align:right;">
5
</td>
<td style="text-align:right;">
1
</td>
<td style="text-align:right;">
20
</td>
</tr>
<tr>
<td style="text-align:right;">
7
</td>
<td style="text-align:left;">
Everton
</td>
<td style="text-align:right;">
1505
</td>
<td style="text-align:right;">
27
</td>
<td style="text-align:right;">
8
</td>
<td style="text-align:right;">
-1
</td>
<td style="text-align:right;">
20
</td>
</tr>
<tr>
<td style="text-align:right;">
8
</td>
<td style="text-align:left;">
Leicester
</td>
<td style="text-align:right;">
1491
</td>
<td style="text-align:right;">
27
</td>
<td style="text-align:right;">
8
</td>
<td style="text-align:right;">
0
</td>
<td style="text-align:right;">
20
</td>
</tr>
<tr>
<td style="text-align:right;">
9
</td>
<td style="text-align:left;">
Burnley
</td>
<td style="text-align:right;">
1460
</td>
<td style="text-align:right;">
30
</td>
<td style="text-align:right;">
7
</td>
<td style="text-align:right;">
2
</td>
<td style="text-align:right;">
19
</td>
</tr>
<tr>
<td style="text-align:right;">
10
</td>
<td style="text-align:left;">
Southampton
</td>
<td style="text-align:right;">
1419
</td>
<td style="text-align:right;">
19
</td>
<td style="text-align:right;">
14
</td>
<td style="text-align:right;">
-4
</td>
<td style="text-align:right;">
20
</td>
</tr>
<tr>
<td style="text-align:right;">
11
</td>
<td style="text-align:left;">
Crystal Palace
</td>
<td style="text-align:right;">
1416
</td>
<td style="text-align:right;">
18
</td>
<td style="text-align:right;">
15
</td>
<td style="text-align:right;">
-4
</td>
<td style="text-align:right;">
20
</td>
</tr>
<tr>
<td style="text-align:right;">
12
</td>
<td style="text-align:left;">
West Ham
</td>
<td style="text-align:right;">
1408
</td>
<td style="text-align:right;">
18
</td>
<td style="text-align:right;">
15
</td>
<td style="text-align:right;">
-3
</td>
<td style="text-align:right;">
20
</td>
</tr>
<tr>
<td style="text-align:right;">
13
</td>
<td style="text-align:left;">
Watford
</td>
<td style="text-align:right;">
1401
</td>
<td style="text-align:right;">
25
</td>
<td style="text-align:right;">
10
</td>
<td style="text-align:right;">
3
</td>
<td style="text-align:right;">
20
</td>
</tr>
<tr>
<td style="text-align:right;">
14
</td>
<td style="text-align:left;">
Bournemouth
</td>
<td style="text-align:right;">
1391
</td>
<td style="text-align:right;">
17
</td>
<td style="text-align:right;">
18
</td>
<td style="text-align:right;">
-4
</td>
<td style="text-align:right;">
20
</td>
</tr>
<tr>
<td style="text-align:right;">
15
</td>
<td style="text-align:left;">
Huddersfield
</td>
<td style="text-align:right;">
1387
</td>
<td style="text-align:right;">
23
</td>
<td style="text-align:right;">
11
</td>
<td style="text-align:right;">
4
</td>
<td style="text-align:right;">
20
</td>
</tr>
<tr>
<td style="text-align:right;">
15
</td>
<td style="text-align:left;">
Stoke
</td>
<td style="text-align:right;">
1387
</td>
<td style="text-align:right;">
20
</td>
<td style="text-align:right;">
13
</td>
<td style="text-align:right;">
2
</td>
<td style="text-align:right;">
20
</td>
</tr>
<tr>
<td style="text-align:right;">
17
</td>
<td style="text-align:left;">
West Brom
</td>
<td style="text-align:right;">
1376
</td>
<td style="text-align:right;">
15
</td>
<td style="text-align:right;">
19
</td>
<td style="text-align:right;">
-2
</td>
<td style="text-align:right;">
20
</td>
</tr>
<tr>
<td style="text-align:right;">
18
</td>
<td style="text-align:left;">
Brighton
</td>
<td style="text-align:right;">
1364
</td>
<td style="text-align:right;">
21
</td>
<td style="text-align:right;">
12
</td>
<td style="text-align:right;">
6
</td>
<td style="text-align:right;">
20
</td>
</tr>
<tr>
<td style="text-align:right;">
19
</td>
<td style="text-align:left;">
Swansea
</td>
<td style="text-align:right;">
1354
</td>
<td style="text-align:right;">
13
</td>
<td style="text-align:right;">
20
</td>
<td style="text-align:right;">
-1
</td>
<td style="text-align:right;">
20
</td>
</tr>
<tr>
<td style="text-align:right;">
20
</td>
<td style="text-align:left;">
Newcastle
</td>
<td style="text-align:right;">
1340
</td>
<td style="text-align:right;">
18
</td>
<td style="text-align:right;">
15
</td>
<td style="text-align:right;">
5
</td>
<td style="text-align:right;">
19
</td>
</tr>
</tbody>
</table>

Overall, Elo and points totals appear to be well correlated, with a few
exceptions. For example, Tottenham are third in terms of Elo, but 5th in
points. Likewise, Man Utd are second in the actual league, but
only have the 4th highest Elo. Looking at the tail end of the league and
we see similar phenomena. Brighton are 18th in Elo, but 
12th in the actual league, a difference of 6 ranks!

There are a number of reasons for this behaviour: the most obvious being
that points don't take the opponent's strength into consideration, while
Elo does. Winning a game against a team in the top 6th will result in
more Elo points than against a relegation candidate, but both are awarded with 3 points. A strength of Elo is that by taking opponent
strength into account, it shows a fixture-independent table, while
ranking by points isn't entirely fair if a team has managed to have
fewer games against the top 6.

Due to the inclusion of margin of victory in the Elo update equation,
([see Elo explanation
post](http://www.stuartlacy.co.uk/2017/08/31/implementing-an-elo-rating-system-for-european-football/)),
a win by a larger score results in additional Elo points. This could
partly explain why Stoke are ranked 15th in Elo but 13th by points, as
they have the second worst goal difference in the league (-18). Another
potential explanation for this discrepancy is how promoted
teams are handled. Currently, only teams in the top 4 European leagues
are tracked, so when a team is promoted up to the Premier League (or La
Liga etc...), it is assigned the average rating of the relegated teams.
So Newcastle, Brighton, and Huddersfield were all given the same rating
(1350) at the start of the season, which isn't entirely accurate and it
may take longer than 20 games for their ratings to converge on their
actual values. This last point is quite important, Elo rating in my implementation is a
continuous score with only a soft-reset each season, whereas points are
wiped clean each summer. Just because a team finished a season on a high, it
doesn't mean they are going to start the next season at the same level.


The correlation between Elo rank and points rank is shown below. Teams **above** the blue line have worse Elo than their
points would suggest (thereby **overperforming** in the real league), while
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

![](/img/eloreview_29122017/rankcorrelation.png)

Expected goals
--------------

No football analytics post is complete without mentioning 
**expected goals (xG)**, the stat so beloved by analytics and yet so poorly
understood by football 'experts'. I'm using the table [found
here](https://pbs.twimg.com/media/DRzmN8hWkAAKL0C.jpg:large) of expected points, provided by
[Gracenote Sports](https://twitter.com/GracenoteLive) and [Simon
Gleave](https://twitter.com/SimonGleave) **although note that it is one matchday behind**. Under-performing
teams are highlighted in red and over-performing in black, calculated as
a 3 point difference compared to the expected total. It provides an
alternative view of performance looking at match level data rather than
just the result. Importantly, these two methods of calculating over and
under-performance are not directly comparable, with the Elo method purely
based on match outcome and the other comparing outcomes with how
the match was played.

There are a number of differences with my Elo rating. Firstly, the top 6
are ordered differently, with Liverpool and Arsenal up moving into
positions 2 and 3 respectively (although it looks like Arsenal are only
ahead of Spurs on goal difference). Their model has identified
Liverpool, Arsenal, and Spurs as under-performing, while according to
Elo both Liverpool and Arsenal are ranked relatively fairly,
although it agrees that Spurs are under-performing. Both systems agree
that Man Utd are doing better than expected.

The biggest under-performer by xG is Crystal Palace, who are the joint
biggest under-performer by Elo, along with Southampton and Bournemouth.
The bottom half of the table doesn't contain any under-performing teams.
The over-performing teams as identified by xG are Burnley (first by
quite some margin), and Huddersfield, both of which are considered to be over-performing
by Elo but to a lesser extent, with Brighton the most over-performing
according to Elo.

Rating change over the season
-----------------------------

I'm interested to see how the team's ratings have changed over the
course of the season. I've plotted the temporal trend below, although it
can be hard to identify which team is which, any suggestions for how to
better visualise this with 20 lines when there isn't much y-separation
would be welcome!

The most immediate finding is the large separation between the top 6 and the bottom 14.
This is slightly worrying as it leads to a
sense of inevitability in games between a top-6 and a bottom-14 team,
although looking at it from a more positive perspective it allows for
potentially exciting games of football whenever the top-6 play each
other. Man City's fantastic season is shown here as they started the
season ranked 3rd after Chelsea and Spurs, and overtook Spurs by the
start of November and never looked back, while Spurs started to slide
down the table. From my own perspective, I'm heartened to see
Liverpool's improvement following their 3-0 away win at Stoke at the end
of November.

Looking at the bottom of the ratings, we can see Newcastle,
Huddersfield, and Brighton starting the season at 1350 Elo, the lowest
of all teams, with Huddersfield immediately jumping up with their 3-0
away win at Crystal Palace, before falling back into the mid-table and
being kept company by Brighton, while Newcastle start off well but start on a losing run in mid-November.

![](/img/eloreview_29122017/ratingtrend.png)

The table below displays the change in Elo across the season so
far, once again demonstrating Man City's superiority, having gained more
than double the number of rating points of the second most improved team
(Man Utd). Swansea's dismal season is shown here, having lost 73 points
over the course of the season.

<table class="table table-striped table-hover table-condensed table-responsive" style="width: auto !important; margin-left: auto; margin-right: auto;">
<thead>
<tr>
<th style="text-align:left;">
Team
</th>
<th style="text-align:right;">
Current elo
</th>
<th style="text-align:right;">
$\Delta elo$
</th>
</tr>
</thead>
<tbody>
<tr>
<td style="text-align:left;">
Man City
</td>
<td style="text-align:right;">
1812
</td>
<td style="text-align:right;">
139
</td>
</tr>
<tr>
<td style="text-align:left;">
Man Utd
</td>
<td style="text-align:right;">
1701
</td>
<td style="text-align:right;">
71
</td>
</tr>
<tr>
<td style="text-align:left;">
Liverpool
</td>
<td style="text-align:right;">
1692
</td>
<td style="text-align:right;">
68
</td>
</tr>
<tr>
<td style="text-align:left;">
Burnley
</td>
<td style="text-align:right;">
1460
</td>
<td style="text-align:right;">
54
</td>
</tr>
<tr>
<td style="text-align:left;">
Huddersfield
</td>
<td style="text-align:right;">
1387
</td>
<td style="text-align:right;">
37
</td>
</tr>
<tr>
<td style="text-align:left;">
Chelsea
</td>
<td style="text-align:right;">
1728
</td>
<td style="text-align:right;">
24
</td>
</tr>
<tr>
<td style="text-align:left;">
Brighton
</td>
<td style="text-align:right;">
1364
</td>
<td style="text-align:right;">
14
</td>
</tr>
<tr>
<td style="text-align:left;">
Leicester
</td>
<td style="text-align:right;">
1491
</td>
<td style="text-align:right;">
10
</td>
</tr>
<tr>
<td style="text-align:left;">
Arsenal
</td>
<td style="text-align:right;">
1650
</td>
<td style="text-align:right;">
7
</td>
</tr>
<tr>
<td style="text-align:left;">
Watford
</td>
<td style="text-align:right;">
1401
</td>
<td style="text-align:right;">
7
</td>
</tr>
<tr>
<td style="text-align:left;">
Tottenham
</td>
<td style="text-align:right;">
1718
</td>
<td style="text-align:right;">
1
</td>
</tr>
<tr>
<td style="text-align:left;">
Newcastle
</td>
<td style="text-align:right;">
1340
</td>
<td style="text-align:right;">
-10
</td>
</tr>
<tr>
<td style="text-align:left;">
Crystal Palace
</td>
<td style="text-align:right;">
1416
</td>
<td style="text-align:right;">
-27
</td>
</tr>
<tr>
<td style="text-align:left;">
Everton
</td>
<td style="text-align:right;">
1505
</td>
<td style="text-align:right;">
-39
</td>
</tr>
<tr>
<td style="text-align:left;">
Bournemouth
</td>
<td style="text-align:right;">
1391
</td>
<td style="text-align:right;">
-47
</td>
</tr>
<tr>
<td style="text-align:left;">
West Ham
</td>
<td style="text-align:right;">
1408
</td>
<td style="text-align:right;">
-54
</td>
</tr>
<tr>
<td style="text-align:left;">
West Brom
</td>
<td style="text-align:right;">
1376
</td>
<td style="text-align:right;">
-58
</td>
</tr>
<tr>
<td style="text-align:left;">
Stoke
</td>
<td style="text-align:right;">
1387
</td>
<td style="text-align:right;">
-59
</td>
</tr>
<tr>
<td style="text-align:left;">
Southampton
</td>
<td style="text-align:right;">
1419
</td>
<td style="text-align:right;">
-65
</td>
</tr>
<tr>
<td style="text-align:left;">
Swansea
</td>
<td style="text-align:right;">
1354
</td>
<td style="text-align:right;">
-73
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
with the team in 3rd separated from the remaining 17 teams by 161
points, and only 246 points separating the 4th place team from last. By calculating the standard
deviation of Elo, we get a measure of the spread of skill in the league, with a more competitive league having a smaller skill range. This value is 154 for the Premier League and 146 for La Liga, which isn't a large difference.

There are also some discrepancies between the Elo ranking and the
actual league position, the biggest of which by far is Girona, who lie
in 8th in the league, but only have the 17th best Elo. On the other
hand, Espanyol have the 11th best Elo but are placed in 16th.

<table class="table table-striped table-hover table-condensed table-responsive" style="width: auto !important; margin-left: auto; margin-right: auto;">
<thead>
<tr>
<th style="text-align:right;">
Elo rank
</th>
<th style="text-align:left;">
Team
</th>
<th style="text-align:right;">
Elo
</th>
<th style="text-align:right;">
Points
</th>
<th style="text-align:right;">
Points rank
</th>
<th style="text-align:right;">
Rank difference
</th>
<th style="text-align:right;">
Played
</th>
</tr>
</thead>
<tbody>
<tr>
<td style="text-align:right;">
1
</td>
<td style="text-align:left;">
Barcelona
</td>
<td style="text-align:right;">
1870
</td>
<td style="text-align:right;">
45
</td>
<td style="text-align:right;">
1
</td>
<td style="text-align:right;">
0
</td>
<td style="text-align:right;">
17
</td>
</tr>
<tr>
<td style="text-align:right;">
2
</td>
<td style="text-align:left;">
Real Madrid
</td>
<td style="text-align:right;">
1758
</td>
<td style="text-align:right;">
31
</td>
<td style="text-align:right;">
4
</td>
<td style="text-align:right;">
-2
</td>
<td style="text-align:right;">
16
</td>
</tr>
<tr>
<td style="text-align:right;">
3
</td>
<td style="text-align:left;">
Atletico Madrid
</td>
<td style="text-align:right;">
1730
</td>
<td style="text-align:right;">
36
</td>
<td style="text-align:right;">
2
</td>
<td style="text-align:right;">
1
</td>
<td style="text-align:right;">
17
</td>
</tr>
<tr>
<td style="text-align:right;">
4
</td>
<td style="text-align:left;">
Valencia
</td>
<td style="text-align:right;">
1569
</td>
<td style="text-align:right;">
34
</td>
<td style="text-align:right;">
3
</td>
<td style="text-align:right;">
1
</td>
<td style="text-align:right;">
17
</td>
</tr>
<tr>
<td style="text-align:right;">
5
</td>
<td style="text-align:left;">
Villarreal
</td>
<td style="text-align:right;">
1568
</td>
<td style="text-align:right;">
27
</td>
<td style="text-align:right;">
6
</td>
<td style="text-align:right;">
-1
</td>
<td style="text-align:right;">
17
</td>
</tr>
<tr>
<td style="text-align:right;">
6
</td>
<td style="text-align:left;">
Sevilla
</td>
<td style="text-align:right;">
1564
</td>
<td style="text-align:right;">
29
</td>
<td style="text-align:right;">
5
</td>
<td style="text-align:right;">
1
</td>
<td style="text-align:right;">
17
</td>
</tr>
<tr>
<td style="text-align:right;">
7
</td>
<td style="text-align:left;">
Athletic Bilbao
</td>
<td style="text-align:right;">
1549
</td>
<td style="text-align:right;">
21
</td>
<td style="text-align:right;">
11
</td>
<td style="text-align:right;">
-4
</td>
<td style="text-align:right;">
17
</td>
</tr>
<tr>
<td style="text-align:right;">
8
</td>
<td style="text-align:left;">
Real Sociedad
</td>
<td style="text-align:right;">
1508
</td>
<td style="text-align:right;">
23
</td>
<td style="text-align:right;">
8
</td>
<td style="text-align:right;">
0
</td>
<td style="text-align:right;">
17
</td>
</tr>
<tr>
<td style="text-align:right;">
9
</td>
<td style="text-align:left;">
Celta Vigo
</td>
<td style="text-align:right;">
1489
</td>
<td style="text-align:right;">
21
</td>
<td style="text-align:right;">
11
</td>
<td style="text-align:right;">
-2
</td>
<td style="text-align:right;">
17
</td>
</tr>
<tr>
<td style="text-align:right;">
10
</td>
<td style="text-align:left;">
Eibar
</td>
<td style="text-align:right;">
1474
</td>
<td style="text-align:right;">
24
</td>
<td style="text-align:right;">
7
</td>
<td style="text-align:right;">
3
</td>
<td style="text-align:right;">
17
</td>
</tr>
<tr>
<td style="text-align:right;">
11
</td>
<td style="text-align:left;">
Espanyol
</td>
<td style="text-align:right;">
1452
</td>
<td style="text-align:right;">
17
</td>
<td style="text-align:right;">
16
</td>
<td style="text-align:right;">
-5
</td>
<td style="text-align:right;">
16
</td>
</tr>
<tr>
<td style="text-align:right;">
12
</td>
<td style="text-align:left;">
Leganes
</td>
<td style="text-align:right;">
1441
</td>
<td style="text-align:right;">
21
</td>
<td style="text-align:right;">
11
</td>
<td style="text-align:right;">
1
</td>
<td style="text-align:right;">
16
</td>
</tr>
<tr>
<td style="text-align:right;">
13
</td>
<td style="text-align:left;">
Alaves
</td>
<td style="text-align:right;">
1417
</td>
<td style="text-align:right;">
15
</td>
<td style="text-align:right;">
17
</td>
<td style="text-align:right;">
-4
</td>
<td style="text-align:right;">
17
</td>
</tr>
<tr>
<td style="text-align:right;">
14
</td>
<td style="text-align:left;">
Getafe
</td>
<td style="text-align:right;">
1415
</td>
<td style="text-align:right;">
23
</td>
<td style="text-align:right;">
8
</td>
<td style="text-align:right;">
6
</td>
<td style="text-align:right;">
17
</td>
</tr>
<tr>
<td style="text-align:right;">
15
</td>
<td style="text-align:left;">
Malaga
</td>
<td style="text-align:right;">
1396
</td>
<td style="text-align:right;">
11
</td>
<td style="text-align:right;">
19
</td>
<td style="text-align:right;">
-4
</td>
<td style="text-align:right;">
17
</td>
</tr>
<tr>
<td style="text-align:right;">
16
</td>
<td style="text-align:left;">
Real Betis
</td>
<td style="text-align:right;">
1390
</td>
<td style="text-align:right;">
21
</td>
<td style="text-align:right;">
11
</td>
<td style="text-align:right;">
5
</td>
<td style="text-align:right;">
16
</td>
</tr>
<tr>
<td style="text-align:right;">
17
</td>
<td style="text-align:left;">
Girona
</td>
<td style="text-align:right;">
1385
</td>
<td style="text-align:right;">
23
</td>
<td style="text-align:right;">
8
</td>
<td style="text-align:right;">
9
</td>
<td style="text-align:right;">
17
</td>
</tr>
<tr>
<td style="text-align:right;">
18
</td>
<td style="text-align:left;">
La Coruna
</td>
<td style="text-align:right;">
1351
</td>
<td style="text-align:right;">
12
</td>
<td style="text-align:right;">
18
</td>
<td style="text-align:right;">
0
</td>
<td style="text-align:right;">
16
</td>
</tr>
<tr>
<td style="text-align:right;">
19
</td>
<td style="text-align:left;">
Levante
</td>
<td style="text-align:right;">
1350
</td>
<td style="text-align:right;">
18
</td>
<td style="text-align:right;">
15
</td>
<td style="text-align:right;">
4
</td>
<td style="text-align:right;">
17
</td>
</tr>
<tr>
<td style="text-align:right;">
20
</td>
<td style="text-align:left;">
Las Palmas
</td>
<td style="text-align:right;">
1323
</td>
<td style="text-align:right;">
11
</td>
<td style="text-align:right;">
19
</td>
<td style="text-align:right;">
1
</td>
<td style="text-align:right;">
16
</td>
</tr>
</tbody>
</table>

### Serie A

Serie A is characterized by 2 dominant teams, Juventus and Napoli, who
only have 3 Elo point separating them (and 1 point). Roma also look
strong but the gap to the 4th place team is 120 points. The standard
deviation of Elo for Serie A is 162, which is the highest of the
European leagues, suggesting that there is greater variability in team
skill.

There are a number of over-performers, such as Sampdoria who are placed
in 6th but have the 10th highest Elo, but interestingly very few
under-performers, with no team being rated 2 positions better by Elo
than their actual standing. I'm also amazed to see Benevento having picked up a solitary point from 18 games, well deserving of the lowest Elo score across all 4 leagues. This doesn't necessarily mean that Benevento are the worst team in these leagues, but they are the furthest from their league's average.

<table class="table table-striped table-hover table-condensed table-responsive" style="width: auto !important; margin-left: auto; margin-right: auto;">
<thead>
<tr>
<th style="text-align:right;">
Elo rank
</th>
<th style="text-align:left;">
Team
</th>
<th style="text-align:right;">
Elo
</th>
<th style="text-align:right;">
Points
</th>
<th style="text-align:right;">
Points rank
</th>
<th style="text-align:right;">
Rank difference
</th>
<th style="text-align:right;">
Played
</th>
</tr>
</thead>
<tbody>
<tr>
<td style="text-align:right;">
1
</td>
<td style="text-align:left;">
Juventus
</td>
<td style="text-align:right;">
1803
</td>
<td style="text-align:right;">
44
</td>
<td style="text-align:right;">
2
</td>
<td style="text-align:right;">
-1
</td>
<td style="text-align:right;">
18
</td>
</tr>
<tr>
<td style="text-align:right;">
2
</td>
<td style="text-align:left;">
Napoli
</td>
<td style="text-align:right;">
1800
</td>
<td style="text-align:right;">
45
</td>
<td style="text-align:right;">
1
</td>
<td style="text-align:right;">
1
</td>
<td style="text-align:right;">
18
</td>
</tr>
<tr>
<td style="text-align:right;">
3
</td>
<td style="text-align:left;">
Roma
</td>
<td style="text-align:right;">
1753
</td>
<td style="text-align:right;">
38
</td>
<td style="text-align:right;">
3
</td>
<td style="text-align:right;">
0
</td>
<td style="text-align:right;">
17
</td>
</tr>
<tr>
<td style="text-align:right;">
4
</td>
<td style="text-align:left;">
Inter
</td>
<td style="text-align:right;">
1633
</td>
<td style="text-align:right;">
37
</td>
<td style="text-align:right;">
4
</td>
<td style="text-align:right;">
0
</td>
<td style="text-align:right;">
17
</td>
</tr>
<tr>
<td style="text-align:right;">
5
</td>
<td style="text-align:left;">
Lazio
</td>
<td style="text-align:right;">
1619
</td>
<td style="text-align:right;">
36
</td>
<td style="text-align:right;">
5
</td>
<td style="text-align:right;">
0
</td>
<td style="text-align:right;">
17
</td>
</tr>
<tr>
<td style="text-align:right;">
6
</td>
<td style="text-align:left;">
Atalanta
</td>
<td style="text-align:right;">
1592
</td>
<td style="text-align:right;">
27
</td>
<td style="text-align:right;">
6
</td>
<td style="text-align:right;">
0
</td>
<td style="text-align:right;">
18
</td>
</tr>
<tr>
<td style="text-align:right;">
7
</td>
<td style="text-align:left;">
Fiorentina
</td>
<td style="text-align:right;">
1566
</td>
<td style="text-align:right;">
26
</td>
<td style="text-align:right;">
8
</td>
<td style="text-align:right;">
-1
</td>
<td style="text-align:right;">
18
</td>
</tr>
<tr>
<td style="text-align:right;">
8
</td>
<td style="text-align:left;">
Torino
</td>
<td style="text-align:right;">
1521
</td>
<td style="text-align:right;">
24
</td>
<td style="text-align:right;">
9
</td>
<td style="text-align:right;">
-1
</td>
<td style="text-align:right;">
18
</td>
</tr>
<tr>
<td style="text-align:right;">
9
</td>
<td style="text-align:left;">
Milan
</td>
<td style="text-align:right;">
1496
</td>
<td style="text-align:right;">
24
</td>
<td style="text-align:right;">
9
</td>
<td style="text-align:right;">
0
</td>
<td style="text-align:right;">
18
</td>
</tr>
<tr>
<td style="text-align:right;">
10
</td>
<td style="text-align:left;">
Sampdoria
</td>
<td style="text-align:right;">
1490
</td>
<td style="text-align:right;">
27
</td>
<td style="text-align:right;">
6
</td>
<td style="text-align:right;">
4
</td>
<td style="text-align:right;">
17
</td>
</tr>
<tr>
<td style="text-align:right;">
11
</td>
<td style="text-align:left;">
Udinese
</td>
<td style="text-align:right;">
1478
</td>
<td style="text-align:right;">
24
</td>
<td style="text-align:right;">
9
</td>
<td style="text-align:right;">
2
</td>
<td style="text-align:right;">
17
</td>
</tr>
<tr>
<td style="text-align:right;">
12
</td>
<td style="text-align:left;">
Bologna
</td>
<td style="text-align:right;">
1438
</td>
<td style="text-align:right;">
24
</td>
<td style="text-align:right;">
9
</td>
<td style="text-align:right;">
3
</td>
<td style="text-align:right;">
18
</td>
</tr>
<tr>
<td style="text-align:right;">
13
</td>
<td style="text-align:left;">
Sassuolo
</td>
<td style="text-align:right;">
1425
</td>
<td style="text-align:right;">
20
</td>
<td style="text-align:right;">
14
</td>
<td style="text-align:right;">
-1
</td>
<td style="text-align:right;">
18
</td>
</tr>
<tr>
<td style="text-align:right;">
14
</td>
<td style="text-align:left;">
Chievo
</td>
<td style="text-align:right;">
1412
</td>
<td style="text-align:right;">
21
</td>
<td style="text-align:right;">
13
</td>
<td style="text-align:right;">
1
</td>
<td style="text-align:right;">
18
</td>
</tr>
<tr>
<td style="text-align:right;">
15
</td>
<td style="text-align:left;">
Genoa
</td>
<td style="text-align:right;">
1388
</td>
<td style="text-align:right;">
17
</td>
<td style="text-align:right;">
15
</td>
<td style="text-align:right;">
0
</td>
<td style="text-align:right;">
18
</td>
</tr>
<tr>
<td style="text-align:right;">
16
</td>
<td style="text-align:left;">
Cagliari
</td>
<td style="text-align:right;">
1380
</td>
<td style="text-align:right;">
17
</td>
<td style="text-align:right;">
15
</td>
<td style="text-align:right;">
1
</td>
<td style="text-align:right;">
18
</td>
</tr>
<tr>
<td style="text-align:right;">
17
</td>
<td style="text-align:left;">
Crotone
</td>
<td style="text-align:right;">
1338
</td>
<td style="text-align:right;">
15
</td>
<td style="text-align:right;">
17
</td>
<td style="text-align:right;">
0
</td>
<td style="text-align:right;">
18
</td>
</tr>
<tr>
<td style="text-align:right;">
18
</td>
<td style="text-align:left;">
SPAL
</td>
<td style="text-align:right;">
1327
</td>
<td style="text-align:right;">
15
</td>
<td style="text-align:right;">
17
</td>
<td style="text-align:right;">
1
</td>
<td style="text-align:right;">
18
</td>
</tr>
<tr>
<td style="text-align:right;">
19
</td>
<td style="text-align:left;">
Verona
</td>
<td style="text-align:right;">
1322
</td>
<td style="text-align:right;">
13
</td>
<td style="text-align:right;">
19
</td>
<td style="text-align:right;">
0
</td>
<td style="text-align:right;">
17
</td>
</tr>
<tr>
<td style="text-align:right;">
20
</td>
<td style="text-align:left;">
Benevento
</td>
<td style="text-align:right;">
1222
</td>
<td style="text-align:right;">
1
</td>
<td style="text-align:right;">
20
</td>
<td style="text-align:right;">
0
</td>
<td style="text-align:right;">
18
</td>
</tr>
</tbody>
</table>

### Bundesliga

The German league looks to be the most competitive, with the leaders
having the lowest Elo of these 4 leagues and bottom-placed team having
the highest. This is reflected in the standard deviation of Elo at 107,
far lower than the other leagues. A similar finding has been [identifed
previously](http://stuartlacy.co.uk/2016/07/23/is-la-liga-the-most-predictable-european-football-league/),
where the bookies were less accurate at predicting Bundesliga matches
than the 3 other leagues.

<table class="table table-striped table-hover table-condensed table-responsive" style="width: auto !important; margin-left: auto; margin-right: auto;">
<thead>
<tr>
<th style="text-align:right;">
Elo rank
</th>
<th style="text-align:left;">
Team
</th>
<th style="text-align:right;">
Elo
</th>
<th style="text-align:right;">
Points
</th>
<th style="text-align:right;">
Points rank
</th>
<th style="text-align:right;">
Rank difference
</th>
<th style="text-align:right;">
Played
</th>
</tr>
</thead>
<tbody>
<tr>
<td style="text-align:right;">
1
</td>
<td style="text-align:left;">
Bayern Munich
</td>
<td style="text-align:right;">
1793
</td>
<td style="text-align:right;">
41
</td>
<td style="text-align:right;">
1
</td>
<td style="text-align:right;">
0
</td>
<td style="text-align:right;">
17
</td>
</tr>
<tr>
<td style="text-align:right;">
2
</td>
<td style="text-align:left;">
Borussia Dortmund
</td>
<td style="text-align:right;">
1626
</td>
<td style="text-align:right;">
28
</td>
<td style="text-align:right;">
3
</td>
<td style="text-align:right;">
-1
</td>
<td style="text-align:right;">
17
</td>
</tr>
<tr>
<td style="text-align:right;">
3
</td>
<td style="text-align:left;">
Bayern Leverkusen
</td>
<td style="text-align:right;">
1575
</td>
<td style="text-align:right;">
28
</td>
<td style="text-align:right;">
3
</td>
<td style="text-align:right;">
0
</td>
<td style="text-align:right;">
17
</td>
</tr>
<tr>
<td style="text-align:right;">
4
</td>
<td style="text-align:left;">
Hoffenheim
</td>
<td style="text-align:right;">
1558
</td>
<td style="text-align:right;">
26
</td>
<td style="text-align:right;">
7
</td>
<td style="text-align:right;">
-3
</td>
<td style="text-align:right;">
17
</td>
</tr>
<tr>
<td style="text-align:right;">
5
</td>
<td style="text-align:left;">
Schalke
</td>
<td style="text-align:right;">
1553
</td>
<td style="text-align:right;">
30
</td>
<td style="text-align:right;">
2
</td>
<td style="text-align:right;">
3
</td>
<td style="text-align:right;">
17
</td>
</tr>
<tr>
<td style="text-align:right;">
6
</td>
<td style="text-align:left;">
Leipzig
</td>
<td style="text-align:right;">
1550
</td>
<td style="text-align:right;">
28
</td>
<td style="text-align:right;">
3
</td>
<td style="text-align:right;">
3
</td>
<td style="text-align:right;">
17
</td>
</tr>
<tr>
<td style="text-align:right;">
7
</td>
<td style="text-align:left;">
Borussia Moenchengladbach
</td>
<td style="text-align:right;">
1525
</td>
<td style="text-align:right;">
28
</td>
<td style="text-align:right;">
3
</td>
<td style="text-align:right;">
4
</td>
<td style="text-align:right;">
17
</td>
</tr>
<tr>
<td style="text-align:right;">
8
</td>
<td style="text-align:left;">
Augsburg
</td>
<td style="text-align:right;">
1498
</td>
<td style="text-align:right;">
24
</td>
<td style="text-align:right;">
9
</td>
<td style="text-align:right;">
-1
</td>
<td style="text-align:right;">
17
</td>
</tr>
<tr>
<td style="text-align:right;">
9
</td>
<td style="text-align:left;">
Ein Frankfurt
</td>
<td style="text-align:right;">
1477
</td>
<td style="text-align:right;">
26
</td>
<td style="text-align:right;">
7
</td>
<td style="text-align:right;">
2
</td>
<td style="text-align:right;">
17
</td>
</tr>
<tr>
<td style="text-align:right;">
10
</td>
<td style="text-align:left;">
Hertha
</td>
<td style="text-align:right;">
1474
</td>
<td style="text-align:right;">
24
</td>
<td style="text-align:right;">
9
</td>
<td style="text-align:right;">
1
</td>
<td style="text-align:right;">
17
</td>
</tr>
<tr>
<td style="text-align:right;">
11
</td>
<td style="text-align:left;">
Wolfsburg
</td>
<td style="text-align:right;">
1466
</td>
<td style="text-align:right;">
19
</td>
<td style="text-align:right;">
12
</td>
<td style="text-align:right;">
-1
</td>
<td style="text-align:right;">
17
</td>
</tr>
<tr>
<td style="text-align:right;">
12
</td>
<td style="text-align:left;">
Werder Bremen
</td>
<td style="text-align:right;">
1451
</td>
<td style="text-align:right;">
15
</td>
<td style="text-align:right;">
16
</td>
<td style="text-align:right;">
-4
</td>
<td style="text-align:right;">
17
</td>
</tr>
<tr>
<td style="text-align:right;">
13
</td>
<td style="text-align:left;">
Mainz
</td>
<td style="text-align:right;">
1414
</td>
<td style="text-align:right;">
17
</td>
<td style="text-align:right;">
14
</td>
<td style="text-align:right;">
-1
</td>
<td style="text-align:right;">
17
</td>
</tr>
<tr>
<td style="text-align:right;">
14
</td>
<td style="text-align:left;">
Freiburg
</td>
<td style="text-align:right;">
1411
</td>
<td style="text-align:right;">
19
</td>
<td style="text-align:right;">
12
</td>
<td style="text-align:right;">
2
</td>
<td style="text-align:right;">
17
</td>
</tr>
<tr>
<td style="text-align:right;">
15
</td>
<td style="text-align:left;">
Hannover
</td>
<td style="text-align:right;">
1393
</td>
<td style="text-align:right;">
23
</td>
<td style="text-align:right;">
11
</td>
<td style="text-align:right;">
4
</td>
<td style="text-align:right;">
17
</td>
</tr>
<tr>
<td style="text-align:right;">
16
</td>
<td style="text-align:left;">
Hamburg
</td>
<td style="text-align:right;">
1375
</td>
<td style="text-align:right;">
15
</td>
<td style="text-align:right;">
16
</td>
<td style="text-align:right;">
0
</td>
<td style="text-align:right;">
17
</td>
</tr>
<tr>
<td style="text-align:right;">
16
</td>
<td style="text-align:left;">
Koln
</td>
<td style="text-align:right;">
1375
</td>
<td style="text-align:right;">
6
</td>
<td style="text-align:right;">
18
</td>
<td style="text-align:right;">
-2
</td>
<td style="text-align:right;">
17
</td>
</tr>
<tr>
<td style="text-align:right;">
18
</td>
<td style="text-align:left;">
Stuttgart
</td>
<td style="text-align:right;">
1370
</td>
<td style="text-align:right;">
17
</td>
<td style="text-align:right;">
14
</td>
<td style="text-align:right;">
4
</td>
<td style="text-align:right;">
17
</td>
</tr>
</tbody>
</table>