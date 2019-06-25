+++
date = 2019-06-24
draft = false
tags = ["data", "society", "R"]
title = "Is the Andrew Marr show biased towards women?"
math = false
+++

I came across [a tweet](https://twitter.com/piersmorgan/status/114272772560771891) from Piers Morgan this morning in which he suggested that the BBC is favouring women since 43 out of the 53 paper reviewers on The Andrew Marr Show in 2019 were women.
Unfortunately I was a day late to this hot-take, fortunately this is because I don't follow Piers Morgan. 
However, I knew that there must be more to it than a single PC-baiting statistic and knowing that I had a ~3 hour train journey coming up this evening I thought I'd look into it a bit more.
Spoiler alert: [Betteridge's law](https://en.wikipedia.org/wiki/Betteridge%27s_law_of_headlines) is true once again.

I firstly scraped and cleaned the full list of guests on The Andrew Marr Show from 2012 to the current day; I've put the [code and clean data on GitHub](https://github.com/stulacy/andrewmarrequality) in case others want to use this dataset.

Looking at the overall proportion of female guests and the BBC does seem to have made an effort to strive for equal representation over the last 7 years (using a moving average filter with a window size of 4 weeks).
Back in 2012 only just over a quarter of Andrew Marr's guests were women, this figure is nearly half today.
![Overall proportion of guests that are women](/img/andrewmarrequality-20190624/overall_proportion.png)

However, Piers Morgan was suggesting that the BBC is biased in favour of women, how can that be the case?
Well, the Andrew Marr Show website helpfully indicates who is doing the papers review. 
If we plot the same values as before but this time separating guests into either Main Guests or Newspaper Reviewers, we see a markedly different story.
While yes, since 2015 women have been over represented on average in the newspaper review section, this pales in comparison with the main guests where even now women only make up around 30% of the invitees, and the flat trend hardly suggests this will change anytime soon.

![Proportion of guests that are women grouped by show role](/img/andrewmarrequality-20190624/stratified_proportion.png)

The most obvious explanation for this difference is that these two groups represent different populations. I.e. journalists will often be invited on to review the newspapers, while the main guests are typically those in higher levels of power, such as those involved in politics where women are less likely to reach the highest ranks (32% of all MPs are female as of 2017).

Of course, further research would be needed to thoroughly investigate such claims, but I hope this brief post has highlighted the dangers of summarising a large amount of data from a complex issue with many different factors into a single statistic.
I could easily produce my own cherry-picked stats here, such as the fact that out of 334 shows in this time-frame there has only been 1 with an all-female main lineup versus 93 all-male, but in general it's best to view as much raw data as possible to place any figures in context.

