+++
date = 2015-01-06
draft = false
tags = ["academia", "R"]
title = "Dplyr incompatibility with xtable"
math = false
+++

I've been working on another paper today and decided to update my previous `xtable` function (as described <a href="http://stuartlacy.co.uk/20141217-sweavetables">here</a>) to use `dplyr`, as I  want to fully get to grips with Hadley Wickham's wonderful ecosystem of packages including `dplyr` (and its predecessor `plyr`), `ggplot2` and `tidyr` (and its predecessor `reshape2`). I <a href="http://stuartlacy.co.uk/20141217-dplyr">mentioned</a> this before Christmas but have only got round to it now, which included a few hours of struggling with tidyr to make it do what I want! However in updating my function to use dplyr's `summarise` function instead of `aggregate` I came across an odd bug that got me stuck for an hour or so.

Let's start off by making an example dataframe with the accuracies of three different classification algorithms on 3 standard <a href="http://archive.ics.uci.edu/ml/">UCI</a> datasets, with each algorithm being run on each dataset 5 times to account for the stochastic nature of these models.

```r
library(dplyr)
library(xtable)
df.cls <- data.frame(dataset=rep(c("Iris", "Heart", "Liver")),
                             algorithm=rep(c("ANN", "GP", "CGP"), each=15),
                             run=rep(seq(5), each=3),
                            accuracy=runif(45))
```

Then using dplyr we can easily form a new dataframe with the mean accuracy from each run for each algorithm on each dataset.

```r
results <- df.cls %>%
                   group_by(dataset, algorithm) %>%
                   summarise(mean=mean(accuracy))
```

The problem comes when we try to form an xtable from this dataframe, to be used in LaTeX documents.

```r
print(xtable(results))
```

This will produce the following error: `Error in .subset2(x, i, exact = exact) : subscript out of bounds`.
I spent at least an hour trying to debug this, thinking the problem was due to my processing with dplyr as I was chaining together more than 5 different functions which I was not familiar with. I eventually came across a [bug report on Github](https://github.com/hadley/dplyr/issues/656) which described the same problem. The person indicated their problem was solved upon updating to dplyr 0.3.0.9, although I prefer to stick to CRAN releases for stability (currently on 0.3.0.2).

To solve it simply cast the object as a dataframe in the xtable call, as such:

```r
print(xtable(as.data.frame(results)))
```

Simply using `data.frame` (without the `as.`) will work but has some odd effects, such as spaces in column names being turned into full stops, presumably to make it 'correct' R syntax. I hope anyone who encounters the same issue comes across this blog straight away and saves themselves the hassle I went through!
