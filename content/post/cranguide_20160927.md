+++
date = 2016-09-27
draft = false
tags = ["CRAN", "R", "software development"]
title = "Guide to publishing R packages on CRAN"
math = false
+++

I recently give a talk at my university's R User group on how to publish packages to CRAN ([slides here](/downloads/cranslides_handout.pdf)). This isn't an easy topic to distill into a 60 minute slot, and so I had to abandon my original idea of a hands on workshop with examples in favour of a condensed summary of the main challenges in the submission process. This mostly focused on the issue of Namespaces, since this is a rather complex topic to understand if you're coming from a non-software engineering background, as it doesn't come up in day-to-day statistical analysis. I also provided some motivation for publishing to CRAN in the first place, surprising myself with how many reasons there are! If you're interested in publishing at all, or are just new to programming in R I'd strongly recommened having a look through to understand the CRAN ecosystem a bit better, although bear in mind this talk was aimed at an academic audience of statisticians, rather than software engineers. Once you've had a look at the slides, Hadley Wickham's written a fantastic book covering all relevant topics in depth, which I used as a constant reference. It's available online [here](http://r-pkgs.had.co.nz/).
