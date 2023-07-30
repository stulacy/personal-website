+++
date = 2023-06-28
draft = false
tags = ["R", "data science"]
title = "Speeding up R workshop"
math = false
+++

Just a quick update, more to test that the website infrastructure is still running than anything else, as it's been 4 years since my last post. 
I ran a workshop ([slides here](stuartlacy.uk/speeding-up-R-workship-2023-06-28.html)) at the [University's Research Coding Club](https://researchcodingclub.github.io/about/) on speeding up data analysis in R last month that might be useful for anyone who stumbles across this page in the future.

The Research Coding Club is an informal collective of people from across the entire University who write software to aid their research.
Its main purpose is to provide a forum for getting support with specific software-related problems, which takes place both virtually and also with in-person drop-in sessions, although every month we host guest talks about a variety of subjects, often related to training general skills (Linux, HPC, Git etc...).

I delivered the talk last month as I was keen for the opportunity to share some general tips for performing data analysis in R based on certain bad practices I'd observed from beginners to the language. 
Since I only had a 1 hour slot I tied these ideas together into a single talk loosely organised around speeding up R, although since it was exam time there were no free seminar rooms and I had to run the session as a talk rather than my preferred medium of an interactive workshop.
I say loosely because the talk was less concerned with micro-optimisations to eke out every last drop of speed from a complex simulation, but instead identifying certain low-hanging fruit to streamline day-to-day routine data science work, both in terms of the code performance but also its legibility.

The talk covered several areas, each of which I ideally would have had a full hour to fully explore:

  - Vectorisation - what vectorisation is, the importance of knowing the fast vectorised functions in the standard library, why the `Vectorize` function isn't a silver bullet, and introducing the motivation for `Rcpp`
  - Joins - Joins might not be traditionally thought of as providing speed benefits, but I wanted to show that they are extremely useful in many situations beyond the basic use case of joining together 2 independent datasets based on a common key, and can provide some very neat (and fast!) solutions to certain problems (especially now that non-equi joins are available in the `tidyverse`)
  - `data.table` - Introducing its basic syntax, motivation for using it, and showing the two `tidyverse`-style interface libraries of `tidytable` and `dtplyr`
  - `duckdb` and `RSQLite` - Demonstrating their use for larger-than-memory datasets
  - `Rcpp` - Identifying situations when it's useful (i.e. when a for-loop is unavoidable) and a brief introduction to its data structures and use

Unfortunately I think I was being too ambitious in trying to cover so many areas in 1 hour and I ended up racing through, but I hope to have at least made people aware of some of these areas and to be able to refer back to the slides to get the ball rolling, in say using joins to replace a slow for-loop that combines two datasets based on time-intervals.
I also benefited from finally getting some motivation to try `duckdb` which has now become a staple tool in my kit, even on relatively small datasets.
I like quantitative comparisons and was really interested to see my [real-world benchmarks](stuartlacy.co.uk/speeding-up-R-workship-2023-06-28.html#/overall-benchmark) showing that `data.table` can ran ~20x faster than standard `tidyverse` functions for certain routine tasks, and I was particularly impressed that the additional overhead costs of `tidytable` and `dtplyr` didn't stop them from achieving ~10x speedups.
However, when I increased the dataset size to 5 million rows these gaps decreased and [`duckdb` stormed into the lead](stuartlacy.uk/speeding-up-R-workship-2023-06-28.html#/benchmark---all-5-million-rows).

The code and slides are on [GitHub](https://github.com/stulacy/speeding-up-R-workshop/), and I've uploaded the [slides here](stuartlacy.uk/speeding-up-R-workship-2023-06-28.html) too.
