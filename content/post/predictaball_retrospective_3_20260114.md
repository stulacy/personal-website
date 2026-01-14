+++
date = 2026-01-14
draft = false
tags = ["Predictaball", "Evolutionary algorithms"]
title = "Predictaball retrospective part 3 - Using evolution to predict football matches"
math = true
+++

<link rel="stylesheet" href="/css/quarto.css" />

# Introduction

The Elo models introduced [last time](https://stuartlacy.uk/2026/01/09/predictaball-retrospective-part-2-elo-rating-system/), which were the models used on [Predictaball](predictaball.net) from 2017-2019, worked very well but with some limitations.
Firstly, the parameters used in the rating update equation (home advantage and margin of victory multiplier) were chosen manually by inspection for each league.
If I wanted to apply Predictaball to a whole new set of leagues (yet alone sports!) I'd need to go through each one to identify new parameters - hardly ideal!
Furthermore, it uses a hand-crafted rating update equation that was selected because it fitted with inutition of how the rating update should be performed, rather than being driven by the data.
In this post I'll talk through how I optimised the parameters for a joint rating-prediction model instead.
The resulting model has been used as the default Predictaball model since 2019 (not necessarily because it's an optimal model, but life has gotten in the way since then).


<div class="cell">
<details class="code-fold">
<summary>
  Show the code
</summary>
<div class="code-copy-outer-scaffold">
<pre>
<code class="language-r hljs">
library(tidyverse)
library(knitr)        # Displaying tables
library(kableExtra)   # Displaying tables
library(ggrepel)      # Plotting
library(ggridges)     # Plotting
library(latex2exp)    # Math-mode in plots
library(patchwork)    # Combining plots
library(Rcpp)         # Interface with C++
library(tictoc)       # timing
options(knitr.table.html.attr = "quarto-disable-processing=true")
</code>
</pre>
</div>
</details>
</div>


# Model

We want to use a similar model structure to [last time](https://stuartlacy.uk/2026/01/09/predictaball-retrospective-part-2-elo-rating-system/), but with free parameters that can be estimated from the data instead.
The match prediction model will largely remain the same as an Ordinal Logistic Regression, although the league-specific intercept that allows for modelling differing home advantage levels across leagues will be moved into $\phi$ rather than having league-specific cut-points $\kappa$, as this reduces the number of parameters to estimate (1 $\alpha$ per league rather than 2 $\kappa$).

$$\text{result}\_i \sim \text{Ordered-logit}(\phi\_i,k\_i)$$

$$\phi\_i = \alpha\_\text{league[i]} + \beta\_1 (\psi\_\text{home[i]} - \psi\_\text{away[i]})$$

$$k\_i = \kappa$$

The rating update function uses the same terms as Elo but is parameterized by $K$ - which can vary by league - and $\beta\_2, \beta\_3$ in $G(x)$. 
Previously $K$ has been held constant at 20 across leagues so it will be interesting to see if it changes between leagues when estimated from the data.
$\beta\_2$ and $\beta\_3$ scale the margin of victory multiplier, which is now a negative exponential rather than $\log\_2$. This is to allow a maximum possible multiplier (governed by $\beta\_2$), while $\beta\_3$ governs the rate of increase of the multiplier with each additional goal.

$$\delta = K\_\text{league}G(\Delta)OE$$

$$\Delta = \text{goals}\_\text{home} - \text{goals}\_\text{away}$$
$$G(x) = \beta\_2 (1-\exp(-\beta\_3 x))+1$$

Therefore, there are 13 parameters (7 for the match prediction ordinal logistic regression and 6 for the rating update), as displayed below.

$$\Theta = \{
    \alpha\_\text{league[1]},
    \alpha\_\text{league[2]},
    \alpha\_\text{league[3]},
    \alpha\_\text{league[4]},
    \beta\_1,
    \kappa\_{[1]},
    \kappa\_{[2]},
    K\_\text{league[1]},
    K\_\text{league[2]},
    K\_\text{league[3]},
    K\_\text{league[4]},
    \beta\_2,
    \beta\_3
\}$$

# Evolutionary Algorithms

## History lesson

Evolutionary algorithms (EA) are a field of general optimization algorithms that place very little assumptions on the function being optimized. They don't require gradients or assume anything about the output distribution, instead all they require is some way of ranking candidate solutions.
They were part of a wave of bio-inspired computing approaches in the 1970s, particularly during the AI winter as an alternative to neural networks (another bio-inspired computation method).
EAs take inspiration from evolutionary processes by maintaining a pool (a 'population' in EA speak) of randomly initialized candidate solutions and iteratively selecting 'parents' according to their 'fitness' (success at the optimization task) and 'breeding' children through several operators including 'crossover' and 'mutation' for a number of iterations ('generations').
Candidate solutions are represented by their 'genotype', which can be as simple as a vector of parameters, as well as a 'phenotype', which is the behaviour of that genotype.
I.e. in this example the genotype is 13 floating point numbers ($\Theta$) and the phenotype is the match prediction and rating update equations above.
In a different problem the exact same genotype of 13 floats could correspond to entirely different behaviour.

There are various flavours of EAs:

  - Genetic Algorithms (GAs): the most basic example with a straight forward often integer-valued genotype and usually only uses mutation
  - Evolutionary Strategies (ESs): very similar to GAs, except that the mutation properties are part of an individual's genotype, rather than being external
  - Genetic Programming (GP): the genotype encodes a program and as such is much more varied than that in a GA or ES, as the values map to function sets and more complex operators. Typically tree-based programs are used, but a graph-based method called [Cartesian Genetic Programming (CGP)](https://cs.ijs.si/ppsn2014/files/slides/ppsn2014-tutorial3-miller.pdf) was pioneered at my old research group

I used an EA for this application as firstly they are quick to implement and require little hyper-parameter optimization to get 'good-enough' results. 
They don't necessarily converge to a local optima the way a gradient approach can, but are good at exploring the full solution space. I did intend to follow this up with an neural-network based method optimized via backpropagation, but again life got in the way... Although this is still on my radar now I'm starting to get a bit more free time so watch this space.

## Fitness function

The fitness function is key to an EA as it defines how to rank the candidate solutions.
It is a function that takes as argument a candidate genotype and returns a score of goodness-of-fit according to the phenotype.
The actual score itself doesn't matter, only the ranking.

In this instance, the fitness function will run through multiple seasons from multiple leagues using the rating update and match prediction equations with a given candidate set $\Theta$ and return a single score evaluating how accurate the predictions are.
In particular, the score is the sum of the RPS of each match prediction from all leagues and seasons.
The fitness function therefore guides towards a solution that maximizes predictive accuracy, rather than having an explicit objectives for the rating system.
The source code (in `C++` for speed) is shown below - it's rather long, mostly due to munging data structures.

That's the general idea anyway. Some more technical subtleties are that the RPS isn't accumulated over the first 3 seasons to allow time for the team ratings to settle.
The team ratings themselves all start at 2,000 rather than the 1,000 used for Elo.
However, teams that aren't in the league at the start of the dataset (2005-2006 season) but who later join via promotion are handled differently.
Rather than assume that these teams are 'average', which should correspond to a mid-table team, these teams are expected to be more on a par with the lower ranked teams.
Therefore the newly promoted teams to a league are given the average rating of the teams that have been relegated.
Furthermore, given that a fair amount can change over the summer in terms of transfers and coaching staff changeover, not to mention that ratings can inflate towards the end of a season when there's less to play for (for many teams if not all), each team's rating is dialled back by 20% at the end of a season.


<div class="cell">
<details class="code-fold">
<summary>
  Show the code
</summary>
<div class="code-copy-outer-scaffold">
<pre>
<code class="language-cpp hljs">
#include <Rcpp.h>
#include <cmath>
#include <vector>

using namespace Rcpp;

double sigmoid(double x) {
    return 1 / (1 + exp(-x));
}

double rps(double phome, double pdraw, int home, int draw) {
    double home_cont, draw_cont, full;
    home_cont = pow(phome - home, 2);
    draw_cont = pow(phome - home + pdraw - draw, 2);
    full = 0.5 * (home_cont + draw_cont);
    return full;
}

std::tuple<double, double, double> prediction_network(NumericVector params, int elo_diff, 
                                                      IntegerVector leagues) {
    
        double mu, q1, q2;
        mu = params[0] * leagues[0] + params[1] * leagues[1] + params[2] * leagues[2] + params[3] * leagues[3] + params[4] * elo_diff;
        
        q1 = sigmoid(params[5] - mu);
        q2 = sigmoid(params[6] - mu);
        
        double paway = q1;
        double pdraw = q2 - q1;
        double phome = 1 - q2;
        
        return std::make_tuple(paway, pdraw, phome);
}

double rating_network(NumericVector params, std::tuple<double, double, double> probs, 
                   IntegerVector leagues, int mov, IntegerVector outcomes) {
    double phome = std::get<2>(probs);
    double pdraw = std::get<1>(probs);
    double paway = std::get<0>(probs);
    double K = params[7] * leagues[0] + params[8] * leagues[1] + params[9] * leagues[2] + params[10] * leagues[3];
    double O = outcomes[0] * 0 + outcomes[1] * 0.5 + outcomes[2] * 1;
    double E = ((paway + 2 * pdraw + 3 * phome) - 1) / 2;
    double mov_multiplier = params[11]*(1-exp(-params[12]*mov))+1;
    double output = mov_multiplier * K * (O - E);
    
    return output;
}

// [[Rcpp::export]]
double update_ratings(NumericVector params, NumericVector probs, IntegerVector outcomes,
                   IntegerVector leagues, IntegerVector mov) {
    
    std::tuple<double, double, double> probs_tup;
    std::get<0>(probs_tup) = probs[0];
    std::get<1>(probs_tup) = probs[1];
    std::get<2>(probs_tup) = probs[2];
    return rating_network(params, probs_tup, leagues, mov[0], outcomes);
}

// [[Rcpp::export]]
NumericVector predict_match(NumericVector params, IntegerVector elo_diff, IntegerVector leagues) {
    std::tuple<double, double, double> raw_out;
    raw_out = prediction_network(params, elo_diff[0], leagues);
    
    NumericVector out_vec(3);
    out_vec(0) = std::get<0>(raw_out);
    out_vec(1) = std::get<1>(raw_out);
    out_vec(2) = std::get<2>(raw_out);
    return out_vec;
}


// [[Rcpp::export]]
List season_rps_fitnessfunction(NumericVector params,
                                  int ngames,
                                  int nteams,
                                  int nleagues,
                                  IntegerVector home_teams,
                                  IntegerVector away_teams,
                                  IntegerMatrix leagues,
                                  IntegerMatrix outcomes,
                                  IntegerVector season,
                                  int nseasons,
                                  IntegerMatrix promoted,
                                  IntegerVector mov,
                                  IntegerVector leaguemembership,
                                  double start_rating,
                                  double reset,
                                  int burnin,
                                  NumericVector initial_ratings,
                                  bool training
) {
    
    int prev_season = 0;
    int this_season;
    double cum_rps = 0;
    double match_rps, rating_change;
    int home_team, away_team;
    double rel_rating_sum, rel_rating_avg;
    IntegerVector season_col;
    int num_rel_teams;
    
    std::tuple<double, double, double> predicted_probs;
    int season_val;
    
    // Form ratings
    NumericVector ratings(nteams);
    if (training) {
        for (int j = 0; j < nteams; ++j) {
            ratings(j) = start_rating;
        }
    } else {
        for (int j = 0; j < nteams; ++j) {
            ratings(j) = initial_ratings(j);
        }
    }
    
    // Change limit to ngames to remove debug
    for (int i = 0; i < ngames; ++i) {
        //Rcpp::Rcout << "\nGame: " << i << "\n~~~~~~~~\n";
        this_season = season[i];
        if (prev_season != this_season) {
            std::vector<std::pair<int, int>> prom_teams, rel_teams;
            std::vector<int> teams_in_this_season;
            
            // Get promoted and relegated teams
            season_col = promoted(_, this_season);
            for (int j = 0; j < season_col.size(); ++j) {
                season_val = season_col[j];
                
                if (season_val == 1 || season_val == 2) {
                    teams_in_this_season.push_back(j);
                }
                if (season_val == 2) {
                    prom_teams.push_back(std::make_pair(j, leaguemembership[j]));
                } else if (season_val == 3) {
                    rel_teams.push_back(std::make_pair(j, leaguemembership[j]));
                }
            }
            
            // Update each league separately 
            for (int l=0; l < nleagues; ++l){
                num_rel_teams = 0;
                rel_rating_sum = 0;
                
                // Get mean rating of relegated team
                for (auto it : rel_teams) {
                    if (it.second == l) {
                        rel_rating_sum += ratings(it.first);
                        num_rel_teams++;
                    }
                }
                rel_rating_avg = rel_rating_sum / num_rel_teams;
                
                // Assign it to promoted teams
                for (auto it : prom_teams) {
                    if (it.second == l) {
                        ratings(it.first) = rel_rating_avg;
                    }
                }
            }
            
            // Soft-reset ratings
            for (auto it : teams_in_this_season) {
                ratings(it) = reset * ratings(it) + ((1-reset) * start_rating);
            }
            
            prev_season = this_season;
        }
        
        home_team = home_teams(i);
        away_team = away_teams(i);
        
        // Predict outcome
        predicted_probs = prediction_network(params, ratings(home_team) - ratings(away_team), 
                                             leagues(i, _));
        
        match_rps = rps(std::get<2>(predicted_probs), std::get<1>(predicted_probs), outcomes(i, 2), outcomes(i, 1));
        if (i > burnin) {
            cum_rps += match_rps;
        }
        
        // Obtain rating update
        rating_change = rating_network(params, predicted_probs, leagues(i, _), 
                                       mov[i], outcomes(i, _));
        // Update ratings, keep ratings change as positive = home team upate
        ratings(home_team) += rating_change;
        ratings(away_team) -= rating_change;
    }
    List ret;
    ret["rps"] = cum_rps;
    ret["ratings"] = ratings;
    return ret;
}

</code>
</pre>
</div>
</details>
</div>


And this fitness function can be loaded and made available to the R interpreter via the magic of `Rcpp`.


<div class="cell">
<details class="code-fold" open="">
<summary>
  Show the code
</summary>
<div class="code-copy-outer-scaffold">
<pre>
<code class="language-r hljs">
sourceCpp("fitness_function_seasonrps.cpp")

</code>
</pre>
</div>
</details>
</div>

## Preparing evolutionary strategy

I'll setup the evolutionary strategy as needed, using [CMA-ES](https://ieeexplore.ieee.org/document/6790628) because it's well supported, shown to perform well, and is written in Java so it's both fast and easy to run from R (through `rJava`). 
We need to provide a stopping criterion, here when RPS doesn't increase by more than 0.01.

Side-note: it's funny how Python is called the universal glue language while here we are using a Java evolutionary algoritm library with a C++ fitness function, all called from within R with minimal boilerplate. (No shade to Python of course!).



<div class="cell">
<details class="code-fold" open="">
<summary>
  Show the code
</summary>
<div class="code-copy-outer-scaffold">
<pre>
<code class="language-r hljs">
library(rCMA)
cmaobj <- cmaNew()
cmaSetStopTolFun(cmaobj, 0.01)  # Stop evolution when don't have 1 RPS improvement

</code>
</pre>
</div>
</details>
</div>

To help give evolution a guiding hand we can provide some initial values for the 13 values in $\Theta$.
I won't provide initial values for parameters 1-4 - the league specific intercepts $\alpha$, as they depend on $\kappa$, likewise $\beta_1$ in slot 5 depends on both $\alpha$ and $\kappa$ and is awkward to interpret at the best of times.



<div class="cell">
<details class="code-fold" open="">
<summary>
  Show the code
</summary>
<div class="code-copy-outer-scaffold">
<pre>
<code class="language-r hljs">
n_params <- 13
init_params <- rep(0, n_params)

</code>
</pre>
</div>
</details>
</div>


$\kappa$, which takes up parameter positions 6 and 7 is more straight forward, as it's just the logit transformed cumulative probabilities of an away win and draw.
From the training set across all leagues the rates of each 3 outcomes are 46.9% (H) / 25.3% (D) / 27.8% (A), so these can be used as initial values.


<div class="cell">
<details class="code-fold" open="">
<summary>
  Show the code
</summary>
<div class="code-copy-outer-scaffold">
<pre>
<code class="language-r hljs">
logit <- function(x) log(x / (1-x))
init_params[6] <- logit(0.278)
init_params[7] <- logit(0.278 + 0.253)
</code>
</pre>
</div>
</details>
</div>

We'll set K to 20 for all leagues to be consistent with the Elo model.


<div class="cell">
<details class="code-fold" open="">
<summary>
  Show the code
</summary>
<div class="code-copy-outer-scaffold">
<pre>
<code class="language-r hljs">
init_params[8:11] <- 20
</code>
</pre>
</div>
</details>
</div>

The final parameters for the rating update equations are $\beta_2$ and $\beta_3$ in $G(x)$, which govern the maximum possible multiplier and the rate of increase with goal difference respectively.
It seems reasonable to cap the multiplier at 3, and 0.3 was chosen for $\beta_3$ from inspection.

<div class="cell">
<details class="code-fold" open="">
<summary>
  Show the code
</summary>
<div class="code-copy-outer-scaffold">
<pre>
<code class="language-r hljs">
init_params[12] <- 3
init_params[13] <- 0.3
</code>
</pre>
</div>
</details>
</div>

The CMA object (because of course we're working with Java everything has to be an object) can now be initialised.


<div class="cell">
<details class="code-fold" open="">
<summary>
  Show the code
</summary>
<div class="code-copy-outer-scaffold">
<pre>
<code class="language-r hljs">
cmaInit(cmaobj, seed=3, dimension=n_params, initialX=init_params)
</code>
</pre>
</div>
</details>
</div>


## Preparing dataset

A lot of data munging needs to be done to get the tabular dataset into the various data structures needed by the fitness function, in particular:

  - Encode league as 1-hot
  - Encode match outcomes as 1-hot
  - Convert season labels into integer ids
  - Convert team names into integer ids
  - Encode team membership in each season as 0-3:
    - 0: Not played
    - 1: Played this season as well as last
    - 2: Was promoted at start of season
    - 3: Was relegated at end of last season
    
This is all carried out by the following tedious code.

<div class="cell">
<details class="code-fold">
<summary>
  Show the code
</summary>
<div class="code-copy-outer-scaffold">
<pre>
<code class="language-r hljs">
train_df <- df |>
    filter(dset == 'training')

seasons <- sort(unique(train_df$season))
seasons_ids <- as.integer(
    factor(
        train_df$season,
        levels=seasons
    )
)
nseasons <- length(seasons)

# Should be able to assume that all teams have played home and away but let's be safe
teams <- union(unique(train_df$home), unique(train_df$away))  
hometeam_ids <- as.integer(
    factor(
        train_df$home,
        levels=teams
    )
)
awayteam_ids <- as.integer(
    factor(
        train_df$away,
        levels=teams
    )
)
nteams <- length(teams)

leagues <- unique(train_df$league)
league_df <- data.frame(league=factor(train_df$league, levels=leagues))
league_mat <- model.matrix( ~ league - 1, league_df)

team_league_ids <- train_df |>
    distinct(league, team=home) |>
    mutate(
        teamid = as.integer(factor(team, levels=teams)),
        leagueid = as.integer(factor(league, levels=leagues))
    ) |>
    arrange(teamid) |>
    pull(leagueid)

outcome_mat <- as.matrix(
    model.matrix( 
        ~ outcome - 1, 
        data.frame(
            outcome = factor(
                train_df$result,
                levels=c('away', 'draw', 'home'),
                labels=c(-1, 0, 1)
            )
        )
    )
)

prom_rel_mat <- matrix(0, nrow=nteams, ncol=nseasons)
for (seas in seasons) {
    all_home <- train_df %>%
                    filter(season == seas) %>%
                    distinct(home) %>%
                    pull(home)
    all_away <- train_df %>%
                    filter(season == seas) %>%
                    distinct(away) %>% 
                    pull(away)
    all_teams <- union(all_home, all_away)
    prom_rel_mat[match(all_teams, teams), which(seas == seasons)] <- 1
    
    if (seas != '2005-2006') {
        promoted_teams <- setdiff(all_teams, prev_teams)
        relegated_teams <- setdiff(prev_teams, all_teams)
        
        # Encoding is:
        #  - 0: Not played
        #  - 1: Played this season and last season too
        #  - 2: Promoted at start of this season 
        #       (i.e. were in below division last season)
        #  - 3: Relegated at end of last season
        #       (i.e. played this season and relegated at end)
        
        prom_rel_mat[match(promoted_teams, teams), which(seas == seasons)] <- 2
        prom_rel_mat[match(relegated_teams, teams), which(seas == seasons)] <- 3
    }
    prev_teams <- all_teams
}

</code>
</pre>
</div>
</details>
</div>


## Running

We're just about ready to run the evolutionary strategy! The final issue is that the C++ fitness function takes a lot of arguments relating to the various data structures needed, however, the fitness function that CMA requires must be parameterised solely by its parameters (the chromosome being evolved). We can achive this by creating a partial function in R with arity 1 (parameters) with all the other arguments filled out.


<div class="cell">
<details class="code-fold" open="">
<summary>
  Show the code
</summary>
<div class="code-copy-outer-scaffold">
<pre>
<code class="language-r hljs">
fitfunc <- function(params) {
    season_rps_fitnessfunction(
        params = params,
        ngames= nrow(train_df),
        nteams= nteams,
        nleagues= length(leagues),
        home_teams = as.integer(hometeam_ids)-1,
        away_teams = as.integer(awayteam_ids)-1,
        leagues = league_mat,
        outcomes = outcome_mat,
        season = as.integer(seasons_ids)-1,
        promoted = prom_rel_mat,
        mov = as.integer(abs(train_df$home_score - train_df$away_score)),
        leaguemembership = as.integer(team_league_ids)-1,
        start_rating=2000,
        nseasons= nseasons, 
        reset=0.8,
        burnin= 1446 * 3,  # 3 seasons burn in
        initial_ratings = rep(2000, nteams),
        training=TRUE
    )$rps
}  
</code>
</pre>
</div>
</details>
</div>


Finally, CMA allows for constraints, here the 2 necessary constraints are $\kappa$ being ordered (Stan did this for us with its `ordered` datatype) and $\beta_2$ being positive to ensure positive $G(x)$.
And it's computer go brrr time!

This only took 2.5 minutes on my laptop which is pretty decent given that it's evaluating 10 seasons from 4 leagues 15 times per generation (default population size in CMA-ES for this number of parameters).

<div class="cell">
<details class="code-fold" open="">
<summary>
  Show the code
</summary>
<div class="code-copy-outer-scaffold">
<pre>
<code class="language-r hljs">
tic()
out <- cmaOptimDP(
    cmaobj,
    fitfunc,
    verbose=0,
    isFeasible = function(x) all(c(x[7] > x[6], x[12] >= 0))
)
toc()
</code>
</pre>
</div>
</details>
</div>


```
146.482 sec elapsed
```

The fitness decreases over time nicely (<a href="#fig:fitness">Figure 1</a>), and the ES ran for 1,157 generations until the fitness stopped decreasing.

It looks like it has converged quite nicely and won't benefit much from any hyper-parameter tuning.
If the behaviour of the final model looks weird we can always come back and tweak various settings.


<div class="cell">
<details class="code-fold">
<summary>
  Show the code
</summary>
<div class="code-copy-outer-scaffold">
<pre>
<code class="language-r hljs">
tibble(
    fitness =out$fitnessVec
) |>
    mutate(generation = row_number()) |>
    ggplot(aes(x=generation, y=fitness)) +
        geom_line() +
        theme_minimal() +
        labs(x="Generation", y=TeX("Fitness of highest individual ($\\sum RPS$)"))
</code>
</pre>
</div>
</details>
</div>

<a id="fig:fitness">
<div class="figure" style="text-align: center; width: 672px">
<img src="/img/predictaball_retrospective3_20260114/fig-fitness-1.png" alt="Fitness per generation of the evolutionary strategy"  />
<p class="caption">Figure 1: Fitness per generation of the evolutionary strategy</p>
</div>
</a>


# Behaviour

## Match prediction

Let's have a look at the optimized parameters that the ES has spat out, starting with the 7 relating to the match prediction.
Although taken at face value these are challenging to interpret: $\kappa$ and $\alpha$ only make sense in a joint context - i.e. the probability of a home win **isn't** $1-0.67=33\%$ - and there's no straight forward interpretation for $\alpha$ (it's the amount that $\text{logit}(1-P(H))$ will decrease by for each unit of rating difference$).

<div class="cell">
<details class="code-fold">
<summary>
  Show the code
</summary>
<div class="code-copy-outer-scaffold">
<pre>
<code class="language-r hljs">
inv_logit <- function(x) 1 / (1 + exp(-x))
tribble(
    ~Parameter, ~Value,
    "\U03BA \U2081", inv_logit(out$bestX[6]),
    "\U03BA \U2082", inv_logit(out$bestX[7]),
    "\U03B1 \U2081", inv_logit(out$bestX[1]),
    "\U03B1 \U2082", inv_logit(out$bestX[2]),
    "\U03B1 \U2083", inv_logit(out$bestX[3]),
    "\U03B1 \U2084", inv_logit(out$bestX[4]),
    "\U03B2 \U2081", out$bestX[5],
) |>
    kable("html", digits=3, align=c("l", "c")) |>
    kable_styling(c("striped", "hover"), full_width=FALSE)
</code>
</pre>
</div>
</details>
</div>

<table quarto-disable-processing="true" class="table table-striped table-hover" style="width: auto !important; margin-left: auto; margin-right: auto;">
 <thead>
  <tr>
   <th style="text-align:left;"> Parameter </th>
   <th style="text-align:center;"> Value </th>
  </tr>
 </thead>
<tbody>
  <tr>
   <td style="text-align:left;"> κ ₁ </td>
   <td style="text-align:center;"> 0.372 </td>
  </tr>
  <tr>
   <td style="text-align:left;"> κ ₂ </td>
   <td style="text-align:center;"> 0.668 </td>
  </tr>
  <tr>
   <td style="text-align:left;"> α ₁ </td>
   <td style="text-align:center;"> 0.649 </td>
  </tr>
  <tr>
   <td style="text-align:left;"> α ₂ </td>
   <td style="text-align:center;"> 0.643 </td>
  </tr>
  <tr>
   <td style="text-align:left;"> α ₃ </td>
   <td style="text-align:center;"> 0.641 </td>
  </tr>
  <tr>
   <td style="text-align:left;"> α ₄ </td>
   <td style="text-align:center;"> 0.613 </td>
  </tr>
  <tr>
   <td style="text-align:left;"> β ₁ </td>
   <td style="text-align:center;"> 0.003 </td>
  </tr>
</tbody>
</table>

Instead of inspecting the parameters, it's easier to gauge the prediction behaviour by looking at the relationship between the inputs (league, rating difference), and the output (probability of each outcome).
The function below generates the probabilities for a given set of inputs, using the parameters estimated by the ES.


<div class="cell">
<details class="code-fold" open="">
<summary>
  Show the code
</summary>
<div class="code-copy-outer-scaffold">
<pre>
<code class="language-r hljs">
predict_func <- function(league, delta) {
    if (! league %in% all_leagues) {
      stop(paste0("'", 
                  league, 
                  "' not found. Please select one of ", 
                  paste0(all_leagues, collapse=', '), 
                  "."))
    }
    leagues <- rep(0, length(all_leagues))
    leagues[match(league, all_leagues)] <- 1
    
    # R-implementation
    mu = sum(params[1:4] * leagues ) + params[5] * delta
    q1 = inv_logit(params[6] - mu);
    q2 = inv_logit(params[7] - mu);
    c(q1, q2-q1, 1-q2)
}
</code>
</pre>
</div>
</details>
</div>

The relationship as shown in <a href="#fig:predict-update">Figure 2</a>, is reassuringly (or eerily) similar to the hand-crafted Elo system.


<div class="cell">
<details class="code-fold">
<summary>
  Show the code
</summary>
<div class="code-copy-outer-scaffold">
<pre>
<code class="language-r hljs">
elo_diffs <- seq(-2000, 2000, by=10)
map_dfr(elo_diffs, function(x) {
    df <- predict_func("premiership", x) |>
            as_tibble() |>
            mutate(
                name = c('away_prob', 'draw_prob', 'home_prob'),
                elo_diff = x
            )
}) |>
    mutate(
        value = value * 100,
        name = factor(
            name,
            levels=c("home_prob", "draw_prob", "away_prob"),
            labels=c("Home win", "Draw", "Away win")
        )
    ) |>
    ggplot(aes(x=elo_diff, y=value)) +
        geom_vline(xintercept=0, linetype="dashed", colour="orange") +
        annotate("label", x=0, y=60, label="No difference", colour="orange") +
        geom_line(aes(group=name, colour=name)) +
        theme_minimal() +
        ylim(0, 100) +
        scale_color_brewer("", palette="Dark2") +
        labs(x=TeX("$\\psi_{home} - \\psi_{away}$"),
             y="Probability of outcome (%)") +
        theme(
            legend.position = "bottom",
            panel.grid.minor.y = element_blank()
        )
</code>
</pre>
</div>
</details>
</div>

<a id="fig:predict-update">
<div class="figure" style="text-align: center; width: 672px">
<img src="/img/predictaball_retrospective3_20260114/fig-predict-update-1.png" alt="The match prediction function evolved by the ES"  />
<p class="caption">Figure 2: The match prediction function evolved by the ES</p>
</div>
</a>


## Rating update

The table below shows the parameters related to the rating update. 
$K$ has barely moved from 20 for all 4 leagues, seemingly because this parameter doesn't impact much on the overall match prediction accuracy.
The 2 parameters that govern $G(x)$ are $\beta_2$ and $\beta_3$ and these have changed a fair bit from the initial values: $\beta_2$ (the maximum possible multiplier) has increased to ~5, while the coefficient of goal difference $\beta_3$ has decreased from the starting value of 0.3.

<div class="cell">
<details class="code-fold">
<summary>
  Show the code
</summary>
<div class="code-copy-outer-scaffold">
<pre>
<code class="language-r hljs">
tribble(
    ~Parameter, ~Value,
    "K \U2081", out$bestX[8],
    "K \U2082", out$bestX[9],
    "K \U2083", out$bestX[10],
    "K \U2084", out$bestX[11],
    "\U03B2 \U2082", out$bestX[12],
    "\U03B2 \U2083", out$bestX[13],
) |>
    kable("html", digits=3, align=c("l", "c")) |>
    kable_styling(c("striped", "hover"), full_width=FALSE)
</code>
</pre>
</div>
</details>
</div>

<table quarto-disable-processing="true" class="table table-striped table-hover" style="width: auto !important; margin-left: auto; margin-right: auto;">
 <thead>
  <tr>
   <th style="text-align:left;"> Parameter </th>
   <th style="text-align:center;"> Value </th>
  </tr>
 </thead>
<tbody>
  <tr>
   <td style="text-align:left;"> K ₁ </td>
   <td style="text-align:center;"> 22.347 </td>
  </tr>
  <tr>
   <td style="text-align:left;"> K ₂ </td>
   <td style="text-align:center;"> 21.057 </td>
  </tr>
  <tr>
   <td style="text-align:left;"> K ₃ </td>
   <td style="text-align:center;"> 18.569 </td>
  </tr>
  <tr>
   <td style="text-align:left;"> K ₄ </td>
   <td style="text-align:center;"> 19.924 </td>
  </tr>
  <tr>
   <td style="text-align:left;"> β ₂ </td>
   <td style="text-align:center;"> 4.941 </td>
  </tr>
  <tr>
   <td style="text-align:left;"> β ₃ </td>
   <td style="text-align:center;"> 0.129 </td>
  </tr>
</tbody>
</table>



As ever though, the easiest way to understand these values is to see them in action.
<a href="#fig:G">Figure 3</a> compares the hand-crafted $G(x)$ from [last time](https://stuartlacy.uk/2026/01/09/predictaball-retrospective-part-2-elo-rating-system/) to that from the ES, showing that the parameterization derived automatically from the ES to optimize for match predictive accuracy has a higher multiplier at every goal difference than the hand-crafted one, and due to its differing functional form, $G(1) \neq 1$.


<div class="cell">
<details class="code-fold" open="">
<summary>
  Show the code
</summary>
<div class="code-copy-outer-scaffold">
<pre>
<code class="language-r hljs">
G <- function(x) {
    out$bestX[12]*(1-exp(-out$bestX[13]*x))+1
}
</code>
</pre>
</div>
</details>
</div>



<div class="cell">
<details class="code-fold">
<summary>
  Show the code
</summary>
<div class="code-copy-outer-scaffold">
<pre>
<code class="language-r hljs">
G_old <- function(x) {
    ifelse(x <= 1, 1, log2(x * 1.7))
}
tibble(
    mov = as.integer(0:10),
    new = G(mov),
    old = G_old(mov)
) |>
    pivot_longer(-mov) |>
    mutate(
        name = factor(name, levels=c("new", "old"),
                      labels=c("ES", "Hand-crafted"))
    ) |>
    ggplot(aes(x=mov, y=value, colour=name)) +
        geom_line() +
        theme_minimal() +
        labs(x="Margin of victory", y="G multiplier") +
        scale_colour_brewer("", palette="Set1") +
        theme(
            legend.position = "bottom"
        )
</code>
</pre>
</div>
</details>
</div>

<a id="fig:G">
<div class="figure" style="text-align: center; width: 672px">
<img src="/img/predictaball_retrospective3_20260114/fig-G-1.png" alt="Comparison of G as estimated by the ES vs the hand-crafted Elo version"  />
<p class="caption">Figure 3: Comparison of G as estimated by the ES vs the hand-crafted Elo version</p>
</div>
</a>


The function below puts it all together by calculating the rating update $\delta$ as a function of the predicted probabilities, the margin of victory, the league, and the actual outcome.

<div class="cell">
<details class="code-fold" open="">
<summary>
  Show the code
</summary>
<div class="code-copy-outer-scaffold">
<pre>
<code class="language-r hljs">
rating_update <- function(probs, mov, league, outcome) {
    if (! league %in% all_leagues) {
      stop(paste0("'", 
                  league, 
                  "' not found. Please select one of ", 
                  paste0(all_leagues, collapse=', '), 
                  "."))
    }
    if (! outcome %in% all_outcomes) {
      stop(paste0("'", 
                  outcome, 
                  "' not found. Please select one of ", 
                  paste0(all_outcomes, collapse=', '), 
                  "."))
    }
    leagues <- rep(0, length(all_leagues))
    leagues[match(league, all_leagues)] <- 1
    outcomes <- rep(0, length(all_outcomes))
    outcomes[match(outcome, all_outcomes)] <- 1
    
    K = sum(params[8:11] * leagues)
    O = sum(outcomes * c(0, 0.5, 1))
    E = (sum(probs * c(1, 2, 3)) - 1) / 2
    
    mov_multiplier = params[12]*(1-exp(-params[13]*mov))+1
    mov_multiplier * K * (O - E)
}
</code>
</pre>
</div>
</details>
</div>

This is used to generate <a href="#fig:rating-update">Figure 4</a>, which shows how $\delta$ varies with these inputs under 4 scenarios:

  - 'Average': The average probabilities of each outcome (H|D|A): 46% | 25% | 29%
  - 'Home favourite': Home team heavy favourite: 80% | 10% | 10%
  - 'Away favourite': Away team heavy favourite: 10% | 10% | 80%
  - 'Draw': A draw is the most expected outcome: 10% | 80% | 10%
  
It shows both the non-linear relationship between $\delta$ and goal difference through $G(x)$, as well as highlighting the impact of the pre-match predictions - for example if the home team are the heavy favourite, they can only win a small rating increase no matter how many goals they score, but if they lose they stand to lose a lot.

<div class="cell">
<details class="code-fold">
<summary>
  Show the code
</summary>
<div class="code-copy-outer-scaffold">
<pre>
<code class="language-r hljs">
params <- expand_grid(
    mov=seq(0, 10),
    result=c('away', 'draw', 'home')
) |>
    filter(
        !(mov > 0 & result == 'draw'),
        !(mov == 0 & result %in% c('home', 'away'))
    ) |>
    distinct()
scenarios <- tribble(
    ~scenario, ~home_prob, ~draw_prob, ~away_prob,
    "Average", 0.454, 0.252, 0.294,
    "Home favourite", 0.8, 0.1, 0.1,
    "Away favourite", 0.1, 0.1, 0.8,
    "Draw", 0.1, 0.8, 0.1
) |>
    mutate(
        scenario = factor(
            scenario,
            levels=c(
                "Average",
                "Home favourite",
                "Away favourite",
                "Draw"
            ),
            labels=c(
                "Prediction=Average",
                "Prediction=Home favourite",
                "Prediction=Away favourite",
                "Prediction=Draw"
            )
        )
    )
params <- params |>
    cross_join(scenarios)
map_dfr(1:nrow(params), function(i) {
    delta <- rating_update(
        c(params$away_prob[i], params$draw_prob[i], params$home_prob[i]),
        params$mov[i],
        'premiership',
        params$result[i]
    )
    params[i, ] |>
        mutate(delta=delta)
}) |>
    ggplot(aes(x=mov, y=delta, colour=result)) +
        geom_point() +
        geom_line() +
        theme_minimal() +
        facet_wrap(~scenario) +
        scale_colour_manual("Result", values=c("lightblue", "forestgreen", "lightsalmon")) +
        theme(
            legend.position = "bottom"
        ) +
        labs(
            x=TeX("$goals_{home} - goals_{away}"),
            y=TeX("Rating update $\\delta$")
        )
</code>
</pre>
</div>
</details>
</div>

<a id="fig:rating-update">
<div class="figure" style="text-align: center; width: 672px">
<img src="/img/predictaball_retrospective3_20260114/fig-rating-update-1.png" alt="The evolved rating update function under 4 scenarios for predicted match probabilities"  />
<p class="caption">Figure 4: The evolved rating update function under 4 scenarios for predicted match probabilities</p>
</div>
</a>


That's all for now, the actual predictive accuracy on the test set will be revealed in a future post once all the models have been introduced. 
Next up is the first new modelling work I did on Predictaball in 6 years: particle filters.
