+++
date = 2019-08-06
draft = false
tags = ["football", "Predictaball", "machine learning"]
title = "Optimising a football team rating system for match prediction accuracy"
math = true
+++

While I've been quite happy with the performance of my [Predictaball football rating system](https://www.thepredictaball.com), one thing that that's bothered me since its inception last summer is the reliance on hard-coded parameters.

Similar to many other football rating methods, it's an adaptation of the Elo system that was designed for Chess matches by Arpad Elo in the 1950s. 
His aim was to devise an easily implementable system to rate competitors in a 2-person zero-sum game.

Predictaball, on the other hand, is more concerned with accurately predicting future matches; model interpretability and portability is far less of a requirement.
Furthermore, we have the advantage of an abundance of historical performance data and cheap computation to fit our models.

This post summarises the implementation of a rating system that simultaneously optimises parameters used in both the match prediction model and the rating update function in order to maximise match prediction accuracy.

## Current Elo implementation

I'll firstly run a quick refresher on how the standard Elo method is used by Predictaball.

### Elo expected outcome

The formula for calculating the expected outcome $E$ of a match in Elo's original formulation is shown below, where $\delta$ is the difference in rating between the home and away team, i.e. $\delta = R\_{H} - R\_{A}$.

$$E = \frac{1}{1+10^{- \delta /400}}$$

Note that this is the inverse logit function in base 10 rather than the standard natural base, and is therefore equivalent to logistic regression with 1 predictor ($\delta$) and the coefficient $\beta = \frac{1}{400}$.
The version used by Predictaball (and others including 538) also adds in a factor to account for home advantage, i.e. $\delta = R\_{H} - R\_{A} + \alpha$, where $\alpha$ is estimated from the historical data as the rating difference associated with the proportion of matches that are home wins.

This expected outcome framed as logistic regression is therefore 

$$\text{logit}(P(\text{home win})) = \alpha + \beta(R\_{H} - R\_{A})$$

However, this model isn't overly useful on its own as it doesn't tell us whether an away win or draw is more likely, and we might be more interested in other outcomes such as the number of goals scored (what Predictaball currently outputs).
Furthermore, we might want to include additional predictor variables, such as if a team is missing a key player, or has had a new manager recently.
We definitely have sufficient training data to fit more complex predictive models and not rely upon hardcoded simplifications.

### Rating update equation

The Elo rating update equation is a function of the difference between the actual outcome $O$ ($O \in \left\\{0, 0.5, 1\right\\}$ for away win, draw, and home win respectively), the expected outcome $E$ from the above equation, and a gain factor $K$ that controls how much rating is won/lost per game.
$K$ is traditionally fixed at 20 for a long-running competition or set higher for a short tournament.

$$R^{\prime}\_{H} = R\_{H} + K(O - E)$$

Predictaball also scales the output [to account for the margin of victory (MoV)](https://www.thepredictaball.com/howitworks), similar to the method used by 538

$$R^{\prime}\_{H} = R\_{H} + KG(\text{MoV})(O - E)$$

The functional form of G is shown below, where the multiplier decays as the margin of victory increases.
Also note the adjustment by $\delta$ (labelled $\text{dr}$ in the figure) to stop higher rating teams getting further and further ahead.
The parameters of $G$ were derived by hand by tweaking them until the resulting shape looked appropriate, rather than using a data-driven approach.

![output of G](/img/elo_31082017/unnamed-chunk-16-1.png)

### Secondary prediction network

The final limitation with my implementation is that currently the ratings are retrofitted to my dataset using the above (hardcoded) parameters, and then a secondary predictive model is fitted using the $\delta$ as input predictors to a Bayesian model that predicts how many goals each team will score.
This model is far more flexible and useful than the logistic regression used in the Elo system, but it still seems a waste to fit a second predictive-model separately from the rating system.

## Rating System Optimised for Predictive Accuracy (RSOPA)

The properties of my ideal rating system are:

  - Be entirely data driven, i.e. parameters fitted using historical data
  - Flexible to choice of predictive model
  - Primary objective is predictive accuracy

I'll call this method Rating System Optimised for Predictive Accuracy (RSOPA), just because there aren't enough acronyms in the world already, and will now detail its implementation.

### Predictive model

The crux of the Elo system is the $O-E$ term that is in $\[-1, 1]$.
Any predictive model for $E$ and any measure of outcome $O$ can be used provided a statistic in this range can be calculated.

For an initial proof-of-concept I'm going to use an ordinal multinomial regression for the match prediction model.
This estimates probabilities for each of the 3 possible outcomes while taking the natural ordering into account (away win < draw < home win).
The model is specified as follows

$$\mu = \alpha\_{league} + \beta \delta$$
$$\text{logit}(q\_1) = \eta\_1 - \mu$$
$$\text{logit}(q\_2) = \eta\_2 - \mu$$
$$P(\text{away win}) = q\_1$$
$$P(\text{draw}) = q\_2 - q\_1$$
$$P(\text{home win}) = 1 - q\_2$$
        
Where $\alpha\_{league}$ is a league specific intercept, while the coefficient $\beta$ and the two thresholds $\eta\_1$ and $\eta\_2$ are constant across leagues.

A value for $E$ in $\[0,1\]$ can be obtained by either scaling the linear predictor $\mu$, or by using a more hacky solution as the normalised weighted average of the 3 outcomes.

$$E = \frac{(1P(\text{away win}) + 2P(\text{draw}) + 3P(\text{home win}) - 1)}{3-1}$$

Using the same output formulation as before ($O \in \left\\{0, 0.5, 1\right\\}$), we have a more useful predictive model that can still be incorporated into the Elo rating system.

With 4 leagues in this dataset, this is a total of **7 parameters** to be estimated for the predictive model.

### Rating system

The rating system is much as before

$$R^{\prime}\_{H} = R\_{H} + KG(\text{MoV})(O - E)$$

Where $G(x)$ now takes the general form $G(x) = A(1-exp(-Bx))+1$, where $A$ controls the maximum output and $B$ the decay rate, as shown in the plot below.

![Functional form of G](/img/nnratings_20190805/G_shape_small.png)

Alongside $A$ and $B$, the gain parameter $K$ can also be treated as free, and I'll let it vary by league.
Therefore, there are **6 free parameters to optimise from the rating update model**.

### Parameter optimisation using Genetic Algorithm

In total there are 13 parameters to be solved: 7 from the match prediction model and 6 from the rating update equation.
For an initial proof-of-concept I'm going to solve for these using a [Genetic Algorithm](https://en.wikipedia.org/wiki/Genetic_algorithm) - in particular the [R interface to CMA-ES](https://cran.r-project.org/web/packages/rCMA/index.html). 
As I'm primarily interested in obtaining the most accurate match predictions, my fitness function is the [Ranked Probability Score (RPS)](https://journals.ametsoc.org/doi/pdf/10.1175/MWR3280.1), the specific case of the Brier score when the output is ordered.

I'm training the network on the 13,014 games in the top 4 European leagues (Premiership, Bundesliga, La Liga, Serie A) between the 2005-2006 and 2013-2014 seasons, although I'm allowing 3 seasons for the ratings to settle before the RPS is counted.
To obtain an idea of how well the model works on unseen data I'll form a test set comprising the seasons from 2014-2015 up to and including 2018-2019.

The hardcoded aspects of this model are the functional forms of the 2 models and the starting rating, which I'm setting at 2,000 completely arbitrarily.

## Final model

The table below compares the RPS from RSOPA against two Bayesian models that were fitted to the original hand-implemented rating system. The 'Goals' method predicts the goals scored by each team and was used in the [Predictaball forecasts](https://www.thepredictaball.com) for the 2018-2019 season while 'Ordinal' is an ordinal multinomial regression and was used prior to 2018.

Unsurprisingly, RSOPA does far better on the training set - an improvement of 0.001 rps would be enough to move me up to [5th from 11th in 2017-2018 season](https://stuartlacy.co.uk/2018/06/17/evaluating-the-predictaball-football-rating-system---2018/) - but there is less of an difference on the test set.
This could be one area for future work, adding in regularisation methods or early-stopping to minimise the risk of over-fitting.

<table class="table table-striped table-hover" style="width: auto !important; margin-left: auto; margin-right: auto;">
<thead>
<tr>
<th style="text-align:left;">
Model
</th>
<th style="text-align:right;">
Training data
</th>
<th style="text-align:right;">
Test data
</th>
</tr>
</thead>
<tbody>
<tr>
<td style="text-align:left;">
RSOPA
</td>
<td style="text-align:right;">
0.1998
</td>
<td style="text-align:right;">
0.1974
</td>
</tr>
<tr>
<td style="text-align:left;">
Bayesian Ordinal
</td>
<td style="text-align:right;">
0.2007
</td>
<td style="text-align:right;">
0.1977
</td>
</tr>
<tr>
<td style="text-align:left;">
Bayesian Goals
</td>
<td style="text-align:right;">
0.2009
</td>
<td style="text-align:right;">
0.1978
</td>
</tr>
</tbody>
</table>

The plot below shows the modelled outcome probabilities for a range of $\delta$ across the 4 leagues. 
It immediately highlights the benefits of using a more complex match prediction model than the standard Elo logistic regression as the away win and draw probabilities aren't linearly related. 
Incidentally, the peak in the draw probability at around $\delta = -200$ suggests that the home advantage is worth 200 rating points.
Finally, the known fact that the Bundesliga has a lower proportion of home wins can be seen by the red curve lying lower than the others, resulting from the league specific $\alpha$.

![Probability of different output](/img/nnratings_20190805/match_prediction_small.png)

Moving onto the rating network, and the fitted $G(\text{MoV})$ function is shown below.
It allows for very high multipliers which concerns me slightly as it could allow for stronger teams' ratings to skyrocket.
If this becomes a problem I could reshape $G$ to include a penalty term for the rating difference to reduce this auto-correlation, like what 538 use and I've borowed for the current version of Predictaball.

![Function for G](/img/nnratings_20190805/final_g_small.png)

The team ratings at the end of the 2018-2019 season are shown below for both the current implementation (red) and RSOPA (blue).
RSOPA definitely produces more dispersed teams due to the large multiplier effect of $G$, which is around 3x for a 5-goal win in the current method but 4.4x in the new system.

This has some advantages, in that it's clear to see who the best and worst-rated teams are, for example the top 6 in the Premier League are far easier to identify than under the old system.
But it could be easy to overinterpret these ratings by reading too much into the large differences between the teams, when really these need to be taken into consideration alongside the $\beta$ coefficient.

For example, if we were to predict the outcome of Liverpool vs Everton under the old system we would get H/D/A of 72% / 17% / 11%, which isn't that much less than the 78% / 14% / 8% under RSOPA, so even though Liverpool's rating is far closer to Everton's under the old method the $\beta$ was larger to compensate.

![Team ratings summer 2019](/img/nnratings_20190805/ratings_combined.png)

## Further work

I've run out of time to tinker with this concept further this Summer, but the next thing I'd be looking at improving would be to remove $G$ entirely and encapsulate the predicted scale of victory in $E$, so that the rating update is simply $K(O-E)$ as before.

In particular, I'd like to use my current goal prediction model rather than the ordinal multinomial regression I'm using here and I'd like to go back to using a Bayesian model rather than just using point estimates of parameters.
However, this is computationally more challenging to fit so I'd need to find time to sit down and properly think about it.

As for now, I'm going to use these ratings on [The Predictaball's website](https://thepredictaball.com) for the coming 2019-2020 season and see how they get on. 
