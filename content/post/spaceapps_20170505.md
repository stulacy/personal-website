+++
date = 2017-05-05
draft = false
tags = ["software development"]
title = "Building a 2D No Man's Sky - NASA Space Apps"
math = false
+++

I've never really been much of a hacker, I much prefer to think my projects through entirely and plan them out on pen and paper before starting to write any code. As such I've never really had much interest in a hackathon. With a bit of apprehension then I participated in my first one over the weekend. 

The particular event was [NASA Space Apps](https://2017.spaceappschallenge.org/), where NASA provide lots of data and offer challenges related to modelling certain natural phenomena, providing data visualisation, or prototype hardware tools that fit a particular niche. I was working with some friends who wanted a bit more freedom and so we choose the challenge that allows you to do whatever you want, at the expense of not being eligible for the grand prize.

As my friends are interested in game development they wanted to make a procedurally generated space exploration game, loosely based on No Man's Sky. I had no experience whatsoever with game dev so was skeptical if I could bring anything to the group but decided to go along to try something new. Over the course of the weekend I learnt a lot about game dev, especially procedurally generating content, which wasn't actually that foreign to me as I do a fair bit of stochastic simulation in my day job. I quite like procedurally generated games, since, provided the underlying mechanics are well designed, the game world feels organic and immersive, particularly in contrast to AAA games where you can tell the game designer has just put a few corridors filled with generic bad guys to separate boss fights / cutscenes / general exposition.

We used Python for the primary reason that it's so easy to get something basic up and running, with `pygame` providing the game engine and `Box2D` the physics. To split up the tasks, we formed the game out of three sections:

  1. Exploring a solar system
  2. Landing on a planet
  3. Planet exploration

Exploring the solar system involves an Asteroids like game, where you dodge asteroids and navigate to planets that are orbiting a star, each potentially having their own moons. 

![Exploring a solar system](/img/spaceapps_05052017/space.png)

Once you touch a planet or moon, the game jumps to the landing stage, which is reminiscent of Lunar Lander style games, where you have to safely control your craft down onto the planet surface. 

![Landing the spaceship](/img/spaceapps_05052017/landing.png)

Once a successful landing had been completed, the game skips to a 2D platformer, where you control the astronaut exploring the planet. This stage is the part I worked on, with a friend doing the terrain generation while I did little bits here and there, such as controlling the astronaut and adding craters to the environment.

![Exporing the planet](/img/spaceapps_05052017/planet1.png)

![Exporing the planet 2](/img/spaceapps_05052017/planet2.png)
![Exporing the planet 3](/img/spaceapps_05052017/planet3.png)

The overall aim of the game is to find a planet that is suitable for sustained life, i.e. one that meets the following requirements:

  - Sufficient oxygen in the air
  - Values of gravity that humans can cope with
  - Liveable temperatures
  - The existence of water

Since the game is procedurally generated it can take a while until a suitable planet is found, particularly because there's no guarantee that a suitable planet exists in each solar system. Furthermore, there are 6 main planet archetypes, such as Earth style ones, ice, desert, rock, gas giant, and _other_. Each of these have their own unique appearance (the guy who worked on the planet exploration visuals did a great job) in addition to different distributions for the above resources. I.e. a desert planet is not likely to contain water. When you land on a planet these values (oxygen %, gravity value, existence of water, temperature) are not known but become revealed to the player the longer they spend in the planet. You can jump back into the lander to go back up to the space exploration level at any point to find another planet.

Another aspect to the game is managing three resources:

  - Fuel
  - Health
  - Oxygen

Fuel gets depleted as you use your boosters in both the main space exploration stage and the lander stage, and is also used by your jetpack during the planet exploration minigame. Health gets reduced by colliding with asteroids during space exploration and upon any contact above a certain speed with the ground during the landing stage. Both of these resources get replenished during the planet exploration in the form of random pick-ups dotted around the environment. The final resource, oxygen, is the opposite, as it gets depleted just from spending time on a new planet, but if you retreat back up to your main spaceship it will replenish.

Overall, I'm pleasantly surprised with the progress we made in just over a day. The game is playable, with the procedurally generated content helping replayability, and the art design is very good, particularly during the planet exploration stage. We actually won the user's prize for best project which was a bit of a surprise, as the game doesn't exactly fill any real world niches. If we had more time we'd add more content, particularly during the planet exploration stage, allowing the user to mine a planet for its natural resources while dealing with hostile aliens.

All of our code is [available on Github](https://github.com/AndrewJamesTurner/Every-Womans-Ground), along with instructions for installing the game. We didn't have time to provide a portable binary wrapper, but installing the required dependencies in a virtual environment doesn't take long and is relatively hassle free. Please feel free to provide feedback on github, we probably won't put too much more effort into it but are always interested to hear what others think.
