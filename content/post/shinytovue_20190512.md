+++
date = 2019-05-29
draft = false
tags = ["webdev", "software", "R", "Predictaball"]
title = "5 reasons to move from a Shiny website to a static site"
math = false
+++

Back in March I rewrote [thepredictaball.com](https://thepredictaball.com) from its original R Shiny implementation into a static website using the Vue Javascript framework.
I intended to write about it at the time but I've been busy and hadn't made time for it until now, which is handy given that the football season has just finished!

Excuse the clickbait title, but I genuinely couldn't think of a better way of organising this post. 
It's intended to provide a brief introduction to the benefits of Javascript static site frameworks for people currently running a website in Shiny.
It isn't intended to provide a deep introduction to Vue (maybe that will come in a later post). All I will say is that I went with Vue as I'd heard good things about it, particularly related to its lightweight footprint and its ability to slot into any existing project without requiring a large amount of boiler-plate code.

# Where Shiny shines

I think it's fair to say that Shiny isn't designed for high-performance applications.
It's an extremely user-friendly platform that abstracts away a lot of time-consuming boiler-plate code for both backend and UI, so that people can get websites up and running quickly without having to learn a new technology stack or language(s).
However, the trade-off is a loss of fine control and slightly poor performance.

I still love Shiny and will continue to use it for prototyping and various research projects, but I hope you will see by the end of this post that there are some situations where it can be improved upon without too much additional effort.

# Static vs dynamic site

Throughout this post I refer to websites as being either "static" or "dynamic".
In short, a dynamic website is one where the webpages are generated on the fly for each user request at the server-end, with the content pulled from a database.
This means that the webpages (the HTML that your browser displays) are dynamic and depend on some state (generally time, so that more recent content is displayed first, but also the user that is logged in, user's location etc...).
A static website's content doesn't change, so that the HTML read by the browser will be the same for any user and can thus be downloaded directly from the server without needing to be generated first.

There's a bit more to it than that, and static pages can still interact with DBs through API calls, but the important message to understand is that dynamic websites (which includes any built with Shiny) are much more powerful but this comes at a complexity cost.

Read [this post](https://medium.com/@robmuh/the-static-web-returns-e240dd100d65) to discover how static sites have come back into fashion, while [this post](https://wsvincent.com/static-vs-dynamic-websites-pros-and-cons/) provides a more comprehensive breakdown of the pros and cons of each.

# Requirements

The most important step is to identify whether you are able to easily move away from Shiny or not.
Fortunately, there's really only one situation where this isn't easily feasible, **when the website requires live R-specific computation**.
I.e. if you are just using Shiny to visualise pre-calculated values then this can easily be reproduced in a Javascript framework. 
If the website is doing something more complex, such as calculating a model's output to given inputs, then it depends on the model, as a GLM can be easily implemented in any language given the parameters, whereas a more blackbox machine-learning algorithm will be less portable.

For my uses I just needed a front-end and chose Shiny because I was already familiar with it; I wasn't doing any R-specific calculations on the server end as my data was just being pulled straight from a database.
Everyday my home server (I could run a website off it but I'm not keen on opening up my network) runs a script that scrapes all the fixtures for the day, predicts the outcomes, grabs the previous day's results, and finally saves all this to a database.
My website simply pulls the current ratings and predictions from this database and visualises them.

Because my database only updates once a day I don't need to have a dynamic website, rather I only need some way of updating the website on a daily basis.
This is precisely where a static site comes in handy, building all the required HTML+JS+CSS that can then be simply uploaded to some host (Amazon S3 is particularly cheap as I describe later) without the need for a full web-server.

I could easily append a Vue build + S3 deploy to my daily update script, although I realise this isn't directly applicable to everyone's situation, but the principle remains that if you don't require immediate access to a constantly-updating database then static sites are a strong alternative.

# Benefits

Enough background, why would you want to move your Shiny app to a static site?

## 1. Cost

This was my primary motivation, since I was hosting [Predictaball's website](https://thepredictaball.com) using Amazon Web Services (AWS), in particular the most basic EC2 instance (essentially a virtual server that actually hosted the Shiny site) and the cheapest RDS instance (hosted database system, using Postgres).
I was initially under the year free tier, but this ended in March, motivating me to move to a cheaper setup. 
Now I wasn't exactly facing a massive bill, the total cost coming to around £20 a month, but my current static site on S3 costs **2p a month, 0.1% of the old cost!**

Even better, is that I can now turn off my unused RDS instance in the off-season, whereas if I were still running my dynamic Shiny site I'd need to keep it (and the EC2 instance) online.
Of course I still pay the electricity costs of my home server that runs the update script but I was paying this before already on top of the EC2/RDS services and it isn't much in comparison.

## 2. Higher performance

By nature of having everything written in R, a language that web browsers can't interpret, Shiny requires that everything dynamic has to go through the server. 
For example, the template web-app in RStudio that displays a histogram of data with the user selecting the number of bins has to relay the input back to the server, since the plot is produced using R, whereas this would be much more responsive if it was all run on the client side, such as in a static site.

This relates back to the general snappier and more responsive feel of static sites due to not needing to have everything generated on the server end first.

## 3. Flexibility

Shiny succeeds in providing a framework that can be used to host all manner of websites without much knowledge of web-dev in general; however, the cost of this is user-control.
All Shiny sites must have a backend server running constantly, and access to UI elements is primarily through wrappers to the Bootstrap JS library.
Adding any custom code can be challenging, even to do simple interactions with the DOM (although the [ShinyJS package](https://deanattali.com/shinyjs) helps here), yet alone more advanced requirements such as routing, interfacing with any of the fantastic packages on npm, or accessing external APIs.

While for many situations this trade-off is perfectly acceptable, it can hamper larger projects or those that are going to go into production.

## 4. Develop new skills

Web-development skills are highly in demand right now and are a strong addition to a data-scientist's/statistician's/analyst's/whatever-you-want-to-call-it's toolkit, so that when it comes to deploy models you are more aware of the different available options and are able to make a more informed decision.

This was another big motivation of mine as I don't get to do much serious software development in my current role and I haven't done any web-dev since working on CMSs built in PHP and fancied understanding just what webpack is exactly.
Furthermore, it's provided me with more experience with Continuous Integration (CI) and AWS, both of which are very useful for working in a production environment.

## 5. It's actually rather easy

Depending on your background level of general non-R programming, moving to a standard web framework may be less difficult than you expect.
Javascript, for all its faults, is remarkably easy to get to grips with, since it is weakly typed like R.
A lot of the new aspects then will be specific to the web framework you choose, which for me was **Vue**.
I personally found Vue very enjoyable to work with, with the templating system providing a cleaner partition between the UI and the data than Shiny, where I found I was constantly creating a lot of reactive objects - in particular dynamic UIs resulted in a lot of ugly code.

I was more worried about how easy it would be to get a UI working, so I simply used the same Bootstrap items that Shiny does, but this time I'm interfacing directly with the native implementation rather than through Shiny's wrapper so I had more control.
Of course, if I didn't like Bootstrap I'm free to use a different library or create my own elements.
Likewise, while I missed working with ggplot2, there are lots of thoroughly documented and tested plotting libraries out there for Javascript (I've gone with plotly) that this quickly doesn't become an issue.

# Overall

I hope this post has provided a bit of an insight into a few of the numerous ways in which a website can be configured, and in particular how they differ from Shiny. 
I've recommended Vue as a lightweight Javascript framework without going into too much detail on it, but the documentation and [tutorial](https://vuejs.org/v2/guide/) are very useful and will get you up and running quickly.

Something else that I've briefly glossed over but is important is the ability of static sites to obtain data through API calls.
This can be implemented in a so-called "severless" setup, which is a bit of a misnomer as there will still be a server involved, it's just more de-coupled from the website.
Essentially the website is a static site that dynamically loads data from a DB by accessing APIs that are hosted in the cloud (through services such as AWS Lambda).

This is often far cheaper than continually running a web server as you only pay per API request and is more flexible too, since these APIs can be used for other purposes (mobile apps or public APIs), while still getting the benefits of a static site.
I can't justify such a configuration for Predictaball right now but it's something I'll bear in mind for future projects.
