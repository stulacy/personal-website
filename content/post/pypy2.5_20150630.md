+++
date = 2015-06-30
draft = false
tags = ["Python", "PyPy"]
title = "Evaluating the performance of PyPy 2.5"
math = false
+++

Now that I've finished my teaching qualification (the <a href="http://www.york.ac.uk/admin/hr/researcher-development/ylta/">York Learning and Teaching award</a>) I've had some time to get back into research. I've been updating various bits of software that I haven't used much over the last month or so, one of which was to update PyPy to version 2.5 from 2.3, skipping a version in the process. I expected that I may get a few speed bonuses but there wouldn't be a significant improvement from 2.3 in terms of runtimes. As usual I decided to run a full experiment to find out. I evolved GP expression tree classifiers on ten binary datasets taken from the <a href="http://archive.ics.uci.edu/ml/">UCI repository</a> under 30 repeats of ten-fold cross-validation. I recorded the time taken per fold resulting in 300 samples from each dataset.

![](/img/pypy2.5_27032015/runtimes_all_blog.png)

Overall there is little difference in runtimes between the three PyPy versions, except 2.4 has a slightly reduced range to the other two implementations. I'll have a quick look at the plot per dataset to see if there are any anomalies.

![](/img/pypy2.5_27032015/runtimes_blog.png)

There doesn't appear to be a strong trend between runtime and PyPy versions. On the first dataset PyPy 2.5 is worryingly slower than the other two versions, but on none of the remaining nine datasets does this trend occur again. 

Based purely on these runtime results it'd be safe to say that you might as well use the latest PyPy version as while it doesn't offer any speed advantages it doesn't do any worse either. Or so I thought anyway...

I'd noticed some odd results every now and then with experiments being run under PyPy 2.5 on the cluster I run my simulations on. Basically the runs were halting mid-execution. Initially I'd put this down to being unfortunate with the cluster having downtime but I noticed that this was only occurring on runs using the 2.5 version JIT. I then run another set of experiments this time recording memory usage to investigate what was going on and the results were very unexpected.

![](/img/pypy2.5_27032015/memory_all_blog.png)

Straight away it can be seen that 2.5 uses ~4x as much memory as the older two versions of PyPy. And these results have been recorded on again ten datasets so it does not appear to be a memory leak on one particular setup.

![](/img/pypy2.5_27032015/memory_blog.png)

The above plot highlights how this trend is true across the board, and even PyPy 2.4 has a large range of memory uses, despite the median being comparable to 2.3.

At this point I'd solved the issue of the randomly halting runs. The cluster I use has a default memory usage of 1GB and will terminate your jobs if they try to use more. This also explains why most of the whiskers stop at 1000MB in the plot, as the runs would be stopped soon. 

I could either look more into this and determine the source of this to see if it was a leak in my code or a bug with PyPy but frankly I don't have the time and so decided to revert back to PyPy 2.3 for the time being. While I could ask the cluster to use more than 1GB of memory it seems a waste for a relatively small program which was fine using a quarter of this a few months ago.

In conclusion always look at as much data as you can before you upgrade your workflow. While the runtimes didn't offer any disadvantages to updating PyPy there were clearly problems there. And if you've also noticed odd memory issues when using PyPy 2.5 then don't worry, you're not the only one.
