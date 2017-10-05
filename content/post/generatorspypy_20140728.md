+++
date = 2014-07-28
draft = false
tags = ["Python", "PyPy"]
title = "Recursive generator behaviour in PyPy"
math = false
+++

I encountered an odd bug recently in my GP code. Large amounts of memory were being used, and was increasing substantially at each generation. I realised the problem was most likely in how the tree was traversed to obtain an output for each input data pattern. This was currently done with an iterative approach (which with post-order traversal is not fun!), as I'd assumed that recursive methods would use more memory. Changing this method to a a recursive traversal resulted in quicker run times (by ~50%) and ~100% less memory used with more consistency. Brilliant, lesson learned, never just assume one approach is better than another without quantifying it. I then wanted to go one step further and make a general generator function, that would return the current node at each iteration allowing for the user to specify what to do on each 'visit'. I.e. in the evaluation method it would call the generator function and evaluate each node at each step, or in a method to get the node at a particular index it would iterate through the nodes and stop at the desired one. Python allows for generator functions to be called recursively, and so I implemented it only to discover that it was using even more memory than my original approach. I couldn't see why this would be the case, as one of the selling points of generators is that they are low memory. I could understand it using more memory than the tailor made recursive method as it doesn't have to keep track of the generator, but not to the extent it was at. I posted a [question](http://stackoverflow.com/questions/24962093/why-is-using-a-python-generator-much-slower-to-traverse-binary-tree-than-not) to StackOverflow to try and understand why this was and got some very illuminating responses.

Basically the problem wasn't my code, or generators or recursion, but instead was PyPy itself. According to a Python core developer who replied, PyPy is optimised for function calling, rather than generators (or iterators in general). I verified this by running the EA again with the standard Python interpreter and noticed much more steady memory usage and quicker run speed. So just be careful if you're using PyPy with iterators, don't always assume what's best in standard Python translates to PyPy.