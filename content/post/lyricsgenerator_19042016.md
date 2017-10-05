+++
date = 2016-04-19
draft = false
tags = ["machine learning", "Markov chains"]
title = "Generating Iron Maiden lyrics with Markov chains"
math = false
+++

I've been wanting to play with Markov Chains for a while now, and now that I'm starting to get into Bayesian analysis I'm going to need to use them more often. One fun use of them is to generate text which can (at a stretch) pass as written by a drunk person. For nice examples of them in action have a look at [Garkov (generated Garfield strips)](http://joshmillard.com/garkov/) or even an entire subreddit generated with them [reddit.com/r/subredditsimulator](https://www.reddit.com/r/subredditsimulator). The later is particularly cool as it's got a large text source to build from so it can result in coherent sentences, such as _MRW someone asks me how much she misses me and doesn't show any sign of stopping_ or _Jeb Bush wants Margaret Thatcher statue measuring 250ft planned at Kent University to change last name to Turbo Suit_. 

In this post I provide a very brief introduction to text generation using markov chains, and apply them to writing new lyrics for the classic heavy metal band Iron Maiden. The choice of application domain is a result of listening to them non-stop for days in anticipation of seeing them for the first time in over ten years this summer.

Note that I don't actually provide any code snippets, but if you follow my description of the resulting data structure you should be able to get a similar chain working without any problems; as ever the main issue comes in the data collection stage.


## Markov Chains

Along with being a very handy tool for sampling from posterior distributions, markov chains also find use in time series analysis, defining state transitions as functions of the current state. I've used recurrent neural networks and reservoir computing before for time series analysis, whereby a network tries to map the dynamics of the underlying data producing mechanism, but markov chains are fundamentally simpler (which comes with its own set of pros and cons). Instead of trying to map dynamics, they instead describe a set of state transitions, where the next state is only a function of the current state. While they have been used for high resolution continuous data, I can imagine that markov chains struggle owing to the large number of potential states found when discretising continuous data. RNNs on the other hand excel for such problems. When dealing with discrete data without too many potential values, markov chains then prove more suitable.

One particular application is in textual analysis, which is time series problem where the data is in the form of single letters or words, and so the number of potential states is relatively low (26 letters + symbols if analysing the text on a per character basis, or several hundred to thousand if using whole words). In this post I'm going to describe generating text (song lyrics to be precise) using markov chains as it's a fun application with meaningful output.

For textual analysis, markov chains are very simple unlike RNNs, since you'd need to know in advance the possible system outputs (i.e. letters or words), this could prove problematic. I'll be looking at applying markov chains operating at the level of whole words, although the same principles apply when analysing data character by character. 

To generate song lyrics we need to perform two steps:

  - Build up a state transition model
  - Initialise the chain to a random state and run through to completion


## Pre-processing the data

Before building state transition graphs, you need to have your data appropriately processed. In this case I'll be using Iron Maiden's lyrics from all their songs, although any structured plain text would work.

I want my generated songs to have the same structure as real songs, i.e. similar length lines, grouped into verses, with a standard number of verses before the song finishes. I'm not going to try and get a proper verse/chorus structure functioning, although it wouldn't be too challenging - you'd need your input data to note whether a new stanza (is that the right word?!) denotes a verse or a chorus, then you can repeat your chorus stanza when needed.

To format my songs I need the text that I'm building the chain from to indicate its grammar. Since my data will be tokenized into words to build up the state transition graph, it makes sense to have special words to indicate the end of a line, verse, or song. If these words appear in my generated lyrics then I can simply apply the appropriate grammar.

In particular I'll use 'LEND', 'VEND', 'SEND' to denote line endings, verse endings and song endings. For example, here's the first verse and a half of The Trooper:

_LEND You take my life but I'll take yours too LEND You fire your musket but I'll run you through LEND So when you're waiting for the next attack LEND You'd better stand there's no turning back LEND VEND The bugle sounds as the charge begins LEND_

This encoding allows a song to be stored in a single string, while still maintaining its original structure. Combine all the songs together and you end up with an entire discography in one string.


## Building the state transition model

The next step is to generate the transition probabilities, i.e. if the current word is "turning", what should the next word be? We need to parameterise our chain here to define the current state. The parameter is called "order", and is an integer value indicating how many of the previous words should be included in this state. I.e. with an order=2, the first two states of The Trooper would be "LEND You", "You take", while with an order = 1 they are "You" and "take". To build up the chain we need to keep a record of every possible state and the following word for each of these.

An appropriate data structure for this is a hash table (I actually implemented this code in Python as it's ideal for quick tasks such as this). In this table, each key is a current state (of order `n` words), with a corresponding list of potential next words. 

To build the chain you window over the text a word at a time, recording the current state of `n` words and the next word. Eventually you'll end up with something like the following (for an n=2 chain) on The Trooper lyrics as above:

{ 
  'LEND You': ['take'],
  'You take': ['my'],
  'take my': ['life'],
  ...
  'LEND The': ['mighty', 'screams'],
  'but I'll': ['take', 'run']
}

Once you have built up a dictionary of states you're ready to generate new songs.

## Running the model

To run the model you simply start at a random state (I limit this choice to states beginning with 'LEND' to increase the chances of getting a realistic song), and then select a random next word from the list of possible next states, then using your new current state generate the next state and so on.

As can be imagined, some phrases have a larger number of potential next words than others. If you use a high order than the list of next states will greatly diminish unless you have an extremely large training data set. With song lyrics there isn't a large amount of text, using any order greater than 2 can lead to simply reproducing entire existing songs. For example, if we used order 5, there's most likely only going to be one next state to the current state of "You take my life but" in the entire Iron Maiden discography.


## Example Songs

Now the bit you've all been waiting for, some actual generated lyrics rather than just me droning on!

Here's an example song using a chain of **order one**:

_It wasn't meant for granted 
The evil that has won 
There is my eyes red and no names 
Come the plot 
Comrades dead 
I disappeared so long to hammer them no tales 
There are preparing for my side_

_Don't tell me_

_Packed your mind begins to their turn your deepest space 
Oh Lamia please try to be 
Or some things that who fired first 
As I feel that I go_

_Shadows and shame 
Water water fire_

_If I just can't go down on my soul to say 
It's meaningless and ran like a man 
Don't be free man 
That I ever shall be_

_He knocks you hide yourself a statue carved in your time well_

_Gone are sealed and then we'll jump right into hearts 
I cannot think your back 
To stand with a beer 
Nobody has no life how I don't know where they're saying 
Then on my only me that men fear show the night 
Now a lonely satellite circus just trying to whet the devil himself come your daughter to sleep when man_

_And the mixed emotion and the money 
Nomad where eagles dare._

This is a nice example as while we can pinpoint certain sections to certain songs (well if you listen to as much Maiden as I do you can), but it's still novel enough to be interesting.

Whereas using a chain of **order two** there are more sections of lyrics copied straight from existing songs, such as this:

_Judas my guide_

_Out of the game 
No point asking who's to go over the hill 
They've put the blame unto me 
Telling me to get ready and waiting 
Or it could be someone else 
Is this a dream or is it me I lost my dreams rip the bones 
How come the demons of creation 
Out of the citadel_

_Out beyond the hill 
As you plunge into a new toll_

_There are no errors in the dark of the dead_

_Dusty dreams in fading daylight 
Till the TV 
Don't I believe you have to die 
Us or them-a well rehearsed lie_

_Face at the open door 
Just let yourself go 
Bring your daughter to the clans_

_Viking raiders from afar._

While a chain of **order five** posts the entirety of Moonchild... Guess there aren't too many occurrences of the state "For all the sins you"!

_When Gabriel lies sleeping this child was born to die_

_One more dies one more lives 
One baby cries one mother grieves 
For all the sins you will commit 
You'll beg forgiveness and none I'll give 
A web of fear shall be your coat 
To clothe you in the night 
A lucky escape for you young man 
But I'll see you damned in endless night_

_Moonchild hear the mandrake scream 
Moonchild open the seventh seal 
Moonchild you'll be mine soon child 
Moonchild take my hand tonight._

## Summary

I hope that's been a simple overview of generating text with markov chains. Obviously they're used for far more complex purposes than producing poorly written Iron Maiden songs, but it's a neat concept with many tangible outputs, you just need to source a lot of data to generate anything meaningful.

I haven't provided any code, but if you understand how the chains are fundamentally stored then it shouldn't prove challenging to build your own, provided you use an appropriate data structure.

