+++
date = 2018-03-06
draft = false
tags = ["Echo state networks", "Parkinson's disease", "publication"]
title = "Using echo state networks for classification: A case study in Parkinson's disease"
math = false
+++

I've just had some of my PhD research on adapting Echo State Networks (ESNs) for identifying Parkinson's disease published. The work describes considerations to be made when applying ESNs to classification problems, with a case study of using them to differentiate between Parkinson's Disease patients and healthy subjects based on a longitudinal positional data source. This post will briefly summarise the work, but in case you're interested the published version is available [at the publisher's website](https://www.sciencedirect.com/science/article/pii/S0933365717303482), while I've uploaded a [preprint here](/downloads/lacy_esnparkinsons_aiim_preprint_2018.pdf). There's also free access to the full article [here](https://authors.elsevier.com/a/1WgR13KEGa1e2w) for 50 days.

## Echo state networks

Echo State Networks are a form of Recurrent Neural Network (RNN) comprising a large pool of inter-connected nodes (known as the *reservoir*) with input and output layers providing external interfaces. Recurrent connections are allowed in the reservoir, resulting in a large non-linear dynamical system. This enables them to model sequential data and have been often used for time-series forecasting purposes.

![](/img/pdesn_06052018/esn.png)

One of the main advantages of an ESN over other forms of RNNs is their much simplified training procedure. This is because only the weights of the output neurons are modified during training; the reservoir itself has its connections and weights randomly initialised and are then kept constant. This allows for simple least-squares based routines to be used for model fitting, rather than unrolling backpropagation across time. However, ESNs are commonly applied for time-series modelling rather than classification, and so one aim of this work is to determine an appropriate cost function for such tasks.

## Identifying Parkinson's Disease patients

The data set that was being used in this study was a collection of movement recordings from both Parkinson's Disease patients and control subjects, obtained from a simple movement task. The subject had discrete electromagnetic sensors placed on their thumb and index and were instructed to tap as quickly and with as large amplitude as possible for a period of 30 seconds. This was then repeated using the other hand (note, this is an established test of bradykinesia, a cardinal symptom of Parkinson's Disease, see [Part III here](https://www.parkinsons.org.uk/professionals/resources/unified-parkinsons-disease-rating-scale)). We had previously extracted summary measures from the data, guided by a clinical movement disorder expert, and were interested to see whether the ESNS were able to classify Parkinson's patients as accurately as a simple classifier using these summary features.

![](/img/pdesn_06052018/waveforms.png)

## Findings

The manner in which the classification cost function was implemented had a significant impact on ESN accuracy, with the most important aspect being whether the data was pre-processed into windows (conveniently formed by the cyclic finger tapping motion) or passed in as a single series. The networks were on the whole competitive with the classifiers using the summary features and demonstrated strong discriminative ability (AUC of 0.80 compared to 0.85 for the summary features classifier). However, given that they were trained with minimal guidance, it is an encouraging result and demonstrates the potential for machine learning techniques in such applications with high-dimensional data sets. The main challenge with the use of these techniques in healthcare lies in their opaque nature, as it can be challenging to identify how they are identifying signals in the data, and thus justify their use when more familiar and interpretable methods exist.
