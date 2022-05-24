# Options for unsupervised techniques 

1. Autoencoder architecture, using latent representation for clustering. After clustering I can label all instances in each cluster with the predominant label class(es). 
    - One limitation is that this still relies on the labels we already have. If the model finds some novel relationship that helps in the reconstruction task, it is not necessarily given that this feature would align with the label classes. E.g. labels only state a few classes, missing classes such as flow noise, waves, all/any geological sound sources, and most notably no instances of self-induced noise by the gliders (which is LOUD!)
    