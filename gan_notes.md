# Generative Adversarial Networks

https://www.youtube.com/watch?v=AJVyzd0rqdc

Generate samples from training examples.

## Why study GAN's?
* Simulate possible futures for planning or simulated RL.
   Refer to Paper by chelsey finn
* Missing Data
	1. Semi supervised learning
* Need to create Multi model outputs
	1. Predicitng next frame in video
	2.  Super resolution
	3. Image to Image translation.
	4. 
* Realistic generation tasks

## How do Generative models work?, How do GAN compare to others?
* Most of the generative models do maximum likelihood. 
* GAN's
	1. Use a latent code
	2. Asymptotically consistent
	3. No Markov chains needed
	4. Often regarded as producing best samples

## How do GANs work?

### Adversial Nets Framework
* Generator, Discriminator.

* DCGAN Architecutre
* Generator
* Use a stride greated than 1 when using deconv architecure
* Use bathcnorm at every layer excepth for the last layer of generator network
* 

#### Generator
*  


## Tips and Tricks

* Labels improve subjective sample quality. 
	Learning conditional models gives much better samples.
* One-sided label smoothing for Discrimator
	INstead of 1 form training data use 0.9 or 0.8.
	FOr fake smaples still use 0.0
	Donot smooth generator samples.
	Will not introrduce mistkaes but reduces condifdence of the model.
	
* Important to use batch normalization.
	Batch normalization in G can cause strong intra batch correlation.
	TO fix this, we can do somethings.
		Fix a reference batch and use that for the batch norm.
		
* Balancing G and D
	Usually disrminator wins.
	THis is agood thing.
	Run update on D more than the G. Mixed results.
	
	
* Duplicating the trinaing data is form of overfitting.
	Genrator never sees a training example directly. It only sees the discriminator gradients.
	
* 
## Research Frontiers
* Convergence in GAn training
* Minibatch
* Unbrolled GAN's

## Evaluation of GAN's



## COmbining GAN's with other methods

