## Introduction

### Overview
*statsrat* is a package for using mathematical learning models in psychology and neuroscience research.  It provides a framework for implementing learning models, creating simulated versions of real-world learning experiments, running and analyzing simulations and fitting models to behavioral data.  It is hoped that *statsrat* will make modeling easier for both beginners and experienced modelers.  *statsrat*'s common format for behavioral data, experimental designs and learning models will aid in reproducibility and communication.  The package is designed to save time that would otherwise be spent on tedious programming and debugging.  The *statsrat* user can construct learning models out of modular building blocks, implementing existing models or creating new ones with just a few lines of code.  Instead of painstakingly implementing a simulated version of a real-world learning experiment, someone using *statrat* simply describes the experiment's structure is the same way as a table in research paper.  *statsrat* has an array of functions for plotting, model fitting, data import and simulation that remove the need for complicated custom scripts.  In summary, *statsrat* aims to streamline modeling so that researchers can focus on science.

### Installation
(Insert this later.)

### Scope and limitations
*statsrat* implements algorithmic models of learning and decision making in both humans and non-human animals.  In other words, these models aim to mathematically describe how organisms form, modify and use representations during learning.  This level of analysis is called *algorithmic* in Marr's taxonomy of models *** CHECK AND REF *** .  While algorithmic models often provide insight into neural processes *** REFS TO EXAMPLES *** , they do not try to explicitly model the brain.  *statsrat* follows in the long tradition of looking for commonalities between the basic mechanisms of human and non-human learning *** REFS *** .  It therefore describes both human and animal experiments using a common language, allowing any learning model to be used to simulate either.  Currently, *statsrat* is limited to modeling experiments in which the sequence of predictor stimuli, outcomes etc. is pre-determined.  *statsrat* also cannot simulate tasks in which the learner selects between two or more stimuli that differ in more than perceptual dimension (i.e. a simultaneous discrimination, e.g. *** REF *** ).  It is hoped that these limitations will be removed in the future.

Here are some examples of experiments and learning models that can and cannot be implemented using *statsrat* in its current state:

Experiments that can be simulated in *statsrat*:
* human category learning *** REF ***
* Pavlovian conditioning *** REF ***
* bandit tasks *** REF ***
* animal successive discrimination tasks *** REF ***

Learning models that can be implemented in *statsrat*:
* Rescorla-Wagner and variants *** REF ***
* instance-based learning *** REF ***
* latent cause models *** REF ***

Experiments that cannot be simulated in *statsrat*:
* paired associates list learning tasks *** CHECK AND REFS ***
* multi-stage decision tasks (e.g. Markov decision processes) *** REF ***
* tasks with a performance criterion to move to the next stage (unless modified) *** REF ***

Learning models that cannot be implemented in *statsrat*:
* biologically detailed models that simulate neural dynamics *** REF ***
* deep learning *** REF ***

### Comparison to other packages
(Insert this later.)

### Introduction to package design
*statsrat* is built around three main components: learning model objects, experiment objects and a suite of functions used to perform various simulation tasks.  Here we give an overview of each of these components: more detailed information can be found in the relevant modules.  See the *examples* directory for Jupyter notebooks that show how to use *statsrat* for various simulation and data analysis tasks.

#### Learning model objects
Each learning model belongs to a class of related models that share the same basic design.  Currently, the three model classes are Rescorla-Wagner variants (*** REFS *** , module *rw*), exemplar models (** REFS ** , module *exemplar*), latent cause models (*** REFS *** , module *latent_cause*), and Bayesian regression models (** REFS ** , module *bayes_regr*).  Each model class is characterized by its *simulate* method, which outlines how that model learns and makes decisions.  To define a learning model, the user specifies various model attributes relating to stimulus processing, memory formation, learning rates etc.  See the documentation of each model class for more details.  Each model attribute is either a function or a class, and is associated with certain model parameters that are automatically added to the model definition.  Model attributes are ''plugged into'' the model's *simulate* method to specify exactly how that model works.  Predefined learning models can be found in the ''predef'' sub-modules for each model class along with descriptions and references.

**Note on the term ''model''**: One must take care to avoid confusion about how the term ''model'' is used in *statsrat*, which is different from how it is typically used in machine learning.  In machine learning, the data being modeled are a series of predictors (corresponding to cues or conditioned stimuli) and possibly a series of accompanying class labels or reward values (corresponding to experimental feedback such as category labels or unconditioned stimuli).  Thus, in machine learning a ''model'' is a particular instance of a learning architecture that has been fitted to a dataset, while quantities controlling learning rates etc.\ are considered ''hyperparameters''.  In *statsrat* and psychological learning literature more broadly, the data being modeled are a time series of cues/conditioned stimuli, feedback/unconditioned stimuli and behavioral responses.  Thus ''model'' is synonymous with ''learning architecture'' or ''theory'' and does not correspond to a particular fitted instance of an architecture.  Many learning models are based on the assumption that an organism's behavior is guided by an attempt to predict feedback/unconditioned stimuli from cues/conditioned stimuli, and hence can also be viewed from the machine learning perspective.  Hopefully this distinction will not cause too much confusion.  Throughout the documentation of *statsrat*, we shall endeavor to specify whether the term ''model'' should be understood in the psychological (cues + feedback -> behavior) or machine learning (cues -> feedback) sense, although the former is the default.

##### Running a simple simulation
WRITE THIS

##### Data structure (real or simulated trial by trial data)
WRITE THIS

#### Experiment objects
Experiment objects are the second main component of *statsrat*.  As might be surmised from the name, they are abstract representations of experimental designs.  Experiment objects are used to import real life experimental data for model fitting and other analysis, to perform ordinal adequacy tests (OATs, i.e. simulations testing a model's ability to produce a general pattern of results), and other related tasks.  Each experiment object consists of four attributes:
* response type
* a list of schedules
* a list of ordinal adequacy tests (OATs)
* explanatory notes

What follows is a brief overview of each of these items.  Please see the relevant documentation for more details.  *statsrat* comes with a variety of predefined experiments objects, and these provide good examples upon which to base the user's own experiment objects.

##### Response type
This is the type of behavioral response produced by the participant.  Currently this includes excitatory Pavlovian conditioning, inhibitory Pavlovian conditioning and discrete choices (e.g. choosing one of two levers in a Skinner box, or one of several category labels in a human classification learning experiment).

##### Schedules
A schedule object defines a sequence of conditioned stimuli/cues, responses options, unconditioned stimuli/feedback presented to the learner.  Each experiment has one or more schedule objects.  When there are multiple schedule objects, these typically correspond to separate groups in a between subjects design.   Each schedule object is defined by a list of stage objects in a way that is designed to make defining the schedule as convenient as possible.

##### Ordinal adequacy tests (OATs)
(Insert this later.)

##### Explanatory notes
This is simply some text describing the experiment, for examples giving references and summarizing empirical results.  It is optional but useful.

#### Functions for performing simulation tasks
These functions can be divided into three categories:
* plotting simulation data
* ordinal adequacy tests (OATs), i.e. analyzing data at a merely ordinal level
* importing behavioral data
* fitting a model to trial by trial behavioral data by optimizing free parameters

##### Plotting
It is often very useful to plot simulation data.  Such plots can not only display simulated behavior over time (i.e. classic learning curves) but also model representations of attention, associations, memory retrieval and so on (depending on the particular model in question).  *statsrat* has two built in functions to make such plots quickly and conveniently (they are based on the *plotnine* package, which in turn is based on *ggplot2* for R).  The *learn_plot* function is for plotting data from a single individual dataset, while *multi_plot* is for comparing different datasets.  See the relevant documenation for each function for more details.

##### Performing Ordinal adequacy tests (OATs)
WRITE THIS

##### Importing behavioral data
WRITE THIS AND PUT IT IN THE PROPER PLACE

##### Trial by trial model fitting
WRITE THIS

### Examples
In order to get people started using *statsrat*, I have included several examples of how to use it for different types of modeling work.  Each example consists of a Jupyter notebook with example code and extensive comments.  They can be found in the *examples* directory of the *statsrat* installation.  Here are brief descriptions of each of them.

1. Define several models and Pavlovian conditioning experiments, then perform simulations.  Examine model behavior using plotting functions.  Perform a between subjects ordinal adequacy test (OAT) to search each model's parameter space to determine if it can produce the same ordinal pattern as humans.

2. Test the ability of several models to reproduce the inverse base rate effect ** REF ** , an important phenomenon in human classification learning.  This includes defining the experiment and performing within subjects ordinal adequacy tests (OATs, ** REF ** ) to search each model's parameter space to determine if it can produce the same ordinal pattern as humans.  Use plotting functions to analyze why each model behaves the way it does.

3. Import trial by trial data (to avoid ethical complications, the data is simulated rather than taken from real humans) and use it to fit several models on an individual level.  This includes analysis of how long to run each optimization as well as model comparison based on goodness of fit and ability to predict future data.

Of course ordinal adequacy tests (OATs) can also be performed for Pavlovian experiments.