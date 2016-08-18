![alt text](logo.png "Open for business")

## Overview
Basic introduction. This is part of the Capstone Project for the UC Berkeley MIDS master's degree program.

To learn more about this project, please visit our GitBook located here: [link](https://tracsj.gitbooks.io/true-business-data/content/). This assumes you've read the GitBook so some topics will not be covered here and we will instead focus on the code samples and general information on this repo. Finally, you can also find additional information on our website: [link](https://google.com).

## Sections
Here's a brief description of each section in this repo:

* **Site** - Contains all HTML, CSS, and JS code and resources used on our website.
* **MapReduce** - This contains all the Python code we used to run our Hadoop jobs against the Common Crawl.
* **Classifier** - Includes all the Python code used to train the classifier, choose the best model, and package it so it could use it on any arbitrary dataset. This last part was built into a separate class so we could include it in our MapReduce code.
* **Data** - Contains the business listings, as well as the labeled data collected.

## Site
*STEPHEN TO ADD BRIEF DESCRIPTION AND DETAILS*

## MapReduce
We'll start by providing an introduction on how to access the Common Crawl data before providing details on the final sets of scripts we used. You can find additional information on the Common Crawl [here](http://commoncrawl.org/the-data/get-started/). Note that we built on top of the resources available there, in particular the [cc-mrjob](https://github.com/commoncrawl/cc-mrjob/blob/master/mrcc.py) package. However, we had to make some modifications to get it to work.
   
While we experimented with both WARC and WAT files, we ultimately focused on WET files given they are smaller in size and contained all the text data we needed. 

### Processing WET files
Below we provide more details on how to run an AWS job over the WET files using the two sample files we used to extract data and collect statistics in the early stages of this project. Here's a brief description of each one:

* **wet_word_stats.py** - Collects counts of relevant words regularly tied to businesses per domain/website
* **wet_filtered_extractor.py** - Returns all web pages that are within the supplied 'url.txt', which contains all websites with a Berkeley address. Note that if a website contains more than 1K pages or 1M characters it will only return those with a Berkeley address to reduce the data size. This was required as our intial pull was over 1 TB.   

### Setup
To develop locally, you will need to install the following packages:
 
* `mrjob` - Hadoop streaming framework
* `boto` - library for AWS
* `warc` - library for accessing the web data
* `gzipstream` - library to allow Python stream decompress gzip files
* `tldextract` - used to split the URL into parts

This can all be done using `pip`:

    pip install -r requirements.txt

### Running locally
Running the code locally is made incredibly simple thanks to `mrjob`. Developing and testing your code doesn't actually need a Hadoop installation. First, you'll need to get the relevant demo data locally, which can be done by running:

    ./get-data.sh
    
To run the jobs locally, you can simply run:

    python absolutize_path.py < input/test-1.wet | python wet_word_stats.py --conf-path mrjob.conf --no-output --output-dir out

### Running via Elastic MapReduce (EMR)
As the Common Crawl dataset lives in the Amazon Public Datasets program, you can access and process it without incurring any transfer costs. The only cost that you incur is the cost of the machines and Elastic MapReduce itself. 

To run the job you simply run the command shown below. Just make sure to input your AWS credentials in the `mrjob.conf` file, and also make sure to specify an S3 path you have write access to:

    python wet_word_stats.py -r emr --conf-path mrjob.conf --no-output --output-dir=s3://jvl-mids-w210/test input/test-1.wet

### Code used to generate final Business Listings 
*MICHAEL TO ADD MORE DETAILS. FEEL FREE TO CHANGE NAME TO SOMETHING BETTER! :)*

## Classifier
We trained our business classifier using Python and the `scikit-learn` package to. Before providing more details on this, here's a brief description of the two scripts we wrote:

* **classifier_trainer.py** - Used to train the 3 stage logistic regression ensemble. It includes code to shuffle multiple datasets used to train different models and output the results in human and machine readable form. Finally, it includes a method to store the classifier as a set of pickled files so it can be loaded by a separate class.
* **business_classifier.py** - Once the model is trained and stored as a set of pickled files, this uses it to classify any web page and also provide a method to receive all results from a website/domain to generate the final classification. 

### Trainer
This is the script we used to explore different classification tecniques and compare models. It contains a main class, **BusinessClassifierTrainer** that includes most of the logic to train the models, as well as a **main()** function that is used to drive execution and opens up a bunch of options through arguments passed in the command line when running the script. It contains the following features:

* Imports and shuffles labeled data to generate 4 separate datasets used to test different models. The first two have one train and one test set, while the second two have 3 train sets and one test set. These are used to test two models with slightly different features. For example, we explored models using TF-IDF or a regular CountVectorizer on different copies of the same data.
    * The reason we used 3 train sets is to avoid having data used to train the stage 1 model and then re-used by that same model to generate the input for the stage 2 model. In practice, this caused more harm by reducing the already limited labeled dataset even further. Still, we expect this to perform better with more data, and it was useful having the ability to test this out.    
    * Also note that the shuffling of data is done by domain/website, so that we don't have domains with some web pages in the training set, and others in the test set.  
* Allows the caller to specify the train/test ratio, but also a custom one to create more tha one split. For example you can split the data into 20%, 15%, 50%, 15% sets as long as they all add up to 100%.
* Provides multiple knobs to control different settings used to train and compare models. Here are those options:
    * Use clean text - Uses regular expressions to make lowercase, remove digits and non-standard characters
    * Use labels for training - This refers to using 1s and 0s to train the stage 2 classifier instead of the stage 1 classifier (i.e. probabilities)
    * Use title - Includes the vectorized title as a feature
    * Use content - Includes the vectorized content as a feature
    * Compact form - More details below
* Has methods to display results in human, but also machine readable form. In machine readable form the prints statements are turned off and a single line per dataset shows the following results:
    * Settings used to classify models
    * Accuracy, Precision, Recall, and F1 score
    * Confusion matrix
* Allows you to train multiple models or a single one using all the data. If a single one is specified, which is what you use to train the final classifier, then it stores the pickled files so that you can run it elsewhere. See section below for more details. 
* Uses `pandas` DataFrames to store and extend data. This is particularly useful given there are 3 stages that each build on top of the previous one.

### Business Classifier
Using the classifier is pretty straighforward. The script contains a good example in the main() function that should be self-explanatory. The one thing to be aware is that you need to get the pickled files mentioned in the previous section and copy them over to the folder which you plan on running the script. 

## Data
This section includes the different data we collected or used during this project.

### Business Listings
*PROVIDE DETAILS ON BUSINESS LISTINGS*

### Labeled data
We labeled close to 900 websites that we were able to use for training the classifier. We split the data into two files:

* **labeled_domains.txt** - This contains a list of domains/websites along with a 1 if it's a business, and a 0 if it's not.
* **labeled_berkeley_domains.txt** - Same as above, with the distinction that 1s are only applied to business with a presence in Berkeley.