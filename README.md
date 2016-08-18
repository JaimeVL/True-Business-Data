![alt text](logo.png "Open for business")

## Overview

Basic introduction.

To learn more about this project, please visit our GitBook located here: [link](https://tracsj.gitbooks.io/true-business-data/content/). Also you can find additional information on our website: [link](https://google.com).

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

Running the code locally is made incredibly simple thanks to MrJob. Developing and testing your code doesn't actually need a Hadoop installation. First, you'll need to get the relevant demo data locally, which can be done by running:

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

Along with the scikit-learn package to...

## Data

*Placeholder*