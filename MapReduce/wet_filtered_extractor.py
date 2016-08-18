<<<<<<< HEAD
import re
from mrjob.job import MRJob
from mrjob.step import MRStep
import subprocess

class CCJob(MRJob):
    
    def gz_fetch(self,f):
        ## Makes system call to network gzip file stream
        ## This enables us to not host files locally and to process them extremely quickly
        ## Using the native system library vs python results in 5-7x speedup in empiric tests
        
        # Construct the system call we're using based on the paths file
        if type(f) is str and len(f)>0: aws_str = "zcat <(aws s3 cp s3://commoncrawl/%s --no-sign-request -)" % f
            
        # Open a line-buffered stream to the gzip file stream
        data = subprocess.Popen(aws_str, stdout=subprocess.PIPE, bufsize=1, shell=True, executable = "/bin/bash")
        for line in data.stdout:
            yield line

    def configure_options(self):
        # Specify the name of the file containing the domains we want to keep
        super(CCJob, self).configure_options()
        self.add_passthrough_option('--domain-key', help="Specify the file containing domains to keep")
      
    def mapper(self, _, line):
        # Create domain inclusion array using pregenerated file (step 1)
        incl_domains = []
        with open(self.options.domain_key, "r") as f:
            for d in f:
                incl_domains.append(d.strip())
        # Transform array into set for O(1) access
        incl_domains = set(incl_domains)

        # Core markers of a WARC file for custom WARC processing
        # This process is essential for making this problem tractable on a budget
        warc_start = "WARC-Target-URI"
        content_start = "Content-Length"
        content_end = 'WARC/1.0'
        content_type = "Content-Type"
        
        # Bookkeeping Variables
        content_rolling = False # True when scanning through WARC page content
        warc_buffer = "" # String that rolls up content we're interested in
        warc_uri = "" # URI of the page we're working on
        warc_domain = "" # Domain of the page we're working on
        url = "" # URL after the domain we're working on
        keep = False
        content_flag = False

        # Iterate over lines
        for line in self.gz_fetch(line):
            uline = line.decode('utf-8', errors = 'ignore')
            self.increment_counter("Custom","lines_parsed",1)
            if uline[:-2] == content_end:
                # yield WARC content that we've collected if we're seeing the next record
                if keep: # Based on if we have an address in the list
                    try:
                        warc_buffer.replace("\t", "    ")
                        yield warc_domain+url, warc_buffer.encode('utf-8', 'ignore')
                    except: pass 
                addr_flag = False
                content_rolling = False
                warc_buffer = ""
                warc_uri = ""
                warc_domain = ""
                url = ""
                keep = True
                
            elif uline[:12] == content_type:
                content_flag = uline[14:] == "text/plain"
                
            elif uline[:15] == warc_start and content_flag:
                # If the line indicates the start of a new record, capture and parse the URI
                warc_uri = uline[17:]
                # Split the domain from the relative page
                domainsplit = warc_uri[8:].find("/")+8
                url = warc_uri[domainsplit:]
                # Remove the protocol tag and the www. if we find it
                t1 = warc_uri[:domainsplit].strip()
                t2 = t1[t1.find("//")+2:]
                if t2[:4]==("www."):
                    warc_domain = t2[4:]
                else: warc_domain = t2
                
                # Toggle the "keep" flag based on if our parsed domain is one we want
                if warc_domain in incl_domains:
                    keep = True
                else: keep = False
                    
            elif not keep: continue # Don't roll up content if we've already determined we're not going to keep it
                
            
            elif uline[:14] == content_start:
                # Determine if our record actually has content. If it has a content length, set the rolling flag
                # and start grabbing lines
                content_rolling = True

            elif content_rolling:
                # Roll up content into a single string. Cap it if it's unreasonably long.
                if len(warc_buffer)>10000: continue
                warc_buffer += line.strip()+'\\n'
        try:
            yield warc_domain+url, warc_buffer.encode('utf-8', 'ignore')
        except: pass   
        
    def steps(self):
        return [MRStep(mapper=self.mapper,
                       reducer = None,
                       # Set the timeout for map jobs really high. They take way longer than hadoop expects.
                       jobconf={'mapreduce.task.timeout':'6000000'}
                      )]


if __name__ == '__main__':
=======
import gzip
import os.path as P
import boto
import warc
import tldextract
import re
from collections import Counter
from boto.s3.key import Key
from gzipstream import GzipStreamFile
from mrjob.job import MRJob
from mrjob.step import MRStep

####################
# cc-mrjob library #
####################
## NOTE: We extended the CCJob class found here: https://github.com/commoncrawl/cc-mrjob/blob/master/mrcc.py. Rather,
## than inherit from it we just added our code into it as importing multiple files to Hadoop is always a pain. Also,
## we had to make some changes to get it to run in EMR. I got those here:
##   https://github.com/Smerity/cc-mrjob/pull/8/commits
class CCJob(MRJob):
  def steps(self):
    return [MRStep(mapper_init=self.mapper_init,
                   mapper=self.mapper)]

  # Read the urls file and create dictionary out of it
  def mapper_init(self):
    self.address_pattern = re.compile("berkeley,? ca\.? 947", re.IGNORECASE | re.DOTALL).search
    self.urls = {}

    f = open("urls.txt", "r")
    for line in f:
      line = line.strip().split('\t')
      domain = line[0]
      pages = int(line[1])
      characters = int(line[2])
      more_than_1k = (pages > 1000)

      # Example of things we can filter out
      if pages > 50000 or characters > 10000000:
          continue

      self.urls[domain] = [more_than_1k, pages, characters]

  def mapper(self, _, line):
    f = None

    ## If we're on EC2 or running on a Hadoop cluster, pull files via S3
    local_path = P.join(P.abspath(P.dirname(__file__)), line)
    if self.options.runner in ['emr', 'hadoop'] or not P.isfile(local_path):
      # Connect to Amazon S3 using anonymous credentials
      conn = boto.connect_s3(anon=True)
      pds = conn.get_bucket('aws-publicdatasets')
      # Start a connection to one of the WARC files
      k = Key(pds, line)
      f = warc.WARCFile(fileobj=GzipStreamFile(k))
    ## If we're local, use files on the local file system
    else:
      print 'Loading local file {}'.format(local_path)
      f = warc.WARCFile(fileobj=gzip.open(local_path))

    # Go through every records in the WET file and process the data #
    for i, record in enumerate(f):
      for key, value in self.process_record(record):
        yield key, value

      # Counter to track processed records
      self.increment_counter('commoncrawl', 'processed_records', 1)

  def process_record(self, record):
    # Only look at records that
    if record['WARC-Type'] != 'conversion':
      return

    # Read text contents of WET file
    text = record.payload.read()

    # Get domain
    url = record['WARC-Target-URI']
    ext = tldextract.extract(url)
    subdomain = (ext.subdomain + '.' if len(ext.subdomain) > 0 else '')
    full_domain = subdomain + ext.domain + '.' + ext.suffix

    # Only process if the url is in the url list
    if full_domain in self.urls:
      # If it has more than 1K pages...
      if self.urls[full_domain][0] == True:
        # We should only return URLs that have a Berkeley address
        if self.address_pattern(text):
          yield url, text
      else:
        yield url, text

if __name__ == '__main__':
>>>>>>> 3399ca81d8d952c2b31e9e74bff1da8773a68553
  CCJob.run()