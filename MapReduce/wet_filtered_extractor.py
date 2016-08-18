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
      more_than_1k = int(line[1])
      characters = int(line[2])
      pages = int(line[3])

      # Example of things we can filter out
      if pages > 50000 or characters > 1000000:
          continue

      self.urls[domain] = [more_than_1k, characters, pages]

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
      if self.urls[full_domain][0] == 1:
        # We should only return URLs that have a Berkeley address
        if self.address_pattern(text):
          yield url, text
      else:
        yield url, text

if __name__ == '__main__':
  CCJob.run()