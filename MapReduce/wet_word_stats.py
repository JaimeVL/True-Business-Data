import gzip
import os.path as P
import boto
import warc
import tldextract
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
    return [
      MRStep(mapper=self.mapper,
             combiner=self.reducer,
             reducer=self.reducer)
    ]

  def mapper(self, _, line):
    f = None

    # If we're on EC2 or running on a Hadoop cluster, pull files via S3
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

    for i, record in enumerate(f):
      for key, value in self.process_record(record):
        yield key, value
      self.increment_counter('commoncrawl', 'processed_records', 1)

  # Contains logic to reduce values sent by the process_record function
  def reducer(self, key, values):
    total = None

    # Iterates on all values sent with this key and sums the count of each element resulting in a total count of words.
    for value in values:
      if total == None:
        total = value
      else:
        total = map(lambda x,y: x + y, total, value)

    yield key, total

  # Count the word frequency for some key words, including an "Others" bucket
  def process_record(self, record):
    if record['WARC-Type'] != 'conversion':
      return

    # Read text contents of WET file
    text = record.payload.read()

    # Count each word and add it to proper bag
    counts = [0,0,0,0,0,0]
    for word, count in Counter(text.split()).iteritems():
      lword = word.lower()

      if lword == 'company':
        counts[0] += count
      elif lword == 'business':
        counts[1] += count
      elif lword == 'market':
        counts[2] += count
      elif lword == 'customer':
        counts[3] += count
      elif lword == 'satisfaction':
        counts[4] += count
      else:
        counts[5] += count

    # Get domain
    url = record['WARC-Target-URI']
    ext = tldextract.extract(url)
    subdomain = (ext.subdomain + '.' if len(ext.subdomain) > 0 else '')
    full_domain = subdomain + ext.domain + '.' + ext.suffix

    yield full_domain, counts

if __name__ == '__main__':
  CCJob.run()
