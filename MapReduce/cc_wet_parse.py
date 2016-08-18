
from mrjob.job import MRJob
from mrjob.step import MRStep
import subprocess
import re

class wet_parse(MRJob):
## Iterates over all of common crawl, looking for pages with Berkeley webpages
## Collects summary statistics in the form <domain, [total_pages, total_berk_addr_pages, total_characters]>
## This is Step 1 in the pipeline to collect all web pages from domains containing Berekeley addresses
    
    def gz_fetch(self,f):
        # Construct string to pass to aws cli, gunzip
        if type(f) is str and len(f)>0: aws_str = "zcat <(aws s3 cp s3://commoncrawl/%s --no-sign-request -)" % f

        # Open a line-buffered stream to the gzip file stream
        data = subprocess.Popen(aws_str, stdout=subprocess.PIPE, bufsize=1, shell=True, executable = "/bin/bash")
        for line in data.stdout:
            yield line
    
    def mapper(self, _, line):        
        # Core markers of a WARC file for custom WARC processing
        # This process is essential for making this problem tractable on a budget
        warc_start = "WARC-Target-URI"
        content_start = "Connection: close"
        content_end = 'WARC/1.0'
                
        # Compiled regular expression to match Berkeley addresses
        addr_pattern = re.compile("berkeley,? ca\.? 947", re.IGNORECASE|re.DOTALL).search   
                
        # Bookkeeping variables
        content_rolling = False
        warc_domain = ""
        berk_page = False
        content_len = 0        
        
        for line in self.gz_fetch(line):
            if line[:-2] == content_end:                
                # Always emit 1 record per page
                try:
                    if berk_page:
                        yield (warc_domain, [1,1,content_len])
                    else: yield (warc_domain, [1,0,content_len])
                except UnicodeDecodeError:
                    pass
                
                # Bookkeeping
                berk_page = False
                content_len = 0
                content_rolling = False
                
                # Clear warc boffer, domain, uri
                warc_uri = ""
                warc_domain = ""
                
                title_captured = False
                
            elif line[:15] == warc_start:
                # Capture new address                
                warc_uri = line[17:]
                domainsplit = warc_uri[8:].find("/")+8                    
                # Parse domains, stripping protocol tag and www.
                t1 = warc_uri[:domainsplit].strip()
                t2 = t1[t1.find("//")+2:]
                if t2[:4]==("www."):
                    warc_domain = t2[4:]
                else: warc_domain = t2
            
            elif line[:14] == content_start:
                # Read WARC count of valid characters, keep track of it
                content_rolling = True
                try:
                    content_len = int(line[16:])
                except:
                    content_len = 0
                
            elif content_rolling:                    
                # If we see a Berkeley address, flag the page
                if addr_pattern(line):
                    berk_page = True
                    
        # yield warc content if reached next recordtry:    
        try:
            if berk_page:
                yield (warc_domain, [1,1,content_len])
            else: yield (warc_domain, [1,0,content_len])
        except UnicodeDecodeError:
            pass
                
    def combiner(self, key, value):
        # Combiner is pretty important. Do as much aggregation as possible
        pages, b_pages, content_len = value.next()
        for p, b, c in value:
            pages+=p
            b_pages+=b
            content_len+=c
        yield key, [p,b,c]

            
    def reducer(self, key, value):
        # Reducer just adds up all the counters for each domain, outputting a record if we found at least 1 berkeley address
        pages, b_pages, content_len = value.next()
        for p, b, c in value:
            pages+=p
            b_pages+=b
            content_len+=c
        if b_pages>0:
            yield key, [pages, b_pages, content_len]

    def steps(self):
        return [MRStep(mapper=self.mapper,
                     reducer=self.reducer,
                       jobconf={'mapreduce.task.timeout':'6000000'}
                      )]
    
if __name__ == '__main__':
    wet_parse.run()