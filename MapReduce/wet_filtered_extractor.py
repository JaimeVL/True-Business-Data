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
  CCJob.run()