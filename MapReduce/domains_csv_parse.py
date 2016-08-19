import sys
import subprocess
f_name = sys.argv[1]
# Gather the parsed domains from step 1, and figure out which ones to keep
subprocess.call("/root/hadoop*/bin/hdfs dfs -cat %s/* > %s.csv" % (f_name,f_name), shell=True)
ifile = f_name
ofile = f_name+"-parsed.csv"
with open(sys.argv[1]+'.csv','r') as f, open(ofile,'w') as of:
    for line in f:
        domain, data = line.split('\t')
        domain = domain.strip('"')
        _, _, content = eval(data)
        # Threshold domains at 90% of the total crawl mass, determined offline
        # Also remove non-business domain endings
        if content<222130075 and domain[-3:]!='edu' and domain[-3:]!='gov':
            of.write(domain+'\n')
s = f_name[:4]
subprocess.call("/root/hadoop*/bin/hdfs dfs -rm -r %s-data" %s, shell=True)
subprocess.call("python wet_filtered_pull.py hdfs:///user/root/inputs/%s.wet.gz-splits/* -r hadoop --cleanup TMP --output-dir %s-data --no-output --file %s --domain-key %s --jobconf mapreduce.job.reduces=16"%(s,s,ofile,ofile) , shell=True)