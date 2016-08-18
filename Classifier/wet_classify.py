import re
from mrjob.job import MRJob
from mrjob.step import MRStep
import subprocess
from business_classifier import BusinessClassifierUtility, TextVectorizer

class CCJob(MRJob):

    def mapper_init(self):
        self.classifier = BusinessClassifierUtility()        
        self.addr_pattern = re.compile("berkeley,? ca\.? 947", re.IGNORECASE|re.DOTALL).search
        self.street_pattern = re.compile("\s(hall|ave|avenue|street|st|road|rd|way|wy|dr|drive)" , re.IGNORECASE|re.DOTALL).finditer
    
    def pull_address(self,line):
        address = ""
        
        match = self.addr_pattern(line)
        if match:
            addr_l2_s, addr_l2_e = match.span()
            addr_l2_e += 2
            line1 = line[:addr_l2_s]
            street_match = 0
            for street_match in self.street_pattern(line1):
                pass            
            if street_match:
                addr_l1_s, addr_l1_e = street_match.span()
                digit_flag = False
                i = -1
                for c in line1[addr_l1_s::-1]:
                    if c in "1234567890":
                        digit_flag = True
                    elif digit_flag:
                        address = line[addr_l1_s-i:addr_l2_e]
                        break
                    elif i>30: break
                    i+=1
                if digit_flag:
                    address = line[addr_l1_s-i:addr_l2_e]
            else: address = line[addr_l2_e-23:addr_l2_e]
        return address[-100:]
        
    def mapper(self, _, line):
        values = line.split('\t')
        domain, page_results = self.classifier.process_url(values[0], values[1])
        address = self.pull_address(line)
        yield domain, (page_results, address)
        
    def reducer_init(self):
        self.classifier = BusinessClassifierUtility()
        
    def reducer(self, key, values):
        val_array = []
        address_list = set()
        for v in values:
            try:
                page_prob, address = v
                v0, v1 = page_prob
                val_array.append([v0,v1])
                if len(address)>0:
                    address_list.add(address)
            except IndexError:
                continue
            
        if len(val_array)>0:
            yield key, (self.classifier.classify_domain(val_array)[0], {"addresses": list(address_list)})
        
    
    
    def steps(self):
        return [MRStep(mapper_init=self.mapper_init,
                       mapper=self.mapper,
                       reducer_init=self.reducer_init,
                       reducer = self.reducer,
                      )]


if __name__ == '__main__':
  CCJob.run()
