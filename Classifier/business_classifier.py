import numpy as np
import tldextract
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split, cross_val_score
from sklearn.externals import joblib

# Utility class to store both CountVectorizer and TfidfTransformer used to break up the title and content. These
# have already been fitted during training of the Classifier.
class TextVectorizer():
    def __init__(self, use_tfidf, min_df_value, norm_value='l1'):
        self.use_tfidf = use_tfidf
        self.is_fitted = False
        self.count_vect = CountVectorizer(min_df=min_df_value, analyzer='word', stop_words='english')
        self.tfidf_transformer = TfidfTransformer(norm=norm_value)

    def fit_transform(self, train):
        if self.is_fitted:
            raise ValueError('Object has already been fitted')

        self.is_fitted = True

        train = self.count_vect.fit_transform(train)

        if self.use_tfidf:
            tfidf = self.tfidf_transformer.fit_transform(train)
            return tfidf
        else:
            return train

    def transform(self, data):
        data_vect = self.count_vect.transform(data)
        if self.use_tfidf:
            return self.tfidf_transformer.transform(data_vect)
        else:
            return data_vect

class BusinessClassifier():
    # Loads classifier files trained previously
    def __init__(self):
        self.model_stage1_title = joblib.load('model_stage1_title.pkl')
        self.model_stage1_content = joblib.load('model_stage1_content.pkl')
        self.title_vect = joblib.load('title_vect.pkl')
        self.content_vect = joblib.load('content_vect.pkl')
        self.model_stage2 = joblib.load('model_stage2.pkl')
        self.model_stage3 = joblib.load('model_stage3.pkl')
        self.use_title = (self.model_stage1_title != None and self.title_vect != None)
        self.use_content = (self.model_stage1_content != None and self.content_vect != None)

        if not self.use_title and not self.use_content:
            raise ValueError('Title or content (or both) need to be set')

    # Calculates probability of web page being a business one.
    def process_page(self, url, text):
        # Extract all features
        content = text[:-2]
        content_length = len(content)
        title = content[:content.find('\\n')]
        title = title[:100]
        title_length = len(title)

        # Get domain
        ext = tldextract.extract(url)
        subdomain = (ext.subdomain + '.' if len(ext.subdomain) > 0 else '')
        full_domain = subdomain + ext.domain + '.' + ext.suffix
        relative_page = url[url.find(full_domain)+len(full_domain):]
        url_depth = len([i for i, letter in enumerate(relative_page[:-1]) if letter == '/'])

        features = [url_depth, title_length, content_length]

        # Vectorize title and run stage 1 title model if appropiate
        if self.use_title:
            title_transformed = self.title_vect.transform([title])
            title_prob = self.model_stage1_title.predict_proba(title_transformed)[:,1][0]
            features += [title_prob]

        # Vectorize content and run stage 1 content model if appropiate
        if self.use_content:
            content_transformed =self.content_vect.transform([content])
            content_prob = self.model_stage1_content.predict_proba(content_transformed)[:,1][0]
            features += [content_prob]

        # Run stage 2 model
        page_prob = self.model_stage2.predict_proba(features)[:,1][0]
        page_result = self.model_stage2.predict(features)[0]

        # Return stage 2 results
        return full_domain, [page_prob, page_result]

    # Calculates probability of website/domain being a business one. Needs to collect results from the page calculated
    # using the process_page() function.
    def classify_domain(self, page_results):
        count_business_pages = 0
        sum_probs = 0

        # Go through all page results and calculate aggregate features
        for result in page_results:
            page_prob = result[0]
            page_result = result[1]

            if page_result == 1:
                count_business_pages += 1

            sum_probs += page_prob

        # Finish aggregating features
        perc_business_pages = count_business_pages / (len(page_results) * 1.0)
        avg_prob = sum_probs / (len(page_results) * 1.0)

        # Run stage 3 model
        return self.model_stage3.predict([perc_business_pages, avg_prob])

# Shows a simple way to call the BusinessClassifier
def main():
    classifier = BusinessClassifier()
    domain_info = {}

    count = 0
    with open('data_per_url.txt') as f:
        for line in f:
            count += 1
            if count > 300:
                break

            if count % 10 == 1:
                print count

            # Split row into two parts
            values = line.split('\t')
            domain, page_results = classifier.process_page(values[0], values[1])

            if domain in domain_info:
                domain_info[domain] += [page_results]
            else:
                domain_info[domain] = [page_results]

    with open('classifier_test.txt', 'w+') as w:
        for domain in domain_info:
            result = classifier.classify_domain(domain_info[domain])
            w.write(domain + '\t' + str(result[0]) + '\n')

if __name__ == '__main__':
    main()