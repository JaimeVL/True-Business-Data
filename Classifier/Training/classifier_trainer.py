import sys
import re
import datetime
import random
import pandas
import numpy as np
import tldextract
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, recall_score, precision_score, confusion_matrix, f1_score
from sklearn.externals import joblib

'''
Utility class to store both CountVectorizer and TfidfTransformer used to break up the title and content.
'''
class TextVectorizer():
    def __init__(self, use_tfidf, min_df_value, norm_value='l1'):
        self.use_tfidf = use_tfidf
        self.is_fitted = False
        self.count_vect = CountVectorizer(min_df=min_df_value, analyzer='word', stop_words='english') #count_vect = CountVectorizer(stop_words='english', min_df=2)
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

'''
Class used to train Business Classifier used on websites.
'''
class BusinessClassifierTrainer():
    def __init__(self, data_url):
        self.data_url = data_url
        np.random.seed(1231)
        self.settings = []

    # Import data into dataframe from TSV file containing all the required features (more details below) used for
    # training the models and returning 3 datasets which we'll be used for different models.
    def import_data(self, seed=0, fractions=1, use_clean_text=False, other_settings=[]):
        # Read file which comes in the following format:
        #    0 Label - 1 = business, 0 = not business
        #    1 Url - e.g. http://antiquiet.com/music/2012/10/soundgarden-bones-of-birds-crooked-steps/
        #    2 Relative Url - e.g. /music/2012/10/soundgarden-bones-of-birds-crooked-steps/
        #    3 Url Depth - Number of folders separating it from domain (e.g. 4 for the example above)
        #    4 Title Length - number of characters in title
        #    5 Content Length - number of characters in content
        #    6 Title
        #    7 Content
        df = pandas.read_csv(self.data_url, delimiter="\t", header=None)
        df.columns = ['label', 'url', 'relative_url', 'url_depth', 'title_length', 'content_length', 'title_temp', 'content_temp']
        df['title_temp'] = df['title_temp'].astype(str)
        df['content_temp'] = df['content_temp'].astype(str)

        # Clean title and content text if specified
        if use_clean_text:
            clean_titles = [re.sub(r'[^a-zA-Z\s]', ' ', temp.lower().replace('\\n',' ')) for temp in df['title_temp'].fillna('none')]
            clean_contents = [re.sub(r'[^a-zA-Z\s]', ' ', temp.lower().replace('\\n',' ')) for temp in df['content_temp'].fillna('none')]
        else:
            clean_titles = df['title_temp'].fillna('none')
            clean_contents = df['content_temp'].fillna('none')

        df['title'] = pandas.Series(clean_titles, index=df.index)
        df['content'] = pandas.Series(clean_contents, index=df.index)
        df = df[['label', 'url', 'relative_url', 'url_depth', 'title_length', 'content_length', 'title', 'content']]

        # Add domain column
        domains = [self.get_domain(url) for url in df['url']]
        df['domain'] = pandas.Series(domains, index=df.index)

        # Group by domain column and create dictionary with stats on each one. This is stored in self.domain_stats.
        self.set_domain_stats(df)
        domain_list = self.domain_stats.keys()

        # Update settings used in run
        self.settings += ([seed, fractions, use_clean_text] + other_settings)

        # Copy different list of domains so that the shuffling doesn't impact the next dataset since they are passed
        # by reference.
        domain_list_1 = list(domain_list)
        domain_list_2 = list(domain_list)

        # Shuffle and split data into X and y based on precanned dataset configurations or based on custom fractions.
        if fractions == 1:
            print '+ Using precanned dataset fractions with seed = ' +str(seed)
            fractions_1 = [.25, .75]
            fractions_2 = [.25, .25, .25, .25]
        elif fractions == 2:
            print '+ Using precanned dataset fractions with seed = ' +str(seed)
            fractions_1 = [.4, .6]
            fractions_2 = [.4, .2, .2, .2]
        else:
            print '+ Using custom fractions = ' +str(fractions)
            fractions_1 = fractions
            fractions_2 = fractions

        # Shuffle and split data according to choices above.
        temp_data_sets = self.shuffle_and_split_data(domain_list_1, df, seed, fractions_1)
        data_sets_1 = temp_data_sets[0]
        data_sets_2 = temp_data_sets[1]
        temp_data_sets = self.shuffle_and_split_data(domain_list_2, df, seed, fractions_2)
        data_sets_3 = temp_data_sets[0]
        data_sets_4 = temp_data_sets[1]

        return [data_sets_1, data_sets_2, data_sets_3, data_sets_4]

    # Used to get all the data in one dataset so we can train the final model with all of it.
    def get_full_data(self, use_clean_text=False):
        df = pandas.read_csv(self.data_url, delimiter="\t", header=None)
        df.columns = ['label','url','relative_url','url_depth','title_length','content_length','title_temp','content_temp']
        df['title_temp'] = df['title_temp'].astype(str)
        df['content_temp'] = df['content_temp'].astype(str)

        # Clean title and content text if specified
        # Add clean title and content columns and remove temp ones
        if use_clean_text:
            clean_titles = [re.sub(r'[^a-zA-Z\s]', ' ', temp.lower().replace('\\n',' ')) for temp in df['title_temp'].fillna('none')]
            clean_contents = [re.sub(r'[^a-zA-Z\s]', ' ', temp.lower().replace('\\n',' ')) for temp in df['content_temp'].fillna('none')]
        else:
            clean_titles = df['title_temp'].fillna('none')
            clean_contents = df['content_temp'].fillna('none')

        df['title'] = pandas.Series(clean_titles, index=df.index)
        df['content'] = pandas.Series(clean_contents, index=df.index)
        df = df[['label', 'url', 'relative_url', 'url_depth', 'title_length', 'content_length', 'title', 'content']]

        # Add domain column
        domains = [self.get_domain(url) for url in df['url']]
        df['domain'] = pandas.Series(domains, index=df.index)

        # Group by domain column and create dictionary with stats on each one. This is stored in self.domain_stats.
        self.set_domain_stats(df)

        X = df[['url', 'relative_url', 'url_depth', 'title_length', 'content_length', 'title', 'content', 'domain']]
        y = df[['label']]

        return X, y

    # Extract domain/website from URL
    def get_domain(self, url):
        ext = tldextract.extract(url)
        subdomain = (ext.subdomain + '.' if len(ext.subdomain) > 0 else '')
        return subdomain + ext.domain + '.' + ext.suffix

    # Calculates domain stats from dataframe with one row per web page. This is necessary so we can split train and
    # test data by domain, which avoids data bleeding.
    def set_domain_stats(self, df):
        # Group by domain column and get SUM and COUNT per domain
        grouped = df[['domain', 'label']].groupby(['domain'])
        domain_df = grouped.agg([np.sum, len])

        # Convert multi-index dataframe to list (so it's easier to iterate) and add each domain to a dictionary
        domain_list = zip(domain_df.index, domain_df.label['sum'], domain_df.label['len'])
        self.domain_stats = {}
        for domain_info in domain_list:
            domain = domain_info[0]
            sum_labels = domain_info[1]
            num_rows = domain_info[2]
            label = 1 if sum_labels > 0 else 0

            # Test that the labels are either all 0 or 1. The only check we need to do is to make sure whenever the
            # sum of labels is greater than 0, that it matches the number of rows. If this fails raise exception.
            if sum_labels > 0 and sum_labels != num_rows:
                raise ValueError('Domains should all be labeled the same way')

            self.domain_stats[domain] = [label, sum_labels]

    # Shuffle and split data by domain (since we don't want to mix them in train/test sets) into different fractions
    # as specified. This allow you to split data into test and train data, but also into multiple sets of train data
    # if necessary. The seed allows you to randomize the shuffle yet keep the same one for repeated runs.
    def shuffle_and_split_data(self, domains, features_df, seed, fractions):
        if sum(fractions) != 1:
            raise ValueError('Fractions need to exactly add up to 1')

        # Set seed and shuffle domains
        random.seed(seed)
        random.shuffle(domains)

        # Calculate lengths of each dataset
        num_domains = len(domains)
        lengths = []
        for fraction in fractions:
            lengths += [int(round(num_domains * fraction))]

        # Safeguard in case rounding up/down causes the last set to have > +- 1 domains. In such a case, you
        # need to pick a different set of fractions or update this code to handle this properly.
        if sum(lengths) > (num_domains + 1) or sum(lengths) < (num_domains - 1):
            raise ValueError('Lengths of each fraction should NOT be greater than (num_domains + 1)')

        # Split each dataset composed of different domains
        domain_sets = []
        accumulated_length = 0
        for i in range(0, len(lengths)):
            if i + 1 == len(lengths):
                domain_sets += [domains[accumulated_length:]]
            else:
                domain_sets += [domains[accumulated_length:lengths[i] + accumulated_length]]

            accumulated_length += lengths[i]

        # Create X and y for each dataset by only including the choosen domains and return two copies of these.
        data_sets_1 = []
        data_sets_2 = []
        for set in domain_sets:
            data_set = features_df[features_df['domain'].isin(set)]
            X_1 = data_set[['url','relative_url','url_depth','title_length','content_length','title','content','domain']].copy()
            y_1 = data_set[['label']].copy()
            X_2 = X_1.copy()
            y_2 = y_1.copy()

            data_sets_1 += [[X_1, y_1]]
            data_sets_2 += [[X_2, y_2]]

        return [data_sets_1, data_sets_2]

    # Display scores on each stage
    def display_scores(self, y_pred, y_real, caption):
        compact_form = self.settings[6]
        labels = y_real.label.values
        conf_matrix = confusion_matrix(labels, y_pred)

        if compact_form:
            print '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s' % \
                  (self.settings[0], self.settings[1], self.settings[2], self.settings[3], self.settings[4], self.settings[5],
                   accuracy_score(y_real, y_pred), recall_score(y_real, y_pred), precision_score(y_real, y_pred), f1_score(y_real, y_pred),
                   conf_matrix[0][0], conf_matrix[0][1], conf_matrix[1][0], conf_matrix[1][1])
        else:
            print '\n' + caption
            print ' - Accuracy: %f' % accuracy_score(y_real, y_pred)
            print ' - Recall: %f' % recall_score(y_real, y_pred)
            print ' - Precision: %f' % precision_score(y_real, y_pred)
            print ' - F1: %f' % f1_score(y_real, y_pred)

            labels = y_real.label.values
            print ' - Total rows: %s    Business labels: %s    Non-business labels: %s' % (len(labels), sum(labels), len(labels) - sum(labels))
            print ' - Confusion matrix:\n%s' % (confusion_matrix(labels, y_pred))
            print ''

    # Use this method to train logistic regression models.
    def train_logistic_regression(self, penalty_str, C_val, X_train, y_train, X_test=None, y_test=None, caption='', is_last_stage=False):
        y_test_probs = None
        y_test_predictions = None

        # Train logistic regression model
        model = LogisticRegression(penalty=penalty_str, C=C_val, random_state=21)
        model.fit(X_train, y_train.label.values)
        y_train_probs = model.predict_proba(X_train)[:,1]

        # If test set is provided, then run the model on the test set
        if X_test is not None and y_test is not None:
            # Run model on test set
            y_test_predictions = model.predict(X_test)
            y_test_probs = model.predict_proba(X_test)[:,1]

            # Display scores and get probabilties if compact_form option is false or it's the last stage
            compact_form = self.settings[6]
            if not compact_form or is_last_stage:
                self.display_scores(y_test_predictions, y_test, 'Logistic Regression - ' + caption)

        # Return probabilities from the train data, test data, and the logistic regression model
        return y_train_probs, y_test_probs, y_test_predictions, model

    # Fits a single stage 1 model, either on title or on content.
    def fit_stage1_model(self, is_title, use_tfidf, use_labels_for_training, X_train, y_train, X_test=None, y_test=None):
        test_data_provided = (X_test is not None and y_test is not None)

        if is_title:
            type = 'title'
        else:
            type = 'content'

        train = X_train[type]

        # Fit and transform vectorizer
        vect = TextVectorizer(use_tfidf, .0002, 'l1') #.0002
        transformed_train = vect.fit_transform(train)

        # Add transformed sparse matrices to Dataframe
        X_train[type + '_transformed'] = transformed_train

        # Do the same for the test if it's provided
        if test_data_provided:
            test = X_test[type]
            transformed_test = vect.transform(test)
            X_test[type + '_transformed'] = transformed_test
        else:
            transformed_test = None

        # Train logistic regression model on TF-IDF
        y_train_probs, y_test_probs, _, model = \
            self.train_logistic_regression('l2', 1.0, transformed_train, y_train, transformed_test, y_test, 'Stage 1 on ' + type)

        # Add features to both test and train data
        if use_labels_for_training:
            X_train[type + '_probs'] = y_train['label']
        else:
            X_train[type + '_probs'] = y_train_probs

        # Do the same for the test if it's provided
        if test_data_provided:
            X_test[type + '_probs'] = y_test_probs

        return vect, model

    # Fits the stage 1 models depending on parameters provided.
    def fit_stage1_models(self, features, X_train, y_train, X_test=None, y_test=None, use_title=True, use_content=False, use_tfidf=True, use_labels_for_training=True):
        if not use_title and not use_content:
            raise ValueError('Need to specify the title or content to use (or both)')

        title_vect = None
        title_model = None
        content_vect = None
        content_model = None

        # Fit TF-IDF vectorizer on training data and then train model for title and/or content
        if use_title:
            title_vect, title_model = \
                self.fit_stage1_model(True, use_tfidf, use_labels_for_training, X_train, y_train, X_test, y_test)
            features += ['title_probs']

        if use_content:
            content_vect, content_model = \
                self.fit_stage1_model(False, use_tfidf, use_labels_for_training, X_train, y_train, X_test, y_test)
            features += ['content_probs']

        return title_model, content_model, title_vect, content_vect

    # Runs stage 1 model on any dataset. You just need to make sure the stage 1 model has been trained.
    def run_stage1_model(self, features, X, model_stage1_title, model_stage1_content, title_vect, content_vect):
        # Only transform the title if there's a vectorizer and model for it, and also if a previous method hasn't
        # already added the probabilities to the data. Do the same for the content.
        if title_vect is not None and model_stage1_title is not None and 'title_probs' not in X.columns.values:
            title_transformed = title_vect.transform(X['title'])
            X['title_probs'] = model_stage1_title.predict_proba(title_transformed)[:,1]

        if content_vect is not None and model_stage1_content is not None and 'content_probs' not in X.columns.values:
            content_transformed = content_vect.transform(X['content'])
            X['content_probs'] = model_stage1_content.predict_proba(content_transformed)[:,1]

    # Fit stage 2 model while running stage 1 model if it hasn't been run yet.
    def fit_stage2_model(self, features, model_stage1_title, model_stage1_content, title_vect, content_vect, X_train, y_train, X_test=None, y_test=None):
        # Only apply to test data if it's provided
        if X_test is None or y_test is None:
            X_test_subset = None
        else:
            X_test_subset = X_test[features]

        # Run train data on stage 1 model if it hasn't already. The way we tell is if one of the columns is
        # 'title_probs' or 'content_probs' depending on which features are selected.
        if sum([1 if feature in X_train.columns.values else 0 for feature in features]) != len(features):
            self.run_stage1_model(features, X_train, model_stage1_title, model_stage1_content, title_vect, content_vect)

        # Train logistic regression model on UrlDepth, TitleLenght, TitleProbs
        y_train_probs, _, _, stage2_model = \
            self.train_logistic_regression('l1', 1.0, X_train[features], y_train, X_test_subset, y_test, 'Stage 2')

        return stage2_model

    # Fit stage 3 model on any dataset. Need to provide data that has already been grouped by domain.
    def fit_stage3_model(self, train, test=None):
        features = ['perc_business_pages', 'avg_prob']

        y_train = train[['label']]

        # Set y_test variable depending on whether test set is provided
        if test is not None:
            X_test = test[features]
            y_test = test[['label']]
        else:
            X_test = None
            y_test = None

        _, _, _, stage3_model = \
            self.train_logistic_regression('l1', 1.0, train[features], y_train, X_test, y_test, 'Stage 3', True)

        return stage3_model

    # Group dataframe by domain and generate new features on the results. Should only be called once stage 1 and 2 have
    # been trained.
    def group_results_by_domain(self, X):
        # Group by domain column and get COUNT and MEAN of the 2nd stage probabilites and results
        grouped = X[['domain', 'page_probs', 'page_result']].groupby(['domain'])
        domain_df = grouped.agg([len, np.mean])

        # Convert from multi-index dataframe to a list containing: domain, num pages, % of business pages, avg probability.
        domain_data = zip(domain_df.index, domain_df.page_probs['len'], domain_df.page_result['mean'], domain_df.page_probs['mean'])

        # Find label for the domain and append it to the list of domains. Also, convert from a list of tuples to a list
        # of lists.
        final_data = []
        for data in domain_data:
            domain = data[0]

            if domain not in self.domain_stats:
                raise ValueError("Can't find label for domain. Only use data that has been processed by the import_data() function.")

            final_data += [[domain, data[1], data[2], data[3], self.domain_stats[domain][0]]]

        # Convert back to dataframe
        df = pandas.DataFrame(final_data)
        df.columns = ['domain', 'num_pages', 'perc_business_pages', 'avg_prob', 'label']
        return df

    # Run stage 1 and 2 models to classify a web page. Only works if both stages have been trained.
    def classify_pages(self, features, X, model_stage1_title, model_stage1_content, title_vect, content_vect, model_stage2):
        self.run_stage1_model(features, X, model_stage1_title, model_stage1_content, title_vect, content_vect)

        # Run stage classifier and store results in DataFrame
        X['page_probs'] = model_stage2.predict_proba(X[features])[:,1]
        X['page_result'] = model_stage2.predict(X[features])

    # Run stage 3 model to classify a domain/website. Only works if stage 3 has been trained.
    def classify_domains(self, X, model_stage3):
        features = ['perc_business_pages', 'avg_prob']

        # Run stage classifier and store results in DataFrame
        X['domain_prob'] = model_stage3.predict_proba(X[features])[:,1]
        X['domain_result'] = model_stage3.predict(X[features])

    # Display stage 1 and 2 results. Used for debugging.
    def display_results(self, X_test, y_test, num_results):
        results = X_test.values
        y = y_test.label.values
        count = 0
        for i in range(0, len(results)):
            count += 1
            if count > num_results:
                break

            row = results[i]
            print 'URL: %s' % row[0]
            print 'Title: %s' % row[1]
            print 'URL Depth: %s' % row[2]
            print 'Title length: %s' % row[3]
            print 'Title probability: %s' % row[4]
            print 'Prediction: %s' % row[5]
            print 'Actual value: %s' % y[i]
            print ''

    # Display stage 3 results.  Used for debugging.
    def display_final_results(self, X_test, y_test, num_results):
        results = X_test.values
        y = y_test.label.values
        count = 0
        for i in range(0, len(results)):
            count += 1
            if count > num_results:
                break

            row = results[i]
            print 'Domain: %s' % row[0]
            print 'Number of pages: %s' % row[1]
            print '% of business pages: ' + str(row[2])
            print 'Average page probability: %s' % row[3]
            print 'Domain probability: %s' % row[5]
            print 'Prediction: %s' % row[6]
            print 'Actual value: %s' % y[0]
            print ''

'''
Driver method for traiing multiple models and comparing them
'''
def main():
    # TODO: Replace this with easier to run arguments.
    #arguments = sys.argv[1:]
    arguments = [37,1,'0','1','1','0','0','1','1']

    seed = int(arguments[0])
    test_fraction = float(arguments[1])
    use_cleantext = (arguments[2] == '1')
    use_title = (arguments[3] == '1')
    use_content = (arguments[4] == '1')
    use_labels_for_training = (arguments[5] == '1')
    compact_form =  (arguments[6] == '1')
    train_multiple_models =  (arguments[7] == '1')
    use_tfidf =  (arguments[8] == '1')

    print_line('- Start of program: %s' % datetime.datetime.now(), compact_form)

    # Run multiple models if this specified. This was used to try out different models in order to confirm our theory
    # that the results we were getting were due to variance associated with using too small of a training data set (i.e.
    # need more labeled data).
    if train_multiple_models:
        classifier = BusinessClassifierTrainer('C:\Git\TrueBusinessData\Data\TrainingData\labeled_berkeley_data_per_url.txt') #'labeled_data_per_url.txt'
        data_sets = classifier.import_data(seed, test_fraction, use_cleantext, [use_title, use_content, use_labels_for_training, compact_form])

        print_line('\n** BASIC MODEL 1 WITH TF-IDF **', compact_form)
        train_basic_model(classifier, data_sets[0][0][0], data_sets[0][0][1], data_sets[0][1][0], data_sets[0][1][1],
                          use_title, use_content, True, use_labels_for_training, compact_form)
        return

        print_line('\n** BASIC MODEL 2 WITHOUT TF-IDF **', compact_form)
        train_basic_model(classifier, data_sets[1][0][0], data_sets[1][0][1], data_sets[1][1][0], data_sets[1][1][1],
                          use_title, use_content, False, use_labels_for_training, compact_form)

        print_line('\n** COMPLEX MODEL 3 WITH TF-IDF **', compact_form)
        train_complex_model(classifier, data_sets[2], use_title, use_content, True, compact_form)
        print_line('\n** COMPLEX MODEL 4 WITHOUT TF-IDF **', compact_form)
        train_complex_model(classifier, data_sets[3], use_title, use_content, False, compact_form)

    # Otherwise, train the final classifier
    else:
        train_final_classifier('C:\Git\TrueBusinessData\Data\TrainingData\labeled_berkeley_data_per_url.txt', #'labeled_data_per_url.txt'
                               use_title, use_content, use_tfidf, use_labels_for_training, use_cleantext)

    print_line('- End of program: %s' % datetime.datetime.now(), compact_form)

# Utility function to only display line if it's not in compact form (i.e. machine readable mode)
def print_line(text, compact_form):
    if not compact_form:
        print text

# Runs basic model with a simple train/tests split.
def train_basic_model(classifier, X_test, y_test, X_train, y_train, use_title, use_content, use_tfidf, use_labels_for_training, compact_form):
    print_line('+ Shape of train and test data', compact_form)
    print_line(X_train.shape, compact_form)
    print_line(y_train.shape, compact_form)
    print_line(X_test.shape, compact_form)
    print_line(y_test.shape, compact_form)
    print_line('', compact_form)

    # These are the optional features we can use in stage 2
    features = ['url_depth', 'title_length', 'content_length']

    # Trains the stage 1 models using the various options provided below and updates the X_train and X_test datasets
    # with the new transformed features (e.g. title and/or content)
    model_stage1_title, model_stage1_content, title_vect, content_vect = \
        classifier.fit_stage1_models(features, X_train, y_train, X_test, y_test, use_title, use_content, use_tfidf,
                                     use_labels_for_training)

    print_line('- Fitted stage 1 model: %s' % datetime.datetime.now(), compact_form)

    # Trains stage 2 model used to classify web page.
    model_stage2 = classifier.fit_stage2_model(features, model_stage1_title, model_stage1_content, title_vect, content_vect,
                                               X_train, y_train, X_test, y_test)

    # This method classifies any page using both stage 1 and 2 models.
    classifier.classify_pages(features, X_train, model_stage1_title, model_stage1_content, title_vect, content_vect, model_stage2)
    classifier.classify_pages(features, X_test, model_stage1_title, model_stage1_content, title_vect, content_vect, model_stage2)
    classifier.display_results(X_test, y_test, 0)

    # Group results by domain to generate final_data
    final_train_data = classifier.group_results_by_domain(X_train)
    final_test_data = classifier.group_results_by_domain(X_test)
    model_stage3 = classifier.fit_stage3_model(final_train_data, final_test_data)

    # Classify domains on test data and display results
    classifier.classify_domains(final_test_data, model_stage3)
    classifier.display_final_results(final_test_data, final_test_data[['label']], 0)

    return [model_stage1_title, model_stage1_content, title_vect, content_vect, model_stage2, model_stage3]

# Runs model with 1 test set and 3 train sets. The reason to run it across 3 train sets is to use different data for
# each stage. While this reduces our train data, since each stage depends on the previous one we don't bias our models
# by running them on already trained data.
def train_complex_model(classifier, data, use_title, use_content, use_tfidf, compact_form):
    X_test = data[0][0]
    y_test = data[0][1]
    X_train_1 = data[3][0]
    y_train_1 = data[3][1]
    X_train_2 = data[2][0]
    y_train_2 = data[2][1]
    X_train_3 = data[1][0]
    y_train_3 = data[1][1]

    print_line('+ Shape of train and test data', compact_form)
    print_line(X_train_1.shape, compact_form)
    print_line(y_train_1.shape, compact_form)
    print_line(X_train_2.shape, compact_form)
    print_line(y_train_2.shape, compact_form)
    print_line(X_train_3.shape, compact_form)
    print_line(y_train_3.shape, compact_form)
    print_line(X_test.shape, compact_form)
    print_line(y_test.shape, compact_form)
    print_line('', compact_form)

    # These are the optional features we can use in stage 2
    features = ['url_depth', 'title_length', 'content_length']

    # Trains the stage 1 models using the various options provided below and updates the X_train and X_test datasets
    # with the new transformed features (e.g. title and/or content)
    model_stage1_title, model_stage1_content, title_vect, content_vect = \
        classifier.fit_stage1_models(features, X_train_1, y_train_1, X_test, y_test, use_title, use_content, use_tfidf, False)

    print_line('- Fitted stage 1 model: %s' % datetime.datetime.now(), compact_form)

    # Trains stage 2 model used to classify web page.
    model_stage2 = classifier.fit_stage2_model(features, model_stage1_title, model_stage1_content, title_vect, content_vect,
                                               X_train_2, y_train_2, X_test, y_test)

    # This method classifies any page  any Now we can uses classes single_url_proba() to first run the TF-IDF on our page level data and then predicturl()
    # to predict whether this is a business or not
    classifier.classify_pages(features, X_train_3, model_stage1_title, model_stage1_content, title_vect, content_vect, model_stage2)
    classifier.classify_pages(features, X_test, model_stage1_title, model_stage1_content, title_vect, content_vect, model_stage2)
    classifier.display_results(X_test, y_test, 0)

    # Group results by domain to generate final_data
    final_train_data = classifier.group_results_by_domain(X_train_3)
    final_test_data = classifier.group_results_by_domain(X_test)
    model_stage3 = classifier.fit_stage3_model(final_train_data, final_test_data)

    classifier.classify_domains(final_test_data, model_stage3)
    classifier.display_final_results(final_test_data, final_test_data[['label']], 0)

    return [model_stage1_title, model_stage1_content, title_vect, content_vect, model_stage2, model_stage3]

# Runs the classifier across all 3 stages using any fresh dataset.
def run_classifier(classifier, X, model_stage1_title, model_stage1_content, title_vect, content_vect, model_stage2, model_stage3):
    features = ['url_depth', 'title_length', 'content_length', 'title_probs']
    classifier.classify_pages(features, X, model_stage1_title, model_stage1_content, title_vect, content_vect, model_stage2)
    final_test_data = classifier.group_results_by_domain(X)
    classifier.classify_domains(final_test_data, model_stage3)
    classifier.display_scores(final_test_data[['domain_result']], final_test_data[['label']], 'Final Stage')

# Trains final classifier and stores the model so we can use it to classify websites outside this code.
def train_final_classifier(labeled_data_path, use_title, use_content, use_tfidf, use_labels_for_training=False, use_cleantext=False):
    classifier = BusinessClassifierTrainer(labeled_data_path)
    X, y = classifier.get_full_data(use_cleantext)

    # These are the optional features we can use in stage 2
    features = ['url_depth', 'title_length', 'content_length']

    # Trains the stage 1 models using the various options provided below and updates the X_train and X_test datasets
    # with the new transformed features (e.g. title and/or content)
    print 'Training stage 1: ' + str(datetime.datetime.now())
    model_stage1_title, model_stage1_content, title_vect, content_vect = \
        classifier.fit_stage1_models(features, X, y, None, None, use_title, use_content, use_tfidf, use_labels_for_training)

    # Trains stage 2 model used to classify web page.
    print 'Training stage 2: ' + str(datetime.datetime.now())
    model_stage2 = classifier.fit_stage2_model(features, model_stage1_title, model_stage1_content, title_vect, content_vect, X, y)

    # This method classifies any page using both stage 1 and 2 models.
    print 'Classifying pages: ' + str(datetime.datetime.now())
    classifier.classify_pages(features, X, model_stage1_title, model_stage1_content, title_vect, content_vect, model_stage2)

    # Group results by domain to generate final_data
    final_train_data = classifier.group_results_by_domain(X)
    model_stage3 = classifier.fit_stage3_model(final_train_data)

    # Store classifier
    print 'Storing classifiers: ' + str(datetime.datetime.now())
    joblib.dump(model_stage1_title, 'model_stage1_title.pkl')
    joblib.dump(model_stage1_content, 'model_stage1_content.pkl')
    joblib.dump(title_vect, 'title_vect.pkl')
    joblib.dump(content_vect, 'content_vect.pkl')
    joblib.dump(model_stage2, 'model_stage2.pkl')
    joblib.dump(model_stage3, 'model_stage3.pkl')

if __name__ == '__main__':
    main()