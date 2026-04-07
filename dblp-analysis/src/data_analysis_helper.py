from lxml import etree
import nltk
from nltk.text import Text
import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

class DataAnalysisHelper:
    """
    Helper for Data Analysis. Parses .xml file and keeps code clean.
    """

    def __init__(self, path: str, all_top_tags: tuple[str]):
        self.path = path
        self.all_top_tags = all_top_tags

        nltk.download('stopwords')
        nltk.download('punkt')
        nltk.download('punkt_tab')

    def parse_publications(self, year_range: list[str], target_limit: int = 1200) -> dict:
        '''
        Parses .xml file's to dict. It pics records based on
        input year_range and runs until target_limit is reached - it prevents RAM errors.
        '''
        context = etree.iterparse(self.path, events=("end",), load_dtd=True)
        
        # Keys are going to be years from year_range
        publications = {year: [] for year in year_range}
        completed_years = set() # set for optimization
        counter = 0 
        
        # Main tags for analysis
        target_tags = {"article", "book", "inproceedings"}
        
        # Sub tags for analysis
        fields_to_extract = ["title", "year", "address", "journal", "booktitle", "month"]

        for event, elem in context:
            if elem.tag in self.all_top_tags:
                
                if elem.tag in target_tags:
                    counter += 1
                    if counter % 100000 == 0:
                        print(f"Checked {counter} records...")

                    elem_year = elem.findtext("year")
                    elem_year = elem_year.strip() if elem_year else "None"

                    # Checks by year of publication
                    if elem_year in year_range and elem_year not in completed_years:
                        # Define record
                        record = {"type": elem.tag}
        
                        # checks if there's more than one author
                        authors = elem.findall("author")
                        if authors:
                            record["author"] = ", ".join([a.text for a in authors if a.text])
                        else:
                            record["author"] = "None"

                        # for different subtags
                        for field in fields_to_extract:
                            val = elem.findtext(field)
                            record[field] = val.strip() if val else "None"

                        # Adds new record to set
                        publications[elem_year].append(record)
                        
                        # Logs                                            
                        if len(publications[elem_year]) >= target_limit:
                            completed_years.add(elem_year)
                            print(f"Limit reached for {elem_year}....")
                                
                            if len(completed_years) == len(year_range):
                                print("Limit reached for every year. Ending....")
                                elem.clear()
                                return publications

                # Czyszczenie pamięci
                elem.clear()
                while elem.getprevious() is not None:
                    del elem.getparent()[0]

        return publications
    
    def remove_stop_words(self, tokens_series):
        '''
        For better performence, it's a good idea 
        to get rid off articles, prepositions and pronouns like "the", "and", "is" and "in".
        They don't give no sementic meaning (in this scenario, which is topic modelling and clustering)
        '''

        # Define english stopwords
        stop_words_set = set(stopwords.words('english'))
        
        # Helper function within for filtering
        # and returning applied version to every row
        def filtered_list(tokens:list) -> list:
            clean_list = []
            for word in tokens:
                if word not in stop_words_set:
                    clean_list.append(word)

            return clean_list


        return tokens_series.apply(filtered_list)


    def preprocess_for_nlp(self, df, series_name):
        '''
        Preprocessing function for making data ready for NLP usage.
        '''

        # Normalize data
        series_for_nlp = df[series_name]
        series_for_nlp = series_for_nlp.str.lower().str.replace(r'[^\w\s]', '', regex=True)

        # Tokenize data
        tokens_nlp = series_for_nlp.apply(nltk.word_tokenize)

        return self.remove_stop_words(tokens_nlp)


        





                



