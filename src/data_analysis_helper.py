from lxml import etree
import nltk
from nltk.text import Text
import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from openai import OpenAI
import json

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

    def _is_record_complete_enough(self, record: dict, threshold: float = 0.5) -> bool:
        '''
        Helper function for parse_publications()
        it checks if record has enough non-None values.
        Returns True if filled fields ratio is >= threshold.
        '''
        values = list(record.values())
        filled = sum(1 for v in values if v != "None")
        return filled / len(values) >= threshold

    def parse_publications(self, year_range: list[str], target_limit: int,
                           threshold = False) -> dict:
        '''
        Parses .xml file's to dict. It pics records based on
        input year_range and runs until target_limit is reached - it prevents RAM errors.
        - threshold: if you want to place threshold of how many not NaN's you want in your records: `threshold: True`, default: False
        - target_limit: limit of how many records per year you want
        '''
        context = etree.iterparse(self.path, events=("end",), load_dtd=True)
        
        # Keys are going to be years from year_range
        publications = {year: [] for year in year_range}
        completed_years = set() # set for optimization
        counter = 0 
        
        # Main tags for analysis
        target_tags = {"article", "book", "inproceedings"}
        
        # Sub tags for analysis
        fields_to_extract = ["title", "year", "address", "journal", "booktitle", "month",
                            "publisher", "note", "publnr", "rel"]

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

                        # Skips record if any field has no value
                        if threshold and not self._is_record_complete_enough(record, threshold=0.75):
                            elem.clear()
                            continue

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

    def chunking_data(self, text, chunk_size=5000, overlap=0):
        chunks = []
        for i in range(0, len(text), chunk_size - overlap):
            chunks.append(text[i:i + chunk_size])
        return chunks
    
    def RAG_sub_agent(self, question, embedding_model, api_key):
        ai_client = OpenAI(api_key=api_key)

        user_prompt = f"""
        Napisz WYŁĄCZNIE kod Python dla ChromaDB:
        results = collection.query(query_embeddings=query_vector, n_results=7, where={{...}})

        Dostępne metadane: 'Year' (int), 'Author' (str), 'Publisher' (str).
        Użyj operatorów: $eq, $gt, $lt, $gte, $lte oraz logiki $and/$or jeśli potrzeba.

        Pytanie: "{question}"
        Kod:
        """

        response = ai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Jesteś modelem klasyfikującym (i subagentem) lata, autorów i typy publikacji."
                " Masz pisać zapytania query z chromadb."},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.7
        )

        raw = response.choices[0].message.content
        clean_json = raw.replace("```json", "").replace("```", "").strip()
        
        try:
            return json.loads(clean_json)
        except:
            return None 
        
    
    def RAG_pipeline(self, question, api_key, embedding_model, collection):
        # API for GPT
        ai_client = OpenAI(api_key=api_key)

        clean_query = self.RAG_sub_agent(question, embedding_model, api_key)

        # Embedding question using the same model!
        query_vector = embedding_model.encode([question]).tolist()

        # Define context for LLM input
        results = collection.query(
            query_embeddings=query_vector,
            n_results=7,
            where= clean_query
        )

        # Post process results for better LLM understanding
        context_text = "\n---\n".join(results['documents'][0])

        system_instruction = "Jesteś asystentem naukowym. Odpowiadasz wyłącznie na podstawie dostarczonego kontekstu."
        final_user_prompt = f"""Na podstawie poniższego kontekstu wybierz dokładnie 5.
        Jeśli w kontekście nie ma wystarczającej liczby artykułów, poinformuj o tym.

        KONTEKST:
        {context_text}
        PYTANIE:
        {question}
        Odpowiedź (wypunktowana lista):"""

        response = ai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": final_user_prompt}
            ],
            temperature=0
        )

        return response.choices[0].message.content        
        





                



