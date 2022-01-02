#!/usr/bin/env python
# coding: utf-8

# In[1]:


"""
Place the attack result csv's to the same directory as the script, and call the for the desired results
Possible items for model: BERT, CNN, LSTM
Possible items for defense: WLADL, CLEAN, PWWS (VAT), SEM (not applicable to Bert)
Possible items for attack: PWWS, BAE, FGA
Possible items for dataset: AG_NEWS, IMDB, YahooAnswers
"""


import pandas as pd
import numpy as np


# In[2]:


def analyze_tool(model_name="CNN", defense_name="CLEAN", attack_name="PWWS", dataset_name="AG_NEWS", num_samples=200):
    document_name = model_name + "_" + dataset_name + "_" + defense_name + "_" + attack_name
    results = pd.read_csv(document_name+".csv")
    
    orig_accuracy = (sum(results["result_type"] == "Successful") + sum(results["result_type"] == "Failed"))/num_samples
    attack_success = sum(results["result_type"] == "Failed")/num_samples
    defense_success = sum(results["result_type"] == "Failed")/(sum(results["result_type"] == "Successful") + sum(results["result_type"] == "Failed"))
    query_average = np.mean(results["num_queries"])
    
    print("FILE NAME {}".format(document_name))
    print("============================")
    print("ORIGINAL ACCURACY: {}".format(orig_accuracy))
    print("ATTACKED ACCURACY: {}".format(attack_success))
    print("DEFENSE SUCCESS: {}".format(defense_success))
    print("NUMBER OF AVERAGE QUERIES: {}".format(query_average))
    print("============================")
    text_length_counter = 0
    perturbed_word_counter = 0
    for i in range(num_samples):
        text = results["original_text"][i].split()
        pert_text = results["perturbed_text"][i].count("[[")
        perturbed_word_counter += pert_text
        text_length_counter += len(text)
    
    text_length_counter /= num_samples
    perturbed_word_counter /= num_samples
    print("AVERAGE TEXT LENGTH OF ATTACKED SAMPLES: {}".format(text_length_counter))
    print("AVERAGE NUMBER OF PERTURBED WORDS: {}".format(perturbed_word_counter))


# In[3]:


def compare_defense_to_clean_accuracy(model_name, dataset_name, defense_name):
    
    if model_name != "BERT":
        attacks = ["PWWS", "BAE", "FGA"]
    else:
        attacks = ["PWWS", "BAE"]
    
    clean = "CLEAN"
    
    clean_acc = 0
    def_acc = 0
    
    for item in attacks:
        clean_doc_name = model_name + "_" + dataset_name + "_" + clean + "_" + item
        document_name = model_name + "_" + dataset_name + "_" + defense_name + "_" + item
        clean_results = pd.read_csv(clean_doc_name+".csv")
        results = pd.read_csv(document_name+".csv")
        
        print("ATTACK {}".format(item))
        print("CLEAN {}".format(sum(clean_results["result_type"] == "Failed")/200))
        print("DEFENSE {}".format(sum(results["result_type"] == "Failed")/200))
        print("=================")
        
        clean_acc += sum(clean_results["result_type"] == "Failed")/200
        def_acc += sum(results["result_type"] == "Failed")/200
    
    res_imp = (def_acc - clean_acc)/len(attacks)
    
    print("AVERAGE ACCURACY IMPROVEMENT FOR DEFENSE {} ON MODEL {} AND DATASET {}: {}".format(defense_name, model_name, 
                                                                                     dataset_name, res_imp))


# In[4]:


def compare_defense_to_clean_query(model_name, attack_name, defense_name):
    
    datasets = ["IMDB", "AG_NEWS", "YahooAnswers"]
    
    clean = "CLEAN"
    
    clean_query = 0
    def_query = 0
    
    for item in datasets:
        clean_doc_name = model_name + "_" + item + "_" + clean + "_" + attack_name
        document_name = model_name + "_" + item + "_" + defense_name + "_" + attack_name
        clean_results = pd.read_csv(clean_doc_name+".csv")
        results = pd.read_csv(document_name+".csv")
        
        print("Dataset {}".format(item))
        print("CLEAN {}".format(np.mean(clean_results["num_queries"])))
        print("DEFENSE {}".format(np.mean(results["num_queries"])))
        print("=================")
        
        clean_query += np.mean(clean_results["num_queries"])
        def_query += np.mean(results["num_queries"])
    
    res_imp = (def_query - clean_query)/3
    
    print("AVERAGE QUERY IMPROVEMENT FOR DEFENSE {} ON MODEL {} AND ATTACK {}: {}".format(defense_name, model_name, 
                                                                                     attack_name, res_imp))
    


# In[5]:


model = "BERT"
dataset = "YahooAnswers"
defense = "PWWS"
attack = "BAE"

analyze_tool(model_name = model, dataset_name=dataset, defense_name=defense, attack_name=attack)


# In[6]:


compare_defense_to_clean_accuracy("BERT", "YahooAnswers", "PWWS")


# In[ ]:




