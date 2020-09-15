#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  9 15:22:13 2017

@author: dippercheng
"""
import itertools
import random
import time
import os
import numpy as np
import pandas as pd
from scipy.stats import mode
import matplotlib.pyplot as plt
from adjustText import adjust_text
import pprint

# with open('spectra_name.txt') as f:
#     ms_spectra = tuple([i.strip() for i in f])
# with open('uniprot_text.txt') as f:
#     uniprot = tuple([i.strip() for i in f])

the_threshold = 0.12  #used to filter intensities
score_threshold = 0.88
tolerance = 1000
thr = 0.05

class BacteriaSpectra:
    def __init__(self, spectra_id):
        self.id = spectra_id #load spectra_id, which is also the string of sample path
        if type(spectra_id) == str:
            self.pattern = spectra_id
        else:
            self.pattern = 'spectra/'+ ms_spectra[spectra_id]
            self.taxonomy = self.pattern.split('.')[0]
            self.genus = self.pattern.split()[0].lower()
            self.species = self.pattern.split()[1].lower()

    def plot(self):
        pattern_file = self.get_filtered_pattern(0)
        fig, ax = plt.subplots()
        if type(self.id) != str:
            
            pat_matched_file = self.get_matched_peak(0.05).groupby('mw_raw').min()
            pat_matched_file = pat_matched_file[pat_matched_file.mw < 500]
            pat_index = pat_matched_file.index
            pattern_file['color'] = pattern_file['mz'].apply(
                lambda x:[0.6, 0.6, 0.6, 1] if x in pat_index else [0, 0, 0, 1])

            bars = ax.bar(pattern_file['mz'], pattern_file['int'], width=14, facecolor='k')
        
            texts=[]
            for j,rect in enumerate(bars):
                each_mz = pattern_file.iloc[j]['mz']
                if each_mz in pat_index:
                    each_text = pat_matched_file.loc[each_mz]['gn']
                    left = rect.get_x()+1
                    top = rect.get_y()+rect.get_height()+0.01
                    texts.append(ax.text(left,top,each_text))
                    rect.set_facecolor('#930000')

            adjust_text(texts, add_objects=bars,
                 autoalign='y', only_move={'points':'y', 'text':'y', 'objects':'y'},
                  force_text=0.9, )
        
        bars = ax.bar(pattern_file['mz'], pattern_file['int'], width=14, facecolor='k')

        ax.set_ylabel('Intensity')
        ax.set_xlabel('m/z')
        plt.xlim(3000,12000)
        ax.plot([3000,12000],[0,0],color='black')
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        if type(self.id) != str:
            ax.set_title(self.genus.capitalize() +' ' + self.species)
        return ax

    def get_filtered_pattern(self, int_threshold=the_threshold):
        # loading .csv file containing MS data (intensity to molecular weight) into a pd data frame
        peaks = open(self.pattern, "r") #open to read the peak of sample data
        standard_mass, standard_int = ([], [])
        for peak in peaks: #for each line 
            standard_mass.append(float(peak.split()[0])) #split into standard mass and intensity
            standard_int.append(float(peak.split()[1]))
        peaks.close()
        pattern = pd.DataFrame({'mz': standard_mass, 'int': standard_int})

        # Performing normalisation to fit intensities to range from 0 to 1
        if pattern.int.max() > 100:
            pattern['int'] = pattern.int / pattern.int.max()
        elif pattern.int.max() > 1:
            pattern['int'] = pattern.int / 100

        # Returning only entries where normalised intensity is above specified threshold
        return pattern[pattern.int > int_threshold]

    def get_uniprot_table(self):
        my_table = pd.read_csv('uniprot/' + self.genus + '.csv')[['species', 'gn', 'mw']]
        my_table = my_table[my_table.species == self.species]
        return my_table

    def get_matched_peak(self,int_threshold):
        def match_milk_peak(x):
            raw = self.get_filtered_pattern(int_threshold)
            y = abs(raw['mz'] - x)
            y_delta = y[y < x * tolerance * 1e-6]
            if len(y_delta) == 0:
                return np.nan, np.nan, np.nan
            return round(y_delta.iloc[0] * 1e6 / x, 2), raw.loc[y_delta.index[0]]['mz'], raw.loc[y_delta.index[0]]['int']

        my_table_matched = self.get_uniprot_table()
        ms_interim = my_table_matched['mw'].map(match_milk_peak)
        my_table_matched['mw'] = ms_interim.apply(lambda x:x[0])
        my_table_matched['mw_raw'] = ms_interim.apply(lambda x:x[1])
        my_table_matched['int'] = ms_interim.apply(lambda x:x[2])
        my_table_matched['genus'] = self.genus
        my_table_matched = my_table_matched.dropna()
        my_table_matched.to_csv('matched_result/'+str(self.id)+'.csv')
        return my_table_matched


class IdentifySpectra(BacteriaSpectra):
    def answer(self, model_data, threshold=the_threshold):
        t1 = time.time()
        threshold = the_threshold
        com_table, answer_table, gn_set, gn_value = model_data #we are using these variables to identify the unknown spectra. model data is obtained from training
        pattern = self.get_filtered_pattern(threshold) #filter the sample's peak into its mz and intensity and normalise intensity

        def match_soy_peak(x):
            """
            Attempts to identify presence of gene from MS data.
            :param x: input gene weight
            :return: (float) measure of "closeness" to theoretical weight (???)
            """
            y = abs(pattern['mz'] - x) # y = subtract x from each value in the list pattern
            y = y[y < x * tolerance * 1e-6] #only keep values that are less than the tolerance
            if len(y) == 0:
                return 0.0

            return round(np.exp(-y.iloc[0]*1e3/x), 2)


        def making_faster(my_table, my_gene_set):
            """

            :param my_table: matrix of 10 genes and respective weights for each (genus, species) pair
            :param my_gene_set: list of 10 most informative genes
            :return: List of unique gene weights for all (genus, species, gene) triplets
            """
            my_table = my_table.fillna(0) #pandas dataframe. replace all NaN's with 0.0
            b = set({})
            for i in my_gene_set: #for each gene
                b = b | set(my_table[i].unique())  # union of set. unique gene weights of each gene
            return list(b)

        b = making_faster(com_table, gn_set)   # a list of unique gene weights, apparently (???)

        c = [match_soy_peak(x) for x in b] # c is post-processed unique gene weights, if the gene weights hit threshold

        my_mw_dict = dict(zip(b, c)) #grouping each soypeak(value) to its unique gene weight(key) as a dictionary. eg. "30483": "0.0"

        table = com_table.applymap(lambda x: my_mw_dict[x]) #for each gene, put the soypeak(value) to its corresponding genus,species pair. 

        id_value = (table.dot(gn_value)).sort_values()  # list of "similarity" measure to gene index. Does dot product between table (1276 x 10) of each genus,species's gene weights on actual avg gene weight (10 x 1 ) to get a list of "matching" intensities for each genus,value pair (1276 x 1)
        id_panel = id_value[-4:]  # getting the last 4 (genes with highest similarity measures)

        score_mode = id_panel.mode() #get the value that appears most often
        if not score_mode.empty :
            score = score_mode.iloc[-1] #get the last value
        else:
            score = id_panel.max()

        if score < score_threshold:
            return 'none'

        candidates = id_panel[id_panel == score].index[0] #
        species_index = id_panel[id_panel == score].index #returns the indices that contains the score
        species_candidates = tuple(answer_table.iloc[species_index]['species']) #returns the species that match the score
        answer_candidate = answer_table.iloc[candidates]['genus'] #returns the genus that matches the score
        print("answer candidate:", answer_candidate) #
        t2 = time.time()

        answer_dict = {'score (the most commonly appeared dot product)': score, 'time taken': round(t2-t1, 2),
                       "genera": answer_candidate, "species" : species_candidates,
                       "path": self.pattern}

        pprint.pprint(answer_dict)
        print(answer_dict)
        return answer_dict['genera']


'''
sample_path = 'lab/Bacillus subtilis ATCC 6633.txt'
IdentifySpectra(sample_path).answer(model_data, the_threshold)


with open('spectra_name.txt') as f:
    w = [i for i in f.read().split('\n')]
with open('database_genera.txt') as f:
    genera_list = [i.lower() for i in f.read().split('\n')]

spectra_info = pd.DataFrame(w, columns = ['spectra'])
spectra_info['species'] = spectra_info['spectra'].apply(lambda x: x.split()[1])
spectra_info['genus'] = spectra_info['spectra'].apply(lambda x: x.split()[0].lower())
spectra_info = spectra_info[spectra_info['spectra'].map(lambda x: True if x in name0 else False)]


def if_species_func(x):
    genus = x.split()[0].lower()
    species = x.split()[1]
    if genus not in genera_list:
        return 0
    idid = ms_spectra.index(x)
    table = BacteriaSpectra(idid).get_uniprot_table()
    if len(table) == 0:
        return 1
    else:
        return 2


spectra_info['if_database'] = spectra_info['spectra'].map(if_species_func)

data = spectra_info[spectra_info.if_database == 2]
data ['matched'] = data.index.map(lambda x: len(pd.read_csv('matched_result/'+str(x)+'.csv').mw_raw.unique()))

#spectra_info ['matched'] = spectra_info.index.map(lambda x: len(pd.read_csv('matched_result/'+str(x)+'.csv').mw_raw.unique()))

spectra_info['matched'] = spectra_info.index.map(sm)
def sm(x):
    try:
        return data.iloc[x]['matched']
    except:
        return 0

matched ['uni'] = matched.index.map(lambda x: len(pd.read_csv('uniprot/'+str(x)+'.csv')))

#data ['matched'] = data.index.map(lambda x:len(BacteriaSpectra(x).get_matched_peak(0.05)))


the_data['answer'] = the_data.index.map(lambda x: IdentifySpectra(x).answer(model_data))

new_data['answer'] = new_data.index.map(lambda x: IdentifySpectra(x).answer(model_data))

sdata = data[data.matched>10]
sdata_list = tuple(sdata.index)

model = gn_training(sdata_list, 0.05)
reslut = validation(sdata_list, model, 0.05)

model = gn_training(random.sample(sdata_list, 30), 0.05)

reslut = validation(abc, model, 0.05)
reslut = validation(sdata_list, model, 0.05)
reslut = validation(random.sample(sdata_list, 30), model, 0.05)

def itertools_training():
'''
# analysis 
'''
data = pd.read_csv('thedata.csv', index_col = 0)
data.loc[data.score < 1.103, 'answer'] = 'none'
def mapping(each_panda):
    da = len(each_panda[ each_panda.answer == each_panda.genus ])
    db = len(each_panda[ each_panda.answer != each_panda.genus ])
    dc = len(each_panda[ each_panda.answer == 'none' ])
    return da, db-dc, dc

data.groupby('genus').apply(mapping)

'''

'''
num = pd.read_csv('num_uni.csv', header=None, index_col = 0)
gm = pd.read_csv('g_s.csv')
gm['org'] = gm.genus+'_'+gm.species
def try_num(x):
    try:
        return int(num.loc[x])
    except:
        return 0
gm['num'] = gm['org'].map( try_num )
'''

