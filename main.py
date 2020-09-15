from identification import *
import training
import os

weight_sample = """rpmc,0.999341229
rpmj,0.931126682
rpmh,0.65297138
rpme,0.453734286
rpmg,0.451085325
rplx,0.414614517
rpmd,0.337521184
rpsp,0.328949972
hupa,0.328484009
hupb,0.327136496"""



# sample_path = "data/sample.txt"
sample_path= "data/Escherichia coli DH5 Alpha.txt" #sample we want to analyse



my_gene = pd.read_csv("weight.csv", index_col=0)["mean"]
model_data = training.gene_to_model(my_gene) 
sample = IdentifySpectra(sample_path)
sample.answer(model_data, the_threshold)
