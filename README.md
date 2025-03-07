# BacteriaMSLF
In this work, we developed a bacterial MALDI TOF spectra identification approach without using standard spectra library, but only protein sequences from *UniProt* database. Genetic algorithm with double cross-validation was performed to get the proteins encoded by the most informative genes as protein panels to be considered for bacterial identification. At genus level, the identification accuracy exceeded 80%. Although the performance is still behind to the performance of library search based approach, it can be used as a complementary method when standard library is not available. It can also be used in diverse application areas such as clinical microbiology, biodefense, food safety and environmental health.
## Requirement
Python 3.5 with `numpy`, `pandas`, and `matplotlib` 

[Anaconda] (https://anaconda.org/anaconda/python) is recommended.

## Getting Started 
A good practice will be setting a virtual environment for you to install all packages you need, including a copy of the global python interpreter. This will also help to reduce version conflicts with other projects. 

Create, then activate the virtual environment. 
```
py -3 -m venv .venv
.venv\scripts\activate
```
If there are issues with activating the virtual environment, run the following command to reset the execution policy: 
```
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process
```
before activating the virtual environment again. 

install all the packages using pip like so: 
```
pip install numpy
```
## Database
### Spectra
`BacteriaMSLF` uses peak lists for bacteria identification. A peak list is stored in a space-separated text (.txt) file including the m/z and intensity of all the peaks from a mass spectrum of a strain of bacteria with the first column as m/z and the second column as intensity, e.g.:
```
4025.85 14.9166578754281
4179.39 10.0793564012325
4195.16 100
4279.94 13.6100699949239
4296.99 13.759296917905
4339.2 16.2279091177302
...
9998.94 4.53344723468826
10103.51 25.6021474910836
10245.55 14.5594503134826
10457.14 9.18592847464415
10929.3 2.39529317439903
11115.92 10.3301379804275
11339.22 8.75138710039803
11702.38 2.53578789353994
```
- European Consortium of Microbial Resources Centres (EM-baRC): [http://www.embarc.eu/deliverables/EMbaRC_D.JRA2.1.4_D15.33_MALDI-TOF_DB.pdf]

- Robert Koch-Institute: [http://www.microbe-ms.com/]

- The Public Health Agency of Sweden: [http://spectra.folkhalsomyndigheten.se/spectra/]
### UniProt
`BacteriaMSLF` uses proteomic sequences in ***UniProt***[[http://www.uniprot.org/]
 to match the MALDI-TOF spectra, e.g.:

```
>tr|A0A087INJ0|A0A087INJ0_VIBVL UDP-N-acetylmuramoyl-L-alanyl-D-glutamate--2,6-diaminopimelate ligase OS=Vibrio vulnificus GN=murE PE=3 SV=1
MRNTMNLTNLLAPWLDCPELADITVQSLELDSRQVKQGDTFVAIVGHVVDGRQYIEKAIE
QGANAIIAQSCQQYPSGLVRYQQNVVIVYLEKLDEKLSQLAGRLYQHPEMSLIGVTGTNG
KTTITQLIAQWLELAGQKAAVMGTTGNGFLNALQPAANTTGNAVEIQKTLADLQQQGAKA
TALEVSSHGLVQGRVKALQFAAGVFTNLSRDHLDYHGTMEAYAQAKMTLFTEHQCQHAII
NLDDEVGAQWFQELKQGVGVSLYPQDASVKALWASSVAYAESGITIEFEGCFGQGRLHAP
LIGEFNATNLLLALATLLALGVDKQALLGSAANLRPVLGRMELFQVNSKAKVVVDYAHTP
DALEKALQALRVHCTGHLWAIFGCGGDRDKGKRPMMAEIAERLADHVVLTDDNPRSEDPA
MIVQDMLAGLTRADSAVVEHDRFSALQYALDNAQADDIILLAGKGHEDYQVLKHQTVHYS
DRESAQQLLGISS
```
## Usage
The following examples are running in the `IPython` environment.
Run [main.py] to load the functions.
```
%run main.py
```
### Identificaiton
Load the unknown peak lists.
```py
sample_path = 'data/sample.txt'
sample = IdentifySpectra(sample_path)
sample.get_filterd_pattern(thr)
```
`weight.csv` are obtained from traing, which is same as the the gene weights in the Figure 3.
```py
my_gene = pd.read_csv('weight.csv',index_col=0)['mean']
model_data = gene_to_model(my_gene)
```
Use `model_data` to identify this unknown spectra.
```py
sample.answer(model_data, the_threshold)
```
Output:
```py
{'genera': 'bacillus',
 'score': 1.9266267268100001,
 'species': ('toyonensis', 'anthracis'),
 'time': 2.3}
 ```
The result of identification at genus level is *Bacillus*. At species level, it can not be distinguished between *Bacillus toyonensis, Bacillus anthracis*. The right identification *Bacillus anthracis* is in the group.

### Cleaning

```py
uniprot_path = ...
clean = os.listdir('uniprot_path')
cleaning_data(clean)
```
### Training
```py
training_set = ...
training_data(training_set,double_cross=10)
```

## Details
### ga.py
modification of pyevolve.

**Usage:**
The original gene weight table is:
```
rpmc	1
rpmj	0.947787695
rpmh	0.650960175
rpme	0.236613986
rpmg	0.452056135
rplx	0.39878139
rpmd	0.38311829
rpsp	0.268586678
hupa	0.225921218
hupb	0.230108406
```
Optimization:
```py
import ga
import training

gene_weight_table = ...
parameters = ...
training_set = ...

my_gene = Gene(gene_weight_table)
my_alth = Algorithm(training.validation(), parameters)
my_gene.optimize(my_alth, 50)
print (my_gene.weight)
```
The optimized gene weight table is:
```
rpmc	0.999341229
rpmj	0.931126682
rpmh	0.65297138
rpme	0.453734286
rpmg	0.451085325
rplx	0.414614517
rpmd	0.337521184
rpsp	0.328949972
hupa	0.328484009
hupb	0.327136496
```


### identification.py
package for identification of bacteria spectra
### parse_data.py
cleaning and transforming fasta file into csv file
### plotting.py
plotting fuctions in the paper
### training.py
training functions

## Publications
Ding Cheng, Liang Qiao and Peter Horvatovich, "Toward Spectral Library-Free Matrix-Assisted Laser Desorption/Ionization Time-of-Flight Mass Spectrometry Bacterial Identification", Submitted.

## License

`BacteriaMSLF` is distributed under a BSD license. See the LICENSE file for details.

## Contacts

Please report any problems directly to the github issue tracker. Also, you can send feedback to `liang_qiao AT fudan.edu.cn`.

