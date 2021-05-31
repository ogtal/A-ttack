# A&ttack
Dette repository indeholder kode og modelvægtene til *A&ttack* algortimen. Den er udviklet af [Analyse og Tal F.M.B.A.](www.ogtal.dk) med støtte fra TryghedsFonden. 

Algoritmen er designet til at finde sproglige angreb i korte tesktstykker. Den er blevet brugt til at finde sproglige angreb i den offentlige debat, et arbejde som man kan læse om [her](https://strapi.ogtal.dk/uploads/966f1ebcfa9942d3aef338e9920611f4.pdf). En let tilgængelig beskrivelse af hvordan algoritmen er blevet til kan findes i denne [artikel](https://politiken.dk/indland/art8214841/S%C3%A5dan-fik-en-astrofysiker-og-hans-kollegaer-en-superalgoritme-til-at-forst%C3%A5-hvorn%C3%A5r-vi-mennesker-sviner-hinanden-til-p%C3%A5-Facebook).  


## Definition af sproglige angreb

Algortimen er en binær klassifikationsalgortime der vurdere om et kort tekststykke er udtryk for et sprogligt angreb eller ej. Den helt korte definition af et sprogligt angreb er tekststykker der indeholder:
*Stigmatiserende, nedsættende, krænkende, chikanerende og truende ytringer rettet mod personer eller grupper.*

En udvidet definition af sproglige angreb og eksempler herpå kan findes (her)[definitioner.pdf].

## Beskrivelse af algoritmen

Algortimen er trænet vha. et annoteret datasæt med 67.188 tekststykker. Teksstykkerne er kommentarer og svar afgivet på opslag i en række offentlige Facebook Pages og større grupper. Datasættet er opdelt i et træningsdatasæt (70 %), et evalueringsdatasæt (20 %) og et testdatasæt (10 %).  

Trænings- og evalueringsdatasættet blev brugt til at træne og udvælge den bedste kombination af algoritmearkitektur og hyperparametre. Til det brugte vi den højest macro average F1 score. Efter udvælgelsen af den bedste algoritme blev denne testet på testdatasættet. 

Den bedste model bruger en [dansk electra model](https://huggingface.co/Maltehb/-l-ctra-danish-electra-small-uncased#) som sprogmodel og har et feed forward lag til selve klassificeringen. Se modeldefinitionen i filen `model_def.py`. 

## Resultater

Resultaterne for algoritmen på evalueringsdatasættet er: 
 - Macro averace F1 score: 0.8341
 - Prevision: 0.8389  
 - Recall: 0.8295  
 - Confusion matrix:

|         | Annoteret ikke-angreb | Annoteret angreb  |
| ------------- |:-------------:| :-----:|
| **Ikke-angreb iflg. A&ttack** | 9663 | 712 |
| **Angreb iflg. A&ttack**      | 830  | 2218 |

Og for testdatasættet:
 - Macro averace F1 score: RONNIE RONNIE RONNIE
 - Prevision: RONNIE RONNIE RONNIE  
 - Recall: RONNIE RONNIE RONNIE  
 - Confusion matrix:

|         | Annoteret ikke-angreb | Annoteret angreb  |
| ------------- |:-------------:| :-----:|
| **Ikke-angreb iflg. A&ttack** | RONNIE RONNIE RONNIE | RONNIE RONNIE RONNIE |
| **Angreb iflg. A&ttack**      | RONNIE RONNIE RONNIE  | RONNIE RONNIE RONNIE |


## Brug af algoritmen

```python
from transformers import AutoTokenizer
from model_def import ElectraClassifier
model_checkpoint = 'Maltehb/-l-ctra-danish-electra-small-cased'
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, use_fast=True)

model = ElectraClassifier(model_checkpoint,2)
model_path = 'pytorch_model.bin'
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))

model.eval()
```

## Tak til:


**Projektets annotører**
 - Ida Marcher Manøe
 - Julie Enevoldsen
 - Nikolaj Meldgaard Christensen
 - Naja Bau Nielsen

**Projektets advisory board**
 - Andreas Birkbak, Associat Professor, TANTlab, AAU 
 - Bolette Sandford Pedersen, Professor, CenterforSprogteknologi, KU
 - Leon Derczynski, Associate Professor, ComputerScience, ITU
 - Marianne Rathje, Seniorforsker, Dansk Sprognævn
 - Michael Bang Petersen, Professor, Institut for Statskundskab, AU
 - Rasmus Rønlev, Adjunkt, Center for Journalistik, SDU

**Vores samarbejdspartnere hos TrygFonden**
 - Anders Hede, Forskningschef
 - Christoffer Elbrønd, Projektchef
 - Christian Nørr, Dokumentarist
 - Peter Pilegaard Hansen, Projektchef

**Danske Open-source teknologi vi står på skuldrene af**
 - The Danish Gigaword Project: Leon Derczynski, Manuel R. Ciosici
 - Ælectra: Malte Højmark-Bertelsen

## Kontakt

Spørgsmål til indeholder af dette repository kan sendes til:
Ronnie Taarnborg (ronnie@ogtal.dk)
Edin Lind Ikanovic (edin@ogtal.dk)
