# Dialogue & Narrative Coursework

## Subtask 1
The goal is to predict the knowledge grounding in form of document span for the next agent response given dialogue history and the associated document.

## How to load our datasets
We mostly used the reading_comprehension (`rc`) dataset for our work. While this dataset contains the whole context required to ground the user utterances, it doesnt divide said context into spans as it's done in the 'document_domain' dataset. We merged all this information into a single dataset. Specifically, we added the spans corresponding to the context (see column 'spans'), and the spans corresponding to the grouding (see column 'answers' then key 'spans') to the train and validation doc2dial_rc datasets. To load our datasets, use the following steps for ease of use:
```python
import pandas as pd
import ast
to_dict = lambda ex:ast.literal_eval(ex)
df = pd.read_csv('./data/doc2dial_rc_<train/val>.csv.zip', converters={'answers':to_dict, 'spans':to_dict})
```
