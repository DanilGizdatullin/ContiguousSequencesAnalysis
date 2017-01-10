# Contiguous Sequences Analysis
Some algorithms to work with gapless sequences  
  
## Format of Sequence Data.
To use this library, you need to have data is special format.  
First row is a name of event separated by **;**. The others rows are timestamp(some int value) of current event for  
current user. The format of timestamp is doesn't matter. If event didn't happen you need to skip this event.
  
**Example**  
*label;work;separation;marriage*  
*1;;215;305*  
  
## Create a Special Data Format From CSV  
To build Sequences Tree and using Classifier you need to convert your data in special format.  
**Example**  
*data = [[['1'], ['2'], ['3']], ..., [['1', '2'], ['3'], ['4']]]*  
*labels = [1, ..., 0]*  
You can find some examples of using in *./examples/export_data_from_csv.py*    
  
## Building Rules Trie.  
You can build a simple Rules Trie and Closured Rules Trie, which has only closured patterns.  
Also you can find important rules or hypothesis by minimum growth rate.  
You can find some examples in *./examples/build_rules_trie_on_some_data.py*  
  
## Classifier.  
You can use Simple Classifier, Closure Classifier or Hypothesis Classifier. With this tool you can solve binary and  
multiclassification tasks.  
You can find some examples in *./examples/using_classification.py*  
