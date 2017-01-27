# Contiguous Sequences Analysis
Here one can find several pattern mining algorithms to work with gapless sequences. 
  
## Format of Sequence Data.
To use this library, you need to have data in a special format.  
First row is the name of an event followed by **;** as a separator. The next row contains timestamps (int value) of the events for  
the current user. The format of a timestamp doesn't matter. If event didn't happen, you need to skip this event.
  
**Example**  
*label;work;separation;marriage*  
*1;;215;305*  
  
## Custom Data Format From CSV  
To build Sequence Tree and use our Classifier you need to convert your data in a special format.  
**Example**  
*data = [[['1'], ['2'], ['3']], ..., [['1', '2'], ['3'], ['4']]]*  
*labels = [1, ..., 0]*  
You can find some usage examples in *./examples/export_data_from_csv.py*    
  
## Building Rules Trie.  
One can build a simple Rules Trie and Closed Rules Trie, which contaons only closed patterns.  
Also you can find important rules or hypothesis by setting the value of minimum growth rate.  
You can find some examples in *./examples/build_rules_trie_on_some_data.py*  
  
## Classifier.  
You can use Simple Classifier, Closure Classifier or Hypothesis Classifier. With this tool you can solve binary and  
multi-classification tasks.  
You can find some examples in *./examples/using_classification.py*  
