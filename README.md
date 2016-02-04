# UnbrokenSequenceAnalysis
Some algorithms to work with unbroken sequences  
  
## Format of Sequence Data.
To use this library, you need to have data is special format.  
First row is a name of event separated by **;**. The others rows are timestamp(some int value) of current event for  
current user. The format of timestamp is doesn't matter. If event didn't happen you need to skip this event.
  
**Example**  
*label;work;separation;marriage*  
*1;;215;305*  
  
You can find some examples of using in *./examples/export_data_from_csv.py*  
  
  
## Building Rules Trie.  
