Given an input KPI vector (2-1), we want to know the current network condition. In other words, 
the task is to diagnose the occurrence of ERP, ED and EU. As multiple faults may occur 
simultaneously, e.g., ERP+ED and ERP+EU, the diagnosis problem is treated as a multi-label classification problem2. In particular, we use 1 and 0 to indicate the occurrence and absence of 
a fault, respectively. Let NF   be the number of possible faults considered in a network. Then, 
y = [y1
 ,y2
 ,…,yNF
 ]  {0,1}NF denotes the label vector corresponding to an input KPI instance x, 
where yi = 1 if fault i occurs, and yi = 0 otherwise. The label for normal condition is thus 
y = 0NF
 . We remark that such labeling method, which adopts binary variables for classes to 
indicate whether an instance is associated with the classes, is known as binary relevance in 
literature [36]. We let NK   be the number of input KPIs, i.e., the length of (2-1). The multi
fault diagnosis problem is to find a function f :  NK → {0,1} NF such that f(x) = y, where y is 
the ground truth label of the faults. 
