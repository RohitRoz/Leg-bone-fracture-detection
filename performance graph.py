import numpy as np
import matplotlib.pyplot as plt
 
  
# creating the dataset
data = {'Accuracy ':91.6, 'Sensitivity':88.2, 'Specificity ':77.7}
courses = list(data.keys())
values = list(data.values())
  
fig = plt.figure(figsize = (10, 5))
 
# creating the bar plot
plt.bar(courses, values, color ='maroon',
        width = 0.4)
 
plt.title("Performance Evaluation")
plt.show()