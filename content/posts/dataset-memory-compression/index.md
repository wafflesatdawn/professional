---
title: "Memory compression for large datasets"
date: 2023-05-03T17:57:07-07:00
draft: false
---
Pandas gains great speed from loading everything into RAM but it comes with the obvious constraint of how much has been installed. Over on Kaggle, a user who faces such a constraint has [claimed to reduce his memory consumption by 70%](https://www.kaggle.com/code/nickycan/compress-70-of-dataset/notebook) using simple datatype conversion based on the largest and smallest numbers in each of the columns.

When reading data in pandas, several memory management options exist at the outset that offer tradeoffs such as accuracy and speed:
- `memory_map`: maps the file on disk rather than loading it into memory, reducing memory usage but increasing I/O time, especially if storage is not an SSD or NVME
- `low_memory` and `chunksize`: parses the file in chunks rather than all at once, reducing memory usage but potentially affecting performance. Additionally, the use of a small chunksize may also cause issues with data consistency or integrity, especially if the data contains inter-row dependencies.
- `dtype`: specifies the data types of columns in the resulting DataFrame, allowing for more efficient memory usage, as mentioned above

Commonsense adjustments, such as only reading required columns, can also be made, but the opposite end of the spectrum--distributed computing that uses arbitrary numbers of machines to execute code--also exists. 

An interesting conclusion then, is that a data source's size doesn't necessarily have a 1:1 relation with total memory usage, meaning a 2gb file may actually take more or less than 2gb depending on the datatypes assigned to its columns. Furthermore, the method used to measure the memory usage can also report drastically different numbers, [as described here](https://pythonspeed.com/articles/pandas-dataframe-series-memory-usage/). Of particular note are strings, which can vary greatly (a string could contain a haiku or the full text of a holy book), and can be counted as pointers or the objects themselves.
