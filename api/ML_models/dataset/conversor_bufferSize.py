import pandas as pd 
import numpy as np


df = pd.read_csv('backup//samples_nylonGuitar_8192_Mm.csv')


for sample in range(len(df)):
	convertedY = df.iloc[sample,-1]
	convertedX = df.iloc[sample,:1024]
	convertedX = np.append(convertedX, convertedY)
	try:
		newDf = np.vstack([newDf,convertedX])
	except:
		newDf = np.array(convertedX,dtype=float)
	
newPDdf = pd.DataFrame(newDf)

newPDdf.to_csv('backup//samples_nylonGuitar_1024_Mm.csv',sep=',', header=False, index=False)




print ('done')