import pandas as pd 
import numpy as np

#inputDf = raw_input('First dataset:')
#inputDf2 = raw_input('Second dataset:')
#
#nameNewDataset = raw_input('Name of the new dataset:')

#df = pd.read_csv('%s' % inputDf)

#df2 = pd.read_csv('%s' % inputDf2)

df = pd.read_csv('samples_nylonGuitar_1024_Mm-R02.csv')

df2 = pd.read_csv('samples_steelGuitar_1024-Mm-R01.csv')

#newDataset = ("%s" % nameNewDataset)

newDataset = ('samples_guitar_1024_Mm_R01.csv')

f = open(newDataset, "w")
f.truncate()
f.close()
#newDf = np.array(df.iloc[:,:],dtype=float)

for i in range(len(df)):
	newDfFloat = np.array(df.iloc[i,:-1], dtype = np.float)
	newDfFloat = np.append(newDfFloat[1:], df.iloc[i,-1])
	try:
		newDf = np.vstack([newDf,newDfFloat])
	except:
		newDf = np.array(newDfFloat,dtype=float)

for i in range(len(df2)):
	newDfFloat = np.array(df2.iloc[i,:-1], dtype = np.float)
	newDfFloat = np.append(newDfFloat[1:], df2.iloc[i,-1])
	newDf = np.vstack([newDf,newDfFloat])


newPDdf = pd.DataFrame(newDf)

newPDdf.to_csv("%s" % newDataset,sep=',', header=False, index=False)


print ("done")

	