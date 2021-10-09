from utils.model import Perceptron
from utils.all_utils import prepare_data, save_model,save_plot 
import numpy as np
import pandas as pd

def main(data, eta, epochs, modelname, plotFilename):
    
    df=pd.DataFrame(data)
    print(df)
    X,y=prepare_data(df)
    
    model=Perceptron(eta=eta, epochs=epochs)
    model.fit(X,y)
    _= model.total_loss() # _ is a dummy variable
    save_model(model,filename=modelname)
    save_plot(df, plotFilename, model)

if __name__=='__main__':
    OR={
        "x1":[0,0,1,1],
        "x2":[0,1,0,1],
        "y":[0,1,1,1],
    }
    ETA=0.3 # Vary betweeen 0 and 1
    EPOCHS= 10
    main(data=OR, eta=ETA, epochs=EPOCHS, modelname="or.model", plotFilename="or.png")