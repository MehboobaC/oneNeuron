from utils.model import Perceptron
from utils.all_utils import prepare_data, save_model,save_plot 
import pandas as pd

def main(data, eta, epochs, filename, plotFilename):
    
    df=pd.DataFrame(data)
    print(df)
    X,y=prepare_data(df)
    
    model=Perceptron(eta=ETA, epochs=EPOCHS)
    model.fit(X,y)
    _= model.total_loss()
    save_model(model,filename=filename)
    save_plot(df, plotFilename, model)

if __name__=='__main()__':
    XOR={
        "x1":[0,0,1,1],
        "x2":[0,1,0,1],
        "y":[0,1,1,0],
    }
    ETA=0.3 # Vary betweeen 0 and 1
    EPOCHS= 10
    main(data=XOR, eta=ETA, epochs=EPOCHS, filename="xor.model", plotFilename="xor.png")