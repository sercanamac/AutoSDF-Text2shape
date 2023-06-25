from datasets.base_dataset import BaseDataset
import pandas as pd

class Text2ShapePP(BaseDataset):
    def initialize(self):
        self.text2shapepp = pd.read_csv('../raw_dataset/text2phrase.csv')
        import pdb;pdb.set_trace()
      
    
    
    
    def __getitem__(self, index):
        return 0
        

    def __len__(self):
        return 0
        
        
    def name(self):
        return 'Text2ShapePPSample'