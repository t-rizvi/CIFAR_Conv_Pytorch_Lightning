import pytorch_lightning as pl
from IPython.display import clear_output 

from lightning_models import TwoLayerNet
from data_class import CIFARDataModule

hparams = {
    "batch_size": 32,
    "learning_rate": 1e-3,
}


model = TwoLayerNet(hparams)

data=CIFARDataModule(hparams["batch_size"])
data.prepare_data()

trainer = pl.Trainer(
    weights_summary=None,
    max_epochs=25,
    progress_bar_refresh_rate=25, # to prevent notebook crashes in Google Colab environments
    #Uncomment to use GPU if available
    #gpus=1 
)

trainer.fit(model,train_dataloader=data.train_dataloader(),val_dataloaders=data.val_dataloader())


