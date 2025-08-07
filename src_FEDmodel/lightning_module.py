import pytorch_lightning as pl
import torch

class LitModel(pl.LightningModule):
    def __init__(self, model, config):
        super().__init__()
        self.model = model
        self.config = config
        self.loss_fn = config.LOSS_FN
        # save_hyperparameters()를 사용하면 체크포인트에서 모델을 로드하기 쉬움
        self.save_hyperparameters(ignore=['model'])

        # 테스트 결과를 저장할 리스트
        self.test_predictions = []
        self.test_actuals = []

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def _common_step(self, batch, batch_idx):
        # DataModule에서 받은 배치를 그대로 모델에 전달합니다.
        batch_x, batch_y, batch_x_mark, batch_y_mark = batch
        
        # 모델의 출력 형식에 따라 코드를 수정합니다. 
        # FEDformer는 (예측값, 어텐션) 튜플을 반환합니다.
        outputs, _ = self(batch_x, batch_x_mark, batch_y, batch_y_mark)
        
        pred = outputs[:, -self.config.HORIZON:, :]
        true = batch_y[:, -self.config.HORIZON:, :]
        
        loss = self.loss_fn(pred, true)
        return loss, pred, true

    def training_step(self, batch, batch_idx):
        loss, _, _ = self._common_step(batch, batch_idx)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, _, _ = self._common_step(batch, batch_idx)
        self.log("val_loss", loss, on_epoch=True, prog_bar=True, logger=True)

    def test_step(self, batch, batch_idx):
        loss, pred, true = self._common_step(batch, batch_idx)
        self.log("test_loss", loss, on_epoch=True, logger=True)

        self.test_predictions.append(pred)
        self.test_actuals.append(true)

    def on_test_epoch_end(self):
        # list에 저장된 텐서들을 하나로 합침
        self.test_predictions = torch.cat(self.test_predictions, dim=0)
        self.test_actuals = torch.cat(self.test_actuals, dim=0)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.config.LEARNING_RATE)