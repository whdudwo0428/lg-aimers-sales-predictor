import torch.nn as nn
from abc import ABC, abstractmethod

class BaseModel(nn.Module, ABC):
    """
    모든 모델 아키텍처가 상속받을 기본 추상 클래스입니다.
    이 클래스를 상속받는 모든 모델은 forward 메서드를 반드시 구현해야 합니다.
    """
    def __init__(self):
        super(BaseModel, self).__init__()

    @abstractmethod
    def forward(self, *args, **kwargs):
        """
        모델의 순전파 로직을 정의합니다.
        *args와 **kwargs를 사용하여 다양한 모델의 입력 형식을 유연하게 받을 수 있습니다.
        """
        raise NotImplementedError