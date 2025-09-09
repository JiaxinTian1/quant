from abc import ABC, abstractmethod

class BaseSaver(ABC):
    def __init__(self, quant_service):
        self.quant_service = quant_service
        self.model_path = quant_service.quant_params.model_path
        self.save_path = quant_service.quant_params.save_path
        self.warpped_model = quant_service.warpped_model
    
    @abstractmethod
    def save(self):
        pass