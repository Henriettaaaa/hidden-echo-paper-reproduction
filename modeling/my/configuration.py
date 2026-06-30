from typing import Literal
from transformers.models.qwen2.configuration_qwen2 import Qwen2Config


class AdditionalConfig:
    def __init__(
        self,
        # DP 噪声相关
        privacy_budget: float = 100.0,
        clip_embedding_l2: bool = True,
        noise_type: Literal["Chi", "Gaussian"] = "Chi",
        # HiddenEcho 主体
        lst_reduce_factor: int = 8,
        lst_skip: list[int] = [-1],
        lst_temperature: float = 0.1,
        lst_input_type: Literal["clean", "noisy"] = "clean",
        lst_enable: bool = True,
        lst_random_init: bool = False,
        # HiddenEcho+ 的层选择
        auto_skip: bool = False,
        num_reserved_layers: int = 1,
        num_integrate_step: int = 5,
        num_samples: int = 32,
        keep_last_layer: bool = False,
        num_integrate_batch_size: int = 4,
        # mi_* 是信息瓶颈约束
        mi_downsample_enable: bool = True,
        mi_estimator_iter_num: int = 2,
        mi_estimator_lr: float = 1e-4,
        mi_xz_ratio: float = 0.001,
        mi_yz_ratio: float = 0.001,
        mi_estimator_hidden_dim: int = 128,
        
        use_residual: bool = True,
        **kwargs,
    ):
        self.privacy_budget = privacy_budget
        self.clip_embedding_l2 = (
            clip_embedding_l2  
        )
        self.noise_type = noise_type

        
        self.lst_enable = lst_enable  
        self.lst_reduce_factor = lst_reduce_factor
        self.lst_skip = lst_skip
        self.lst_temperature = lst_temperature
        self.lst_input_type = lst_input_type
        self.lst_random_init = lst_random_init  
        
        
        self.auto_skip = auto_skip 
        self.num_reserved_layers = num_reserved_layers 
        self.num_integrate_step = num_integrate_step
        self.num_samples = num_samples
        self.keep_last_layer = keep_last_layer 
        self.num_integrate_batch_size = num_integrate_batch_size
        
        
        self.mi_downsample_enable = mi_downsample_enable
        self.mi_estimator_iter_num = mi_estimator_iter_num
        self.mi_estimator_lr = mi_estimator_lr
        self.mi_xz_ratio = mi_xz_ratio
        self.mi_yz_ratio = mi_yz_ratio
        self.mi_estimator_hidden_dim = mi_estimator_hidden_dim

        self.use_residual = use_residual  


        
        self.model_cls_module: str | None = kwargs.get("model_cls_module", None)
        self.model_cls_name: str | None = kwargs.get("model_cls_name", None)

# 把额外参数挂到 HF 的 Qwen2Config 上
class MyQwen2Config(Qwen2Config, AdditionalConfig):
    def __init__(
        self,
        **kwargs,
    ):
        Qwen2Config.__init__(self, **kwargs)
        AdditionalConfig.__init__(self, **kwargs)
