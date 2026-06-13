import os
import paddle

USE_TRITON_FUSION = paddle.utils.strtobool(os.getenv("USE_TRITON_FUSION", "0"))

from .bias_relu import FusedBiasReluTriton
from .depthwise_conv import FusedDepthwiseConvK3P1Triton
from .l1_loss import FusedL1LossTriton
from .layernorm import FusedLayerNormTriton
from .simple_gate import FusedSimpleGateTriton
from .simple_gate_avg_pool import FusedSimpleGateAvgPoolTriton
from .weighted_residual_add import FusedWeightedResidualAddTriton
