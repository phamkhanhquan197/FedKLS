from flwr.server.strategy import FedAdam, FedAvg, FedAvgM, FedMedian, FedOpt, FedProx

from mak.strategies.fedlaw_strategy import FedLaw
from mak.strategies.fednova_strategy import FedNovaStrategy as FedNova
from mak.strategies.power_d import PowD
from mak.strategies.scaffold_strategy import ScaffoldStrategy as Scaffold
from mak.strategies.fedklsvd_strategy import FedKLSVDStrategy as FedKLSVD
from mak.strategies.fedawa_strategy import FedAWAStrategy as FedAWA
from mak.strategies.ffa_lora_strategy import FFALoRAStrategy as FFALoRA
from mak.strategies.flex_lora_strategy import FlexLoRAStrategy as FlexLoRA
from mak.strategies.pfedmoap_strategy import PFedMoAPStrategy as PFedMoAP
