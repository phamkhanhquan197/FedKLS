from flwr.server.strategy import FedAdam, FedAvg, FedAvgM, FedMedian, FedOpt, FedProx

from mak.strategies.fedlaw_strategy import FedLaw
from mak.strategies.fednova_strategy import FedNovaStrategy as FedNova
from mak.strategies.power_d import PowD
from mak.strategies.scaffold_strategy import ScaffoldStrategy as Scaffold
from mak.strategies.fedklsvd_strategy import FedKLSVDStrategy as FedKLSVD
from mak.strategies.fedawa_strategy import FedAWAStrategy as FedAWA