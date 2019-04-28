import matplotlib
matplotlib.use('Agg')
from .bot import BaseBot
from .lr_scheduler import TriangularLR, GradualWarmupScheduler
