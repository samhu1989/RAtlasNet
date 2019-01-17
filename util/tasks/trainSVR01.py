#
from .task import Task;
from .datasets import ShapeNet;
class RealTask(Task):
    def __init__(self):
        super(RealTask,self).__init__();