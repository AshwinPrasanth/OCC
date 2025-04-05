# Priority Levels: High > Medium > Low

TASK_PROFILES = {
    "feature_extraction": {"priority": "high", "compute_cost": 8},
    "model_inference": {"priority": "high", "compute_cost": 9},
    "preprocessing": {"priority": "medium", "compute_cost": 5},
    "logging": {"priority": "low", "compute_cost": 2},
    "monitoring": {"priority": "low", "compute_cost": 1},
    "system_update": {"priority": "medium", "compute_cost": 3}
}

PRIORITY_ORDER = {
    "high": 3,
    "medium": 2,
    "low": 1
}
from .task_definitions import TASK_PROFILES, PRIORITY_ORDER
import heapq

class TaskScheduler:
    def __init__(self):
        self.task_queue = []

    def schedule_task(self, task_name, timestamp):
        task = TASK_PROFILES.get(task_name)
        if task:
            priority_score = PRIORITY_ORDER[task['priority']]
            heapq.heappush(self.task_queue, (-priority_score, timestamp, task_name))

    def get_next_task(self):
        if self.task_queue:
            return heapq.heappop(self.task_queue)[2]
        return None

    def has_tasks(self):
        return len(self.task_queue) > 0
import time

class EdgeExecutor:
    def __init__(self):
        self.edge_cpu_capacity = 10  # Example available compute units

    def can_execute(self, task_cost):
        return task_cost <= self.edge_cpu_capacity

    def execute(self, task_name):
        print(f"[EDGE] Executing task: {task_name}")
        time.sleep(1)
def offload_to_cloud(task_name):
    print(f"[CLOUD] Offloading task '{task_name}' to Vertex AI...")
    # You can later integrate with Google Cloud SDK or REST API
from task_manager.task_definitions import TASK_PROFILES
from cloud_manager.vertex_ai_interface import offload_to_cloud

class ResourceAllocator:
    def __init__(self, edge_executor):
        self.edge_executor = edge_executor

    def route_task(self, task_name):
        task_info = TASK_PROFILES[task_name]
        cost = task_info['compute_cost']

        if self.edge_executor.can_execute(cost):
            self.edge_executor.execute(task_name)
        else:
            offload_to_cloud(task_name)
