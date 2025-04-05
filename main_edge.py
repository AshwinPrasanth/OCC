import time
from task_manager.scheduler import TaskScheduler
from task_manager.edge_executor import EdgeExecutor
from cloud_manager.resource_allocator import ResourceAllocator

def simulate_real_time_task_input(scheduler):
    import random
    task_list = ["feature_extraction", "model_inference", "logging", "monitoring", "system_update", "preprocessing"]
    for _ in range(10):
        task_name = random.choice(task_list)
        scheduler.schedule_task(task_name, timestamp=time.time())
        time.sleep(0.5)

def main():
    scheduler = TaskScheduler()
    edge_executor = EdgeExecutor()
    allocator = ResourceAllocator(edge_executor)

    simulate_real_time_task_input(scheduler)

    print("\n🔄 Starting Task Execution...\n")
    while scheduler.has_tasks():
        task_name = scheduler.get_next_task()
        allocator.route_task(task_name)

if __name__ == "__main__":
    main()
from google.cloud import aiplatform

def offload_to_cloud(task_name, input_data):
    aiplatform.init(project="************", location="us-central1")
    endpoint = aiplatform.Endpoint("*********************************************")
    prediction = endpoint.predict(instances=[input_data])
    print(f"[CLOUD] {task_name} cloud result: {prediction}")

class Task:
    def __init__(self, name, priority, deadline_sec, is_critical=True, data=None):
        self.name = name
        self.priority = priority  # 1 = High, 2 = Medium, 3 = Low
        self.deadline = time.time() + deadline_sec
        self.is_critical = is_critical
        self.data = data if data else {"features": [random.random() for _ in range(128)]}

    def __lt__(self, other):
        return self.deadline < other.deadline

# Task Manager with EDF scheduling
class TaskManager:
    def __init__(self):
        self.task_queue = []
        self.edge_load = 0.0  # 0.0 to 1.0
        self.max_edge_capacity = 0.75  # Above this, consider overload
        self.running = True

    def monitor_resources(self):
        # Simulate periodic edge load updates
        while self.running:
            self.edge_load = random.uniform(0.2, 0.95)
            print(f"[Monitor] Current edge load: {self.edge_load:.2f}")
            time.sleep(5)

    def add_task(self, task):
        with task_queue_lock:
            heapq.heappush(self.task_queue, task)
            print(f"[Scheduler] Task '{task.name}' added with deadline in {int(task.deadline - time.time())}s")

    def execute_task(self, task):
        if self.edge_load > self.max_edge_capacity and not task.is_critical:
            print(f"[Offload] Offloading task '{task.name}' to cloud due to high edge load")
            self.offload_to_cloud(task)
        else:
            print(f"[Edge] Executing task '{task.name}' on edge device")
            time.sleep(1)  # Simulate local execution

    def offload_to_cloud(self, task):
        try:
            prediction = endpoint.predict(instances=[task.data])
            print(f"[Cloud] Cloud inference for '{task.name}' result: {prediction}")
        except Exception as e:
            print(f"[Error] Failed to offload task '{task.name}' to cloud: {e}")

    def schedule_tasks(self):
        while self.running:
            with task_queue_lock:
                if self.task_queue:
                    next_task = heapq.heappop(self.task_queue)
                    now = time.time()
                    if now >= next_task.deadline:
                        print(f"[Warning] Task '{next_task.name}' missed deadline!")
                    else:
                        self.execute_task(next_task)
            time.sleep(0.5)

    def stop(self):
        self.running = False

# ---------- Simulation Starts Here ----------

if __name__ == "__main__":
    manager = TaskManager()

    # Start edge resource monitor
    monitor_thread = Thread(target=manager.monitor_resources)
    monitor_thread.start()

    # Start EDF scheduler
    scheduler_thread = Thread(target=manager.schedule_tasks)
    scheduler_thread.start()

    # Simulate task submissions
    task_types = [
        ("Feature Extraction", 1, 10),
        ("Heatmap Processing", 1, 12),
        ("Logging", 3, 25),
        ("Monitoring", 2, 20),
        ("Model Inference", 1, 8),
        ("System Maintenance", 3, 30)
    ]

    for name, priority, deadline in task_types:
        manager.add_task(Task(name=name, priority=priority, deadline_sec=deadline, is_critical=(priority == 1)))

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n[Shutdown] Stopping scheduler and monitor...")
        manager.stop()
        monitor_thread.join()
        scheduler_thread.join()
