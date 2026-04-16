from clearml import Task

# 1. Retrieve the task by its ID
task = Task.get_task(task_id='b32245a6df3b45f6abf674d7c589340a')

# 2. Access the artifact and download it to a local cache
# 'artifact_name' is the name you gave it during upload
#local_path = task.artifacts['generator_final'].get_local_copy()

#print(f"Artifact downloaded to: {local_path}")

print(task)