import mlflow

MLFLOW_TRACKING_URI = "http://ec2-13-62-47-8.eu-north-1.compute.amazonaws.com:5000"
MODEL_NAME = "yt_chrome_plugin_model"

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
client = mlflow.tracking.MlflowClient()

# Get all model versions
versions = client.search_model_versions(f"name='{MODEL_NAME}'")

if not versions:
    print(" No model versions found. Skipping promotion.")
    exit(0)

# Sort by version number
versions = sorted(versions, key=lambda v: int(v.version))

latest_version = versions[-1]
latest_version_num = latest_version.version
latest_stage = latest_version.current_stage

print(f" Latest version: {latest_version_num} | Stage: {latest_stage}")

#  Case 1: Already in Production
if latest_stage == "Production":
    print("Latest model already in Production. Nothing to do.")
    exit(0)

#  Case 2: Latest is in Staging → promote
if latest_stage == "Staging":
    print(f" Promoting version {latest_version_num} to Production")

    client.transition_model_version_stage(
        name=MODEL_NAME,
        version=latest_version_num,
        stage="Production",
        archive_existing_versions=True
    )

    print(" Promotion successful. Old Production archived.")
    exit(0)

#  Case 3: Latest is neither Staging nor Production
print(f" Latest model is in '{latest_stage}'. Skipping promotion.")
exit(0)
