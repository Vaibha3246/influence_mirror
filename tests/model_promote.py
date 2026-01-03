import mlflow

MLFLOW_TRACKING_URI = "http://ec2-13-62-47-8.eu-north-1.compute.amazonaws.com:5000"
MODEL_NAME = "yt_chrome_plugin_model"

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
client = mlflow.tracking.MlflowClient()

#  Get latest STAGING model
staging_versions = client.get_latest_versions(
    MODEL_NAME, stages=["Staging"]
)

if not staging_versions:
    raise RuntimeError(" No model in Staging to promote")

staging_version = staging_versions[0].version

print(f" Promoting version {staging_version} to Production")

#  Promote to Production & auto-archive old Production
client.transition_model_version_stage(
    name=MODEL_NAME,
    version=staging_version,
    stage="Production",
    archive_existing_versions=True
)

print(" Promotion successful")
print(" Previous Production model archived automatically")
