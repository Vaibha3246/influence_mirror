from mlflow.tracking import MlflowClient

client = MlflowClient()
model_name = "youtube_tfidf_vectorizer"

versions = client.get_latest_versions(model_name, stages=["Staging"])
v = versions[0]

print("Run ID:", v.run_id)

print("\nArtifact list:")
arts = client.list_artifacts(v.run_id)
for a in arts:
    print(a)
