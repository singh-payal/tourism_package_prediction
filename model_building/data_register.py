from huggingface_hub.utils import RepositoryNotFoundError, HfHubHTTPError
from huggingface_hub import HfApi, create_repo
import os


repo_id = "singhpayal/tourism_package_prediction_dataset"
repo_type = "dataset"
local_data_folder = "data"
local_data_file = os.path.join(local_data_folder, "tourism.csv")

# Initialize API client
api = HfApi(token=os.getenv("HF_TOKEN"))

# Ensure the local data folder exists
os.makedirs(local_data_folder, exist_ok=True)

# Step 0: Check if the local data file exists
if not os.path.exists(local_data_file):
    print(f"Error: Local data file '{local_data_file}' not found. Please ensure the 'tourism.csv' is in the 'data' folder.")
    exit(1) # Exit if the file is not found

# Step 1: Check if the space exists
try:
    api.repo_info(repo_id=repo_id, repo_type=repo_type)
    print(f"Space '{repo_id}' already exists. Using it.")
except RepositoryNotFoundError:
    print(f"Space '{repo_id}' not found. Creating new space...")
    create_repo(repo_id=repo_id, repo_type=repo_type, private=False)
    print(f"Repository '{repo_id}' created.")

# Upload the local data folder to the Hugging Face dataset repository
api.upload_folder(
    folder_path=local_data_folder,
    repo_id=repo_id,
    repo_type=repo_type,
)
print("Data uploaded to Hugging Face Dataset repository successfully.")
