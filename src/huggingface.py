from huggingface_hub import upload_file

repo_id = "EvelynZ260/recommender"  # 你的 dataset 仓库
files_to_upload = [
    "data/mysql_qa_with_vectors.pkl",
    "data/mysql_qa.faiss",
    "data/mysql_qa_embeddings.npy"
]

for file_path in files_to_upload:
    upload_file(
        path_or_fileobj=file_path,
        path_in_repo=file_path.split("/", 1)[1],  # 去掉 "data/" 前缀
        repo_id=repo_id,
        repo_type="dataset"
    )
