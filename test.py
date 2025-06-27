from huggingface_hub import snapshot_download

# Tải cả repo
snapshot_download(repo_id="mengtingwei/magicface", local_dir="./utils")
