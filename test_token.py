from transformers import AutoTokenizer

# 你复制好的路径，比如：
path = "/path/to/huggingface/hub/models--bert-base-uncased/snapshots/86b5e0934494bd15c9632b12f734a8a67f723594"

# 最小化测试
tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
print("Tokenizer vocab size:", len(tokenizer))
