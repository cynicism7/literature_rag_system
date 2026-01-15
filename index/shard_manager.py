#多盘管理
import os

class ShardManager:
    def __init__(self, index_root):
        self.shards = [
            os.path.join(index_root, d)
            for d in os.listdir(index_root)
        ]

    def iter_indices(self):
        for shard in self.shards:
            yield shard
