from qdrant_client import QdrantClient
from qdrant_client.local.qdrant_local import QdrantLocal

local = QdrantLocal(host="127.0.0.1", port=6333)
local.start()  # Start embedded Qdrant instance

client = QdrantClient(location="http://127.0.0.1:6333")
