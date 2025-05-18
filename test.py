from qdrant_client import QdrantClient, models

client = QdrantClient(path="./qdrant_data")

results = client.scroll(
    collection_name="transcript_search",
    scroll_filter=models.Filter(
        must=[
            models.FieldCondition(
                key="metadata.user_id", match=models.MatchValue(value="user_204")
            ),
            models.FieldCondition(
                key="metadata.language", match=models.MatchValue(value="hi")
            )
        ]
    ),
    limit=5
)

print("Found:", len(results[0]))
for point in results[0]:
    print(point.payload)



# from qdrant_client import QdrantClient

# client = QdrantClient(path="./qdrant_data")
# points, _ = client.scroll(collection_name="transcript_search", limit=5)

# for pt in points:
#     print(pt.payload)
