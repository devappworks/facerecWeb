#!/usr/bin/env python3
"""Test EmbeddingMatcher loading"""
import sys
sys.path.insert(0, '/home/facereco/facerecWeb')

from app.services.embedding_matcher import EmbeddingMatcher

print("Testing EmbeddingMatcher...")
print("-" * 60)

for domain in ['serbia', 'slovenia']:
    print(f"\nDomain: {domain}")
    matcher = EmbeddingMatcher(domain)
    loaded = matcher.load_database()
    print(f"  Loaded: {loaded}")
    print(f"  Embeddings count: {len(matcher.embeddings)}")
    if matcher.embeddings:
        print(f"  First embedding shape: {matcher.embeddings[0].shape}")
        print(f"  First identity: {matcher.identities[0][:80]}...")
        print(f"  First person name: {matcher.person_names[0]}")

print("\n✅ EmbeddingMatcher test complete!")
