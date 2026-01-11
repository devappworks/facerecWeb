"""
Vector Database Service for Face Recognition using pgvector.

This service provides methods to:
- Store face embeddings in PostgreSQL
- Perform fast similarity search using HNSW index
- Manage persons and their face embeddings
"""

import os
import logging
import numpy as np
import psycopg2
from psycopg2.extras import execute_values
from pgvector.psycopg2 import register_vector
from typing import List, Tuple, Optional, Dict, Any
from contextlib import contextmanager

logger = logging.getLogger(__name__)


class VectorDBService:
    """PostgreSQL + pgvector service for face embedding storage and search."""

    def __init__(self, connection_string: str = None):
        """
        Initialize the vector database service.

        Args:
            connection_string: PostgreSQL connection string.
                Format: postgresql://user:password@host:port/database
                If not provided, reads from VECTOR_DB_URL environment variable.
        """
        self.connection_string = connection_string or os.getenv(
            'VECTOR_DB_URL',
            'postgresql://facerecadmin:1Un5J20uOLstxUxJYzvb76GCdAVOjZ5eN17Cq%2FQ%2FbrU%3D@localhost:5432/facerecognition'
        )
        self._conn = None
        self._initialize_connection()

    def _initialize_connection(self):
        """Create database connection and register vector type."""
        try:
            self._conn = psycopg2.connect(self.connection_string)
            register_vector(self._conn)
            logger.info("Connected to pgvector database successfully")
        except Exception as e:
            logger.error(f"Failed to connect to database: {e}")
            raise

    @contextmanager
    def get_cursor(self):
        """Context manager for database cursor with automatic commit/rollback."""
        cursor = self._conn.cursor()
        try:
            yield cursor
            self._conn.commit()
        except Exception as e:
            self._conn.rollback()
            logger.error(f"Database error: {e}")
            raise
        finally:
            cursor.close()

    def find_matches(
        self,
        query_embedding: np.ndarray,
        domain: str,
        threshold: float = 0.35,
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Find similar faces using cosine distance.

        Args:
            query_embedding: 512-dimensional ArcFace embedding (numpy array)
            domain: Domain to search in ('serbia', 'slovenia', etc.)
            threshold: Maximum cosine distance (lower = stricter). Default 0.35
            top_k: Maximum number of results to return

        Returns:
            List of matches, each containing:
            - name: Person's name
            - image_path: Path to matching image
            - distance: Cosine distance (0 = identical, 1 = opposite)
            - confidence: 1 - distance (higher = better match)
        """
        # Ensure embedding is a list for SQL
        if isinstance(query_embedding, np.ndarray):
            emb_list = query_embedding.tolist()
        else:
            emb_list = list(query_embedding)

        query = """
            SELECT
                p.name,
                p.id as person_id,
                f.image_path,
                f.embedding <=> %s::vector AS distance
            FROM face_embeddings f
            JOIN persons p ON f.person_id = p.id
            WHERE p.domain = %s
              AND f.embedding <=> %s::vector < %s
            ORDER BY f.embedding <=> %s::vector
            LIMIT %s
        """

        with self.get_cursor() as cursor:
            cursor.execute(query, (emb_list, domain, emb_list, threshold, emb_list, top_k))
            rows = cursor.fetchall()

        results = []
        for row in rows:
            results.append({
                'name': row[0],
                'person_id': row[1],
                'image_path': row[2],
                'distance': float(row[3]),
                'confidence': 1.0 - float(row[3])  # Convert distance to confidence
            })

        return results

    def find_best_match(
        self,
        query_embedding: np.ndarray,
        domain: str,
        threshold: float = 0.35
    ) -> Optional[Dict[str, Any]]:
        """
        Find the single best matching face.

        Args:
            query_embedding: 512-dimensional ArcFace embedding
            domain: Domain to search in
            threshold: Maximum cosine distance

        Returns:
            Best match dict or None if no match found
        """
        matches = self.find_matches(query_embedding, domain, threshold, top_k=1)
        return matches[0] if matches else None

    def add_person(self, name: str, domain: str, wikidata_id: str = None, occupation: str = None) -> int:
        """
        Add a new person or get existing person ID.

        Args:
            name: Person's name
            domain: Domain ('serbia', 'slovenia', etc.)
            wikidata_id: Optional Wikidata identifier
            occupation: Optional occupation

        Returns:
            Person ID (existing or newly created)
        """
        with self.get_cursor() as cursor:
            # Try to insert, on conflict return existing ID
            cursor.execute("""
                INSERT INTO persons (name, domain, wikidata_id, occupation)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (name, domain)
                DO UPDATE SET
                    wikidata_id = COALESCE(EXCLUDED.wikidata_id, persons.wikidata_id),
                    occupation = COALESCE(EXCLUDED.occupation, persons.occupation),
                    updated_at = NOW()
                RETURNING id
            """, (name, domain, wikidata_id, occupation))
            person_id = cursor.fetchone()[0]

        logger.debug(f"Person '{name}' in domain '{domain}' has ID {person_id}")
        return person_id

    def add_embedding(
        self,
        person_id: int,
        image_path: str,
        embedding: np.ndarray,
        confidence: float = None
    ) -> int:
        """
        Add a face embedding for a person.

        Args:
            person_id: ID of the person
            image_path: Path to the source image
            embedding: 512-dimensional ArcFace embedding
            confidence: Optional detection confidence

        Returns:
            Embedding ID
        """
        if isinstance(embedding, np.ndarray):
            emb_list = embedding.tolist()
        else:
            emb_list = list(embedding)

        with self.get_cursor() as cursor:
            cursor.execute("""
                INSERT INTO face_embeddings (person_id, image_path, embedding, confidence)
                VALUES (%s, %s, %s::vector, %s)
                ON CONFLICT (image_path)
                DO UPDATE SET
                    embedding = EXCLUDED.embedding,
                    confidence = EXCLUDED.confidence
                RETURNING id
            """, (person_id, image_path, emb_list, confidence))
            embedding_id = cursor.fetchone()[0]

        return embedding_id

    def add_person_with_embedding(
        self,
        name: str,
        domain: str,
        image_path: str,
        embedding: np.ndarray,
        confidence: float = None
    ) -> Tuple[int, int]:
        """
        Convenience method to add person and embedding in one call.

        Args:
            name: Person's name
            domain: Domain
            image_path: Image path
            embedding: Face embedding
            confidence: Optional confidence

        Returns:
            Tuple of (person_id, embedding_id)
        """
        person_id = self.add_person(name, domain)
        embedding_id = self.add_embedding(person_id, image_path, embedding, confidence)
        return person_id, embedding_id

    def bulk_add_embeddings(
        self,
        embeddings_data: List[Dict[str, Any]],
        batch_size: int = 1000
    ) -> int:
        """
        Bulk insert embeddings for migration.

        Args:
            embeddings_data: List of dicts with keys:
                - name: Person name
                - domain: Domain
                - image_path: Image path
                - embedding: Embedding array
            batch_size: Number of records per batch

        Returns:
            Total number of embeddings inserted
        """
        total_inserted = 0

        for i in range(0, len(embeddings_data), batch_size):
            batch = embeddings_data[i:i + batch_size]

            with self.get_cursor() as cursor:
                for item in batch:
                    try:
                        person_id = self.add_person(item['name'], item['domain'])
                        self.add_embedding(
                            person_id,
                            item['image_path'],
                            item['embedding']
                        )
                        total_inserted += 1
                    except Exception as e:
                        logger.warning(f"Failed to insert {item['image_path']}: {e}")
                        continue

            logger.info(f"Inserted batch {i // batch_size + 1}, total: {total_inserted}")

        return total_inserted

    def delete_person(self, name: str, domain: str) -> bool:
        """
        Delete a person and all their embeddings.

        Args:
            name: Person's name
            domain: Domain

        Returns:
            True if person was deleted, False if not found
        """
        with self.get_cursor() as cursor:
            cursor.execute("""
                DELETE FROM persons
                WHERE name = %s AND domain = %s
                RETURNING id
            """, (name, domain))
            result = cursor.fetchone()

        if result:
            logger.info(f"Deleted person '{name}' from domain '{domain}'")
            return True
        return False

    def delete_embedding(self, image_path: str) -> bool:
        """
        Delete a specific embedding by image path.

        Args:
            image_path: Path to the image

        Returns:
            True if deleted, False if not found
        """
        with self.get_cursor() as cursor:
            cursor.execute("""
                DELETE FROM face_embeddings
                WHERE image_path = %s
                RETURNING id
            """, (image_path,))
            result = cursor.fetchone()

        return result is not None

    def get_stats(self, domain: str = None) -> Dict[str, int]:
        """
        Get database statistics.

        Args:
            domain: Optional domain filter

        Returns:
            Dict with counts: persons, embeddings, embeddings_per_person
        """
        with self.get_cursor() as cursor:
            if domain:
                cursor.execute("""
                    SELECT
                        (SELECT COUNT(*) FROM persons WHERE domain = %s) as person_count,
                        (SELECT COUNT(*) FROM face_embeddings f
                         JOIN persons p ON f.person_id = p.id
                         WHERE p.domain = %s) as embedding_count
                """, (domain, domain))
            else:
                cursor.execute("""
                    SELECT
                        (SELECT COUNT(*) FROM persons) as person_count,
                        (SELECT COUNT(*) FROM face_embeddings) as embedding_count
                """)

            row = cursor.fetchone()

        person_count = row[0]
        embedding_count = row[1]

        return {
            'persons': person_count,
            'embeddings': embedding_count,
            'embeddings_per_person': round(embedding_count / person_count, 1) if person_count > 0 else 0
        }

    def get_all_persons(self, domain: str) -> List[Dict[str, Any]]:
        """
        Get all persons in a domain.

        Args:
            domain: Domain to query

        Returns:
            List of person dicts with id, name, embedding_count
        """
        with self.get_cursor() as cursor:
            cursor.execute("""
                SELECT p.id, p.name, COUNT(f.id) as embedding_count
                FROM persons p
                LEFT JOIN face_embeddings f ON f.person_id = p.id
                WHERE p.domain = %s
                GROUP BY p.id, p.name
                ORDER BY p.name
            """, (domain,))
            rows = cursor.fetchall()

        return [{'id': r[0], 'name': r[1], 'embedding_count': r[2]} for r in rows]

    def health_check(self) -> bool:
        """
        Check database connection health.

        Returns:
            True if healthy, False otherwise
        """
        try:
            with self.get_cursor() as cursor:
                cursor.execute("SELECT 1")
                return True
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return False

    def close(self):
        """Close database connection."""
        if self._conn:
            self._conn.close()
            logger.info("Database connection closed")


# Singleton instance for the application
_vector_db_instance = None


def get_vector_db() -> VectorDBService:
    """
    Get or create the singleton VectorDBService instance.

    Returns:
        VectorDBService instance
    """
    global _vector_db_instance
    if _vector_db_instance is None:
        _vector_db_instance = VectorDBService()
    return _vector_db_instance
