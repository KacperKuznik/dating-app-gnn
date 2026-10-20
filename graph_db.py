"""Neo4j storage for the speed dating graph.

(:Person {id, wave, gender, age, ...survey answers})-[:MATCH]->(:Person)
(:Person)-[:NO_MATCH]->(:Person)

Every date is stored: MATCH when both said yes, NO_MATCH otherwise, so the real
non-matches stay available for training. A date is mutual, so the arrow (lower id ->
higher id) means nothing; query without it, e.g. (a)-[:MATCH]-(b).
Connection comes from NEO4J_URI / NEO4J_USER / NEO4J_PASSWORD, with defaults that fit
docker-compose.yml.
"""
import os

import pandas as pd
from neo4j import GraphDatabase

URI = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
AUTH = (os.environ.get("NEO4J_USER", "neo4j"), os.environ.get("NEO4J_PASSWORD", "speeddating"))


def connect():
    return GraphDatabase.driver(URI, auth=AUTH)


def _records(df):
    # NaN becomes None, so a blank survey answer is simply not stored as a property
    return df.astype(object).where(df.notna(), None).to_dict("records")


def write_graph(people, pairs):
    """Replace whatever is in the database with these people and dates."""
    with connect() as driver:
        driver.execute_query("MATCH (n) DETACH DELETE n")
        driver.execute_query("CREATE CONSTRAINT person_id IF NOT EXISTS FOR (p:Person) REQUIRE p.id IS UNIQUE")
        driver.execute_query("UNWIND $rows AS row CREATE (p:Person) SET p = row", rows=_records(people))
        # Cypher can't take the relationship type as a parameter, so one query per type
        for rel_type, outcome in (("MATCH", 1), ("NO_MATCH", 0)):
            driver.execute_query(
                f"""
                UNWIND $rows AS row
                MATCH (a:Person {{id: row.person_a}}), (b:Person {{id: row.person_b}})
                CREATE (a)-[:{rel_type}]->(b)
                """,
                rows=_records(pairs[pairs["match"] == outcome]),
            )


def read_people():
    with connect() as driver:
        records, _, _ = driver.execute_query("MATCH (p:Person) RETURN properties(p) AS p ORDER BY p.id")
    return pd.DataFrame([r["p"] for r in records])


def read_pairs():
    with connect() as driver:
        records, _, _ = driver.execute_query(
            """
            MATCH (a:Person)-[d:MATCH|NO_MATCH]->(b:Person)
            RETURN a.id AS person_a, b.id AS person_b,
                   CASE type(d) WHEN "MATCH" THEN 1 ELSE 0 END AS match
            ORDER BY person_a, person_b
            """
        )
    return pd.DataFrame([r.data() for r in records], columns=["person_a", "person_b", "match"])
