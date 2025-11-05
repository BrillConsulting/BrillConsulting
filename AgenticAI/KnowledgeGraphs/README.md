# Knowledge Graphs

Graph-based knowledge representation for AI agents using Neo4j database for semantic reasoning and inference.

## Features

- **Neo4j Integration**: Graph database for knowledge storage
- **Entity Modeling**: Represent concepts, objects, and agents
- **Relationship Mapping**: Define semantic connections
- **Cypher Queries**: Powerful graph query language
- **Knowledge Inference**: Derive new facts from existing knowledge
- **Semantic Reasoning**: Understand context and meaning
- **Graph Traversal**: Navigate knowledge relationships
- **Schema Evolution**: Flexible knowledge structure

## Quick Start

```python
from neo4j import GraphDatabase
from knowledge_graph import KnowledgeGraph

# Initialize connection
kg = KnowledgeGraph(uri="bolt://localhost:7687", user="neo4j", password="password")

# Create entities
kg.create_entity("Person", {"name": "Alice", "age": 30, "role": "AI Agent"})
kg.create_entity("Skill", {"name": "Python Programming", "level": "Expert"})
kg.create_entity("Task", {"name": "Data Analysis", "priority": "high"})

# Create relationships
kg.create_relationship("Alice", "HAS_SKILL", "Python Programming", {
    "years_experience": 8,
    "proficiency": 0.95
})

kg.create_relationship("Alice", "CAN_PERFORM", "Data Analysis")

# Query knowledge
# Find all skills of a person
skills = kg.query("""
    MATCH (p:Person {name: 'Alice'})-[:HAS_SKILL]->(s:Skill)
    RETURN s.name as skill, s.level as level
""")

# Find agents who can perform a task
agents = kg.query("""
    MATCH (a:Person)-[:CAN_PERFORM]->(t:Task {name: 'Data Analysis'})
    RETURN a.name as agent
""")
```

## Use Cases

- **AI Agent Memory**: Store agent experiences and knowledge
- **Recommendation Systems**: Find related entities and patterns
- **Question Answering**: Semantic search and inference
- **Expert Systems**: Model domain expertise
- **Semantic Search**: Context-aware information retrieval
- **Ontology Management**: Define domain vocabularies

## Graph Patterns

### Entity-Relationship Model
```python
# Person -[:KNOWS]-> Person
# Person -[:WORKS_ON]-> Project
# Project -[:USES]-> Technology
# Technology -[:REQUIRES]-> Skill
```

### Knowledge Inference
```python
# If Alice KNOWS Bob and Bob KNOWS Charlie
# Then Alice MIGHT_KNOW Charlie (friend-of-friend)

kg.infer_relationships("""
    MATCH (a:Person)-[:KNOWS]->(b:Person)-[:KNOWS]->(c:Person)
    WHERE NOT (a)-[:KNOWS]->(c)
    MERGE (a)-[:MIGHT_KNOW {confidence: 0.7}]->(c)
""")
```

### Semantic Queries
```python
# Find experts in a technology stack
experts = kg.query("""
    MATCH (p:Person)-[r:HAS_SKILL]->(s:Skill)
    WHERE s.name IN ['Python', 'Machine Learning', 'Deep Learning']
      AND r.proficiency > 0.8
    RETURN p.name, count(s) as skill_count
    ORDER BY skill_count DESC
""")
```

## Advanced Features

### Graph Algorithms
```python
# PageRank for influence
kg.run_algorithm("pagerank", {
    "nodeProjection": "Person",
    "relationshipProjection": "KNOWS"
})

# Community Detection
communities = kg.detect_communities()

# Shortest Path
path = kg.shortest_path("Alice", "Charlie", relationship="KNOWS")
```

### Temporal Knowledge
```python
# Add time-based relationships
kg.create_relationship("Alice", "WORKED_ON", "Project_X", {
    "start_date": "2023-01-01",
    "end_date": "2023-12-31",
    "role": "Lead Developer"
})

# Query historical knowledge
past_projects = kg.query("""
    MATCH (p:Person {name: 'Alice'})-[r:WORKED_ON]->(proj:Project)
    WHERE r.end_date < date()
    RETURN proj.name, r.role
""")
```

## Best Practices

- Design clear entity and relationship types
- Use consistent naming conventions
- Index frequently queried properties
- Normalize data to avoid duplication
- Use parameters in Cypher queries
- Implement proper error handling
- Regular backups of knowledge base
- Monitor query performance

## Author

Brill Consulting
