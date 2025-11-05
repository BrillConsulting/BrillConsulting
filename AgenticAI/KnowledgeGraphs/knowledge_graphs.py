"""
Knowledge Graphs - Agent Knowledge Representation
==================================================

Graph-based knowledge representation for agents with entities, relationships,
and semantic querying capabilities.

Author: Brill Consulting
"""

from typing import List, Dict, Set, Optional, Tuple
from dataclasses import dataclass, field
from collections import defaultdict


@dataclass
class Entity:
    """Knowledge graph entity/node."""
    id: str
    type: str
    properties: Dict[str, any] = field(default_factory=dict)

    def __hash__(self):
        return hash(self.id)

    def __eq__(self, other):
        return isinstance(other, Entity) and self.id == other.id


@dataclass
class Relation:
    """Knowledge graph relationship/edge."""
    from_entity: str
    to_entity: str
    relation_type: str
    properties: Dict[str, any] = field(default_factory=dict)


class KnowledgeGraph:
    """In-memory knowledge graph with semantic querying."""

    def __init__(self, name: str = "default"):
        self.name = name
        self.entities: Dict[str, Entity] = {}
        self.relations: List[Relation] = []
        self.entity_relations: Dict[str, List[Relation]] = defaultdict(list)
        self.reverse_relations: Dict[str, List[Relation]] = defaultdict(list)

    def add_entity(self, entity: Entity):
        """Add entity to graph."""
        self.entities[entity.id] = entity

    def add_relation(self, relation: Relation):
        """Add relationship to graph."""
        self.relations.append(relation)
        self.entity_relations[relation.from_entity].append(relation)
        self.reverse_relations[relation.to_entity].append(relation)

    def get_entity(self, entity_id: str) -> Optional[Entity]:
        """Get entity by ID."""
        return self.entities.get(entity_id)

    def find_entities_by_type(self, entity_type: str) -> List[Entity]:
        """Find all entities of given type."""
        return [e for e in self.entities.values() if e.type == entity_type]

    def get_outgoing_relations(self, entity_id: str) -> List[Relation]:
        """Get all outgoing relations from entity."""
        return self.entity_relations[entity_id]

    def get_incoming_relations(self, entity_id: str) -> List[Relation]:
        """Get all incoming relations to entity."""
        return self.reverse_relations[entity_id]

    def find_related_entities(self, entity_id: str, relation_type: Optional[str] = None) -> List[Entity]:
        """Find entities related to given entity."""
        related = []
        for rel in self.get_outgoing_relations(entity_id):
            if relation_type is None or rel.relation_type == relation_type:
                related_entity = self.get_entity(rel.to_entity)
                if related_entity:
                    related.append(related_entity)
        return related

    def query_path(self, from_id: str, to_id: str, max_depth: int = 3) -> Optional[List[str]]:
        """Find path between two entities."""
        if from_id not in self.entities or to_id not in self.entities:
            return None

        visited = set()
        queue = [(from_id, [from_id])]

        while queue:
            current, path = queue.pop(0)

            if current == to_id:
                return path

            if len(path) > max_depth:
                continue

            if current in visited:
                continue

            visited.add(current)

            for rel in self.get_outgoing_relations(current):
                if rel.to_entity not in visited:
                    queue.append((rel.to_entity, path + [rel.to_entity]))

        return None

    def get_neighbors(self, entity_id: str, depth: int = 1) -> Set[str]:
        """Get all neighbors within given depth."""
        neighbors = set()
        current_level = {entity_id}

        for _ in range(depth):
            next_level = set()
            for entity in current_level:
                for rel in self.get_outgoing_relations(entity):
                    next_level.add(rel.to_entity)
                for rel in self.get_incoming_relations(entity):
                    next_level.add(rel.from_entity)
            neighbors.update(next_level)
            current_level = next_level

        neighbors.discard(entity_id)
        return neighbors

    def get_statistics(self) -> Dict:
        """Get graph statistics."""
        return {
            'entities': len(self.entities),
            'relations': len(self.relations),
            'entity_types': len(set(e.type for e in self.entities.values())),
            'relation_types': len(set(r.relation_type for r in self.relations))
        }


class SemanticQuery:
    """Semantic querying over knowledge graph."""

    @staticmethod
    def find_by_property(kg: KnowledgeGraph, entity_type: str,
                         property_name: str, property_value: any) -> List[Entity]:
        """Find entities by property value."""
        result = []
        for entity in kg.find_entities_by_type(entity_type):
            if entity.properties.get(property_name) == property_value:
                result.append(entity)
        return result

    @staticmethod
    def find_connected_subgraph(kg: KnowledgeGraph, start_id: str, max_size: int = 10) -> KnowledgeGraph:
        """Extract connected subgraph starting from entity."""
        subgraph = KnowledgeGraph(name=f"subgraph_{start_id}")

        visited = set()
        queue = [start_id]
        count = 0

        while queue and count < max_size:
            current_id = queue.pop(0)

            if current_id in visited:
                continue

            visited.add(current_id)
            count += 1

            entity = kg.get_entity(current_id)
            if entity:
                subgraph.add_entity(entity)

            for rel in kg.get_outgoing_relations(current_id):
                subgraph.add_relation(rel)
                if rel.to_entity not in visited:
                    queue.append(rel.to_entity)

        return subgraph


def demo():
    """Demonstration of Knowledge Graphs."""
    print("Knowledge Graphs - Agent Knowledge Representation Demo")
    print("=" * 70)

    # Create knowledge graph
    kg = KnowledgeGraph(name="social_network")

    # Add entities
    print("\n1️⃣  Creating Knowledge Graph")
    print("-" * 70)

    # People
    alice = Entity("alice", "Person", {"name": "Alice", "age": 30})
    bob = Entity("bob", "Person", {"name": "Bob", "age": 28})
    carol = Entity("carol", "Person", {"name": "Carol", "age": 32})
    dave = Entity("dave", "Person", {"name": "Dave", "age": 29})

    # Organizations
    acme = Entity("acme", "Company", {"name": "Acme Corp", "industry": "Tech"})
    beta = Entity("beta", "Company", {"name": "Beta Inc", "industry": "Finance"})

    # Projects
    proj_x = Entity("proj_x", "Project", {"name": "Project X", "status": "active"})

    kg.add_entity(alice)
    kg.add_entity(bob)
    kg.add_entity(carol)
    kg.add_entity(dave)
    kg.add_entity(acme)
    kg.add_entity(beta)
    kg.add_entity(proj_x)

    print(f"✓ Added {len(kg.entities)} entities")

    # Add relationships
    print("\n2️⃣  Adding Relationships")
    print("-" * 70)

    relations = [
        Relation("alice", "bob", "KNOWS", {"since": 2020}),
        Relation("bob", "carol", "KNOWS", {"since": 2019}),
        Relation("carol", "dave", "KNOWS", {"since": 2021}),
        Relation("alice", "acme", "WORKS_FOR", {"position": "Engineer"}),
        Relation("bob", "acme", "WORKS_FOR", {"position": "Designer"}),
        Relation("carol", "beta", "WORKS_FOR", {"position": "Manager"}),
        Relation("alice", "proj_x", "CONTRIBUTES_TO", {}),
        Relation("bob", "proj_x", "CONTRIBUTES_TO", {}),
    ]

    for rel in relations:
        kg.add_relation(rel)

    print(f"✓ Added {len(kg.relations)} relationships")

    # Query graph
    print("\n3️⃣  Querying Knowledge Graph")
    print("-" * 70)

    # Find people
    people = kg.find_entities_by_type("Person")
    print(f"People in graph: {[p.properties['name'] for p in people]}")

    # Find Alice's colleagues
    alice_company = kg.find_related_entities("alice", "WORKS_FOR")
    if alice_company:
        print(f"\nAlice works for: {alice_company[0].properties['name']}")

    # Find who Alice knows
    alice_knows = kg.find_related_entities("alice", "KNOWS")
    print(f"Alice knows: {[kg.get_entity(e.id).properties['name'] for e in alice_knows]}")

    # Path finding
    print("\n4️⃣  Path Finding")
    print("-" * 70)

    path = kg.query_path("alice", "dave")
    if path:
        path_names = [kg.get_entity(eid).properties.get('name', eid) for eid in path]
        print(f"Path from Alice to Dave: {' → '.join(path_names)}")

    # Neighbors
    print("\n5️⃣  Finding Neighbors")
    print("-" * 70)

    neighbors_1 = kg.get_neighbors("alice", depth=1)
    print(f"Alice's 1-hop neighbors: {[kg.get_entity(nid).properties.get('name', nid) for nid in neighbors_1]}")

    neighbors_2 = kg.get_neighbors("alice", depth=2)
    print(f"Alice's 2-hop neighbors: {len(neighbors_2)} entities")

    # Semantic queries
    print("\n6️⃣  Semantic Queries")
    print("-" * 70)

    tech_companies = SemanticQuery.find_by_property(kg, "Company", "industry", "Tech")
    print(f"Tech companies: {[c.properties['name'] for c in tech_companies]}")

    # Extract subgraph
    subgraph = SemanticQuery.find_connected_subgraph(kg, "alice", max_size=5)
    print(f"\nSubgraph from Alice: {len(subgraph.entities)} entities, {len(subgraph.relations)} relations")

    # Statistics
    print("\n7️⃣  Graph Statistics")
    print("-" * 70)

    stats = kg.get_statistics()
    for key, value in stats.items():
        print(f"  {key}: {value}")

    print("\n" + "=" * 70)
    print("✓ Demo Complete!")


if __name__ == '__main__':
    demo()
