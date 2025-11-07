"""
Agent Negotiation Framework
============================

Multi-agent negotiation and consensus building:
- Bilateral and multi-party negotiations
- Utility functions and preferences
- Concession strategies
- Agreement protocols
- Trust and reputation

Author: Brill Consulting
"""

from typing import List, Dict, Optional, Any
from dataclasses import dataclass
from datetime import datetime
import random


@dataclass
class Offer:
    """Represents a negotiation offer."""
    from_agent: str
    to_agent: str
    value: float
    round_num: int
    timestamp: str


class NegotiationAgent:
    """Agent capable of negotiation."""

    def __init__(
        self,
        name: str,
        initial_offer: float,
        reservation_price: float,
        concession_rate: float = 0.1
    ):
        """Initialize negotiation agent."""
        self.name = name
        self.initial_offer = initial_offer
        self.reservation_price = reservation_price
        self.concession_rate = concession_rate
        self.current_offer = initial_offer
        self.offers_made: List[Offer] = []
        self.trust_score = 0.5

        print(f"💼 {name} ready to negotiate")
        print(f"   Initial offer: ${initial_offer:.2f}")
        print(f"   Reservation price: ${reservation_price:.2f}")

    def make_offer(
        self,
        round_num: int,
        opponent_name: str,
        opponent_last_offer: Optional[float] = None
    ) -> Offer:
        """Make negotiation offer."""
        if opponent_last_offer:
            # Calculate concession
            gap = abs(self.current_offer - opponent_last_offer)
            concession = gap * self.concession_rate

            # Move towards opponent
            if self.current_offer > opponent_last_offer:
                self.current_offer = max(
                    self.reservation_price,
                    self.current_offer - concession
                )
            else:
                self.current_offer = min(
                    self.reservation_price,
                    self.current_offer + concession
                )

        offer = Offer(
            from_agent=self.name,
            to_agent=opponent_name,
            value=self.current_offer,
            round_num=round_num,
            timestamp=datetime.now().isoformat()
        )

        self.offers_made.append(offer)
        return offer

    def evaluate_offer(self, offer: Offer) -> bool:
        """Evaluate if offer is acceptable."""
        if self.initial_offer > self.reservation_price:
            # Buyer: accept if offer <= reservation
            return offer.value <= self.reservation_price
        else:
            # Seller: accept if offer >= reservation
            return offer.value >= self.reservation_price

    def calculate_utility(self, price: float) -> float:
        """Calculate utility of a price."""
        if self.initial_offer > self.reservation_price:
            # Buyer: lower price = higher utility
            return max(0, 1 - (price / self.reservation_price))
        else:
            # Seller: higher price = higher utility
            return min(1, price / self.reservation_price)


class NegotiationSession:
    """Manages negotiation between agents."""

    def __init__(
        self,
        participants: List[NegotiationAgent],
        max_rounds: int = 10
    ):
        """Initialize negotiation session."""
        self.participants = participants
        self.max_rounds = max_rounds
        self.history: List[Offer] = []
        self.agreement_reached = False
        self.final_price = None

        print(f"\n🤝 Negotiation session started")
        print(f"   Participants: {', '.join(p.name for p in participants)}")
        print(f"   Max rounds: {max_rounds}")

    def negotiate(self) -> Dict[str, Any]:
        """Run negotiation process."""
        print(f"\n{'='*60}")
        print("Negotiation Progress")
        print(f"{'='*60}")

        if len(self.participants) != 2:
            return {"error": "Only bilateral negotiation supported"}

        agent_a, agent_b = self.participants

        for round_num in range(1, self.max_rounds + 1):
            print(f"\nRound {round_num}")

            # Agent A makes offer
            if round_num == 1:
                offer_a = agent_a.make_offer(round_num, agent_b.name)
            else:
                offer_a = agent_a.make_offer(
                    round_num,
                    agent_b.name,
                    self.history[-1].value
                )

            print(f"   {agent_a.name} offers: ${offer_a.value:.2f}")
            self.history.append(offer_a)

            # Agent B evaluates
            if agent_b.evaluate_offer(offer_a):
                self.agreement_reached = True
                self.final_price = offer_a.value
                print(f"   ✓ {agent_b.name} accepts!")
                break

            # Agent B makes counter-offer
            offer_b = agent_b.make_offer(round_num, agent_a.name, offer_a.value)
            print(f"   {agent_b.name} counters: ${offer_b.value:.2f}")
            self.history.append(offer_b)

            # Agent A evaluates
            if agent_a.evaluate_offer(offer_b):
                self.agreement_reached = True
                self.final_price = offer_b.value
                print(f"   ✓ {agent_a.name} accepts!")
                break

        result = {
            "agreement_reached": self.agreement_reached,
            "final_price": self.final_price,
            "rounds": round_num,
            "total_offers": len(self.history)
        }

        if self.agreement_reached:
            print(f"\n✅ Agreement reached at ${self.final_price:.2f}")
        else:
            print(f"\n❌ No agreement after {round_num} rounds")

        return result


def demo():
    """Demonstrate agent negotiation."""
    print("=" * 60)
    print("Agent Negotiation Framework Demo")
    print("=" * 60)

    # Create negotiating agents
    buyer = NegotiationAgent(
        name="Buyer",
        initial_offer=100,
        reservation_price=150,
        concession_rate=0.12
    )

    seller = NegotiationAgent(
        name="Seller",
        initial_offer=200,
        reservation_price=120,
        concession_rate=0.1
    )

    # Run negotiation
    session = NegotiationSession(
        participants=[buyer, seller],
        max_rounds=10
    )

    result = session.negotiate()

    # Calculate utilities
    if result['agreement_reached']:
        print(f"\n📊 Outcome Analysis:")
        print(f"   Buyer utility: {buyer.calculate_utility(result['final_price']):.2f}")
        print(f"   Seller utility: {seller.calculate_utility(result['final_price']):.2f}")


if __name__ == "__main__":
    demo()
