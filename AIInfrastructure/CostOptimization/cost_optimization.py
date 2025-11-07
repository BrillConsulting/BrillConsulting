"""
Cost Optimization Framework
============================

Track, analyze, and optimize AI infrastructure costs

Author: Brill Consulting
"""

from typing import Dict, List, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
import random


@dataclass
class CostMetrics:
    """Infrastructure cost metrics."""
    total_cost_usd: float
    compute_cost: float
    storage_cost: float
    network_cost: float
    period: str
    timestamp: str


class CostOptimizer:
    """AI infrastructure cost optimizer."""

    def __init__(
        self,
        cloud_provider: str = "aws",
        region: str = "us-east-1"
    ):
        """Initialize cost optimizer."""
        self.cloud_provider = cloud_provider
        self.region = region
        self.cost_history: List[CostMetrics] = []

        print(f"💰 Cost Optimizer initialized")
        print(f"   Provider: {cloud_provider}")
        print(f"   Region: {region}")

    def analyze_costs(
        self,
        time_range: str = "last_30_days"
    ) -> Dict[str, Any]:
        """Analyze infrastructure costs."""
        print(f"\n📊 Analyzing costs: {time_range}")

        # Simulate cost data
        daily_compute = random.uniform(800, 1200)
        daily_storage = random.uniform(100, 200)
        daily_network = random.uniform(50, 100)

        days = 30 if "30" in time_range else 7
        total_cost = (daily_compute + daily_storage + daily_network) * days

        analysis = {
            "time_range": time_range,
            "total_cost_usd": round(total_cost, 2),
            "breakdown": {
                "compute": round(daily_compute * days, 2),
                "storage": round(daily_storage * days, 2),
                "network": round(daily_network * days, 2)
            },
            "daily_average": round(total_cost / days, 2),
            "largest_cost_driver": "compute"
        }

        print(f"   Total cost: ${analysis['total_cost_usd']:,.2f}")
        print(f"   Daily average: ${analysis['daily_average']:,.2f}")
        print(f"   Breakdown:")
        for category, cost in analysis['breakdown'].items():
            pct = (cost / total_cost) * 100
            print(f"      {category}: ${cost:,.2f} ({pct:.1f}%)")

        return analysis

    def get_recommendations(self) -> List[str]:
        """Get cost optimization recommendations."""
        print(f"\n💡 Cost Optimization Recommendations")

        recommendations = [
            "Use spot instances: Save 60-70%",
            "Right-size GPU instances: Save 20-30%",
            "Enable auto-scaling: Save 15-25%",
            "Use reserved instances for base load: Save 30-40%",
            "Implement caching: Reduce compute by 20%",
            "Optimize data transfer: Save 10-15% on network",
            "Use lifecycle policies for storage: Save 25%"
        ]

        for i, rec in enumerate(recommendations[:5], 1):
            print(f"   {i}. {rec}")

        return recommendations

    def enable_spot_instances(
        self,
        max_price_per_hour: float = 2.50,
        fallback_on_demand: bool = True
    ) -> Dict[str, Any]:
        """Enable spot instance optimization."""
        print(f"\n🎯 Enabling spot instances")
        print(f"   Max price: ${max_price_per_hour}/hour")
        print(f"   Fallback to on-demand: {fallback_on_demand}")

        # Simulate spot savings
        on_demand_cost = 8.00  # per hour
        spot_price = 2.40
        savings_percent = ((on_demand_cost - spot_price) / on_demand_cost) * 100

        config = {
            "enabled": True,
            "max_price": max_price_per_hour,
            "current_spot_price": spot_price,
            "on_demand_price": on_demand_cost,
            "estimated_savings_percent": round(savings_percent, 1),
            "estimated_monthly_savings_usd": round((on_demand_cost - spot_price) * 24 * 30, 2),
            "fallback_enabled": fallback_on_demand
        }

        print(f"   ✓ Spot instances enabled")
        print(f"   Estimated savings: {config['estimated_savings_percent']:.1f}%")
        print(f"   Monthly savings: ${config['estimated_monthly_savings_usd']:,.2f}")

        return config

    def forecast_costs(
        self,
        months_ahead: int = 3,
        growth_rate: float = 0.1
    ) -> Dict[str, List[float]]:
        """Forecast future costs."""
        print(f"\n📈 Forecasting costs: {months_ahead} months")
        print(f"   Growth rate: {growth_rate:.1%}/month")

        # Simulate current monthly cost
        current_monthly = 30000

        forecast = {
            "months": [],
            "costs": []
        }

        for month in range(1, months_ahead + 1):
            month_name = (datetime.now() + timedelta(days=30 * month)).strftime("%B %Y")
            projected_cost = current_monthly * ((1 + growth_rate) ** month)

            forecast["months"].append(month_name)
            forecast["costs"].append(round(projected_cost, 2))

            print(f"   {month_name}: ${projected_cost:,.2f}")

        return forecast

    def set_budget_alert(
        self,
        monthly_budget: float,
        alert_threshold: float = 0.8
    ) -> Dict[str, Any]:
        """Set budget alert."""
        print(f"\n🔔 Setting budget alert")
        print(f"   Monthly budget: ${monthly_budget:,.2f}")
        print(f"   Alert at: {alert_threshold:.0%}")

        alert_config = {
            "monthly_budget_usd": monthly_budget,
            "alert_threshold": alert_threshold,
            "alert_amount": monthly_budget * alert_threshold,
            "notifications": ["email", "slack"],
            "enabled": True
        }

        print(f"   ✓ Alert configured")
        print(f"   Will alert at: ${alert_config['alert_amount']:,.2f}")

        return alert_config

    def calculate_per_model_cost(
        self,
        model_name: str,
        requests_per_day: int,
        tokens_per_request: int = 100
    ) -> Dict[str, float]:
        """Calculate cost per model."""
        print(f"\n🔢 Calculating costs for: {model_name}")
        print(f"   Requests/day: {requests_per_day:,}")
        print(f"   Tokens/request: {tokens_per_request}")

        # Simulate cost calculation
        cost_per_1k_tokens = 0.002  # $
        total_tokens_daily = requests_per_day * tokens_per_request
        daily_cost = (total_tokens_daily / 1000) * cost_per_1k_tokens

        costs = {
            "daily_cost_usd": round(daily_cost, 4),
            "monthly_cost_usd": round(daily_cost * 30, 2),
            "cost_per_request": round(daily_cost / requests_per_day, 6),
            "tokens_per_day": total_tokens_daily
        }

        print(f"   Daily: ${costs['daily_cost_usd']:.4f}")
        print(f"   Monthly: ${costs['monthly_cost_usd']:.2f}")
        print(f"   Per request: ${costs['cost_per_request']:.6f}")

        return costs


def demo():
    """Demonstrate cost optimization."""
    print("=" * 60)
    print("Cost Optimization Framework Demo")
    print("=" * 60)

    optimizer = CostOptimizer(cloud_provider="aws", region="us-east-1")

    # Analyze costs
    print(f"\n{'='*60}")
    print("Cost Analysis")
    print(f"{'='*60}")

    analysis = optimizer.analyze_costs(time_range="last_30_days")

    # Get recommendations
    print(f"\n{'='*60}")
    print("Recommendations")
    print(f"{'='*60}")

    recommendations = optimizer.get_recommendations()

    # Enable spot instances
    print(f"\n{'='*60}")
    print("Spot Instance Configuration")
    print(f"{'='*60}")

    spot_config = optimizer.enable_spot_instances(
        max_price_per_hour=2.50,
        fallback_on_demand=True
    )

    # Forecast
    print(f"\n{'='*60}")
    print("Cost Forecast")
    print(f"{'='*60}")

    forecast = optimizer.forecast_costs(months_ahead=3, growth_rate=0.1)

    # Budget alert
    print(f"\n{'='*60}")
    print("Budget Alert")
    print(f"{'='*60}")

    alert = optimizer.set_budget_alert(monthly_budget=35000, alert_threshold=0.8)

    # Per-model costs
    print(f"\n{'='*60}")
    print("Per-Model Cost Analysis")
    print(f"{'='*60}")

    model_costs = optimizer.calculate_per_model_cost(
        model_name="llama2-7b",
        requests_per_day=10000,
        tokens_per_request=150
    )


if __name__ == "__main__":
    demo()
