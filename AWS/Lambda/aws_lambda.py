"""
AWS Lambda
==========

Serverless functions on AWS.

Author: Brill Consulting
"""

from typing import Dict, List
from datetime import datetime


class AWSLambda:
    """AWS Lambda management."""

    def __init__(self, region: str = "us-east-1"):
        self.region = region
        self.functions = {}

    def create_function(self, name: str, runtime: str = "python3.11",
                       handler: str = "lambda_function.lambda_handler") -> Dict:
        """Create Lambda function."""
        print(f"\n⚡ Creating function: {name}")
        print(f"   Runtime: {runtime}")

        function = {
            "function_name": name,
            "function_arn": f"arn:aws:lambda:{self.region}:123456789012:function:{name}",
            "runtime": runtime,
            "handler": handler,
            "role": f"arn:aws:iam::123456789012:role/{name}-role",
            "state": "Active",
            "invocations": 0
        }

        self.functions[name] = function
        print(f"✓ Function created: {function['function_arn']}")

        return function

    def invoke(self, name: str, payload: Dict) -> Dict:
        """Invoke Lambda function."""
        if name not in self.functions:
            return {"error": f"Function {name} not found"}

        print(f"\n🚀 Invoking: {name}")
        print(f"   Payload: {payload}")

        self.functions[name]["invocations"] += 1

        result = {
            "statusCode": 200,
            "body": {"message": "Success", "payload": payload},
            "executed_at": datetime.now().isoformat()
        }

        print(f"✓ Function executed")
        return result

    def add_trigger(self, function_name: str, trigger_type: str, source: str) -> Dict:
        """Add trigger to function."""
        print(f"\n⚡ Adding {trigger_type} trigger to {function_name}")

        trigger = {
            "type": trigger_type,
            "source": source,
            "enabled": True
        }

        print(f"✓ Trigger added")
        return trigger

    def get_summary(self) -> Dict:
        """Get Lambda summary."""
        total_invocations = sum(f["invocations"] for f in self.functions.values())

        return {
            "region": self.region,
            "functions": len(self.functions),
            "total_invocations": total_invocations
        }


def demo():
    """Demo AWS Lambda."""
    print("AWS Lambda Demo")
    print("=" * 60)

    lambda_service = AWSLambda("us-east-1")

    func = lambda_service.create_function("process-order", "python3.11")
    lambda_service.invoke("process-order", {"order_id": "12345"})

    lambda_service.add_trigger("process-order", "API Gateway", "/orders")

    print("\n📊 Summary:")
    summary = lambda_service.get_summary()
    for k, v in summary.items():
        print(f"  {k}: {v}")

    print("\n✓ AWS Lambda Demo Complete!")


if __name__ == '__main__':
    demo()
