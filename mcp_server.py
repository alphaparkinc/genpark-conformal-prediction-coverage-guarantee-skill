"""
MCP Server implementation for Conformal Prediction Coverage Guarantee Skill.
Exposes tools to calibrate conformity scores and generate prediction sets.
"""

import json
import sys
from client import ConformalPredictor

PREDICTOR = ConformalPredictor(alpha=0.10)


def handle_request(req: dict) -> dict:
    method = req.get("method")
    params = req.get("params", {})

    if method == "tools/list":
        return {
            "tools": [
                {
                    "name": "calibrate_conformal",
                    "description": "Calibrate non-conformity threshold from calibration set",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "true_labels": {"type": "array", "items": {"type": "string"}},
                            "prob_distributions": {"type": "array", "items": {"type": "object"}},
                            "alpha": {"type": "number", "default": 0.10}
                        },
                        "required": ["true_labels", "prob_distributions"]
                    }
                },
                {
                    "name": "predict_set",
                    "description": "Generate prediction set with guaranteed coverage",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "prob_distribution": {"type": "object"}
                        },
                        "required": ["prob_distribution"]
                    }
                }
            ]
        }
    elif method == "tools/call":
        tool_name = params.get("name")
        args = params.get("arguments", {})

        if tool_name == "calibrate_conformal":
            alpha = args.get("alpha", 0.10)
            PREDICTOR.alpha = alpha
            cutoff = PREDICTOR.calibrate(args["true_labels"], args["prob_distributions"])
            return {"content": [{"type": "text", "text": json.dumps({"status": "calibrated", "quantile": cutoff, "alpha": alpha})}]}

        elif tool_name == "predict_set":
            res = PREDICTOR.predict_set(args["prob_distribution"])
            return {"content": [{"type": "text", "text": json.dumps(res)}]}

        return {"error": f"Unknown tool: {tool_name}"}

    return {"error": f"Unknown method: {method}"}


def main():
    for line in sys.stdin:
        if not line.strip():
            continue
        try:
            req = json.loads(line)
            resp = handle_request(req)
            resp["id"] = req.get("id")
            sys.stdout.write(json.dumps(resp) + "\n")
            sys.stdout.flush()
        except Exception as e:
            sys.stdout.write(json.dumps({"error": str(e)}) + "\n")
            sys.stdout.flush()


if __name__ == "__main__":
    main()
