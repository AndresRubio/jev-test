"""Requests shaped like TypeSafe's documented /v1/systemone examples."""

JEV_DOC_EXAMPLE = {
    "model": "jev-latest",
    "state": {"body": "billed twice, refund please or we cancel"},
    "questions": {
        "dept": {
            "type": "choice",
            "instructions": "which team?",
            "criteria": {"billing": "refunds", "tech": "bugs"},
        },
        "refund": {
            "type": "noul",
            "instructions": "Does the customer ask for money back?",
            "criteria": {"true": "asks for a refund", "false": "anything else"},
        },
        "urgency": {
            "type": "score",
            "instructions": "How urgent is this?",
            "criteria": ["not urgent", "soon", "blocking"],
        },
    },
}
