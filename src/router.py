import re


def _has(question, words):
    # whole-word match so "who" doesn't fire on "whole", "feel" on "feeling" is fine via stems
    return any(re.search(rf"\b{w}", question) for w in words)


def route_question(question):
    q = question.lower()

    if re.search(r"\bwho\b", q) or _has(q, ["guideline", "recommend"]):
        return "who"
    if _has(q, ["medicine", "drug", "tablet", "treatment", "dose", "dosage"]):
        return "drug"
    if _has(q, ["symptom", "sign", "feel"]):
        return "symptom"
    return "medical"
