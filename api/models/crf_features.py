def token_features(tokens: list[str], i: int) -> dict:
    w = tokens[i]
    f = {"w.lower": w.lower(), "is_noun_like": w.isalpha(), "bias": 1.0}
    if i > 0:
        f["prev"] = tokens[i - 1].lower()
    if i < len(tokens) - 1:
        f["next"] = tokens[i + 1].lower()
    return f
