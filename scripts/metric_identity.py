"""Preserve the first occurrence of each valid official connectivity identity."""


def unique_metric_order(order, identities):
    seen, result = set(), []
    for index in order:
        key = identities[index]
        valid = isinstance(key, str) and len(key) == 14 and key.isascii() and key.isalpha() and key.isupper()
        if valid:
            if key in seen:
                continue
            seen.add(key)
        result.append(int(index))
    return result
