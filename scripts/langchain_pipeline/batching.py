def format_batch(triplets: list[tuple[str, str, str]]) -> str:
    return "\n".join(f"{i}: ({s}, {p}, {o})" for i, (s, p, o) in enumerate(triplets))

def make_batches(triplets: list[tuple[str, str, str]], size: int):
    return [triplets[i:i + size] for i in range(0, len(triplets), size)]
