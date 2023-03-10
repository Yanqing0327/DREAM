from query_strategies import RandomSampling, KMeansSampling

def get_strategy(name):
    if name == "RandomSampling":
        return RandomSampling
    elif name == "KMeansSampling":
        return KMeansSampling
    else:
        raise NotImplementedError