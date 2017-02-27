from scipy import spatial


def calculate_cosine_similarity(v1, v2):
    try:
        cosine = 1 - spatial.distance.cosine(v1, v2)
    except ValueError:
        cosine = 0
    finally:
        pass
    return cosine
