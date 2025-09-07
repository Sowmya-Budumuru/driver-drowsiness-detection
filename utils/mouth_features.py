from scipy.spatial import distance

def lip_aspect_ratio(mouth):
    A = distance.euclidean(mouth[13], mouth[19])  # vertical
    B = distance.euclidean(mouth[14], mouth[18])
    C = distance.euclidean(mouth[15], mouth[17])
    avg_vertical = (A + B + C) / 3.0
    horizontal = distance.euclidean(mouth[12], mouth[16])
    lar = avg_vertical / horizontal
    return lar
