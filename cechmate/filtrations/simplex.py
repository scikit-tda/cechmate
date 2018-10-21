class Simplex:
    """
    A class to help sort simplices.  Sorted by distance, and sorted by order
    if tied


    TODO: Why not just use something like key=lambda x: (x.dist, x.idxs) ?


    """

    def __init__(self, idxs, dist):
        self.idxs = idxs
        self.dist = dist

    def __eq__(self, other):
        return (self.dist == other.dist) and (len(self.idxs) == len(other.idxs))

    def __ne__(self, other):
        return not (self.dist == other.dist) or not (len(self.idxs) == len(other.idxs))

    def __lt__(self, other):
        if self.dist < other.dist:
            return True
        elif self.dist == other.dist:
            if len(self.idxs) < len(other.idxs):
                return True
        return False

    def __le__(self, other):
        return self.__eq__(other) or self.__lt__(other)

    def __gt__(self, other):
        if self.dist > other.dist:
            return True
        elif self.dist == other.dist:
            if len(self.idxs) > len(other.idxs):
                return True
        return False

    def __ge__(self, other):
        return self.__eq__(other) or self.__gt__(other)

    def __repr__(self):
        return "%s %s" % (self.idxs, self.dist)
