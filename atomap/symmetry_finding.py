"""Various functions for finding lattice symmetry and sorting vectors.

This module contains functions related to sorting and finding unique
vectors.
Used to find unique crystallographic directions, after finding the
translation symmetries from a list of (x, y) positions.
"""
import math
import copy


def _remove_parallel_vectors(vector_list, distance_tolerance):
    """Remove parallel and antiparallel vectors from a list.

    Prefers having a positive long axis.

    Parameters
    ----------
    vector_list : list of tuples
        A list containing the vectors, in the form [(x0, y0), (x1, y1), ...].
    distance_tolerance : float
        Must be positive non-zero number.

    Returns
    -------
    new_vector_list : list of tuples

    Examples
    --------
    Remove vectors which have the same direction, but different length

    >>> import atomap.symmetry_finding as sf
    >>> vector_list = [(10, 10), (20, 20)]
    >>> sf._remove_parallel_vectors(vector_list, distance_tolerance=2)
    [(10, 10)]

    Remove antiparallel vectors

    >>> vector_list = [(10, 10), (-10, -10)]
    >>> sf._remove_parallel_vectors(vector_list, 2)
    [(10, 10)]

    Two unique vectors

    >>> vector_list = [(10, 10), (-10, -10), (-20, -20), (10, 0), (-10, 0)]
    >>> sf._remove_parallel_vectors(vector_list, 2)
    [(10, 0), (10, 10)]

    """
    # This function works by iterating over all the vectors, starting with the
    # shortest one, and comparing each vector to all the other vectors.
    # If a vector is either parallel or antiparallel to another vector,
    # it is added to the remove_vector_list.
    vector_list = _sort_vectors_by_length(vector_list)
    vector_list = _remove_duplicate_vectors(vector_list, distance_tolerance)
    remove_vector_list = []
    for zone_index, zv in enumerate(vector_list):
        for n in range(-4, 5):
            # To find the vectors which are pointing the same or opposite
            # direction, but with different length, the n_vector is
            # made and iterated between -4 and 5.
            n_vector = (n * zv[0], n * zv[1])
            len_vector = math.hypot(zv[0], zv[1])
            for temp_index, temp_zv in enumerate(vector_list[zone_index + 1 :]):
                dist_x = temp_zv[0] - n_vector[0]
                dist_y = temp_zv[1] - n_vector[1]
                distance = math.hypot(dist_x, dist_y)
                if distance < distance_tolerance:
                    # If the distance from the zv, and the
                    # temp_zv is within the shortest_vector/tolerance,
                    # further checks are done.
                    # If the vectors do not have the same distance,
                    # temp_zv is added to remove_vector_list.
                    # If they have the same length, the principal vector
                    # component (the longest one in zv) is used
                    # to determine which vector is added to
                    # remove_vector_list.
                    len_temp_vector = math.hypot(temp_zv[0], temp_zv[1])
                    if abs(len_vector - len_temp_vector) < len_vector / 10:
                        if abs(zv[0]) >= abs(zv[1]):
                            long_vector = 0
                        else:
                            long_vector = 1
                        if zv[long_vector] > temp_zv[long_vector]:
                            remove_vector_list.append(temp_zv)
                        else:
                            remove_vector_list.append(zv)
                    else:
                        remove_vector_list.append(temp_zv)

    new_vector_list = []
    for vector in vector_list:
        if vector not in remove_vector_list:
            new_vector_list.append(vector)

    return new_vector_list


def _remove_duplicate_vectors(vector_list, distance_tolerance):
    """Remove duplicate vectors from a list of vectors.

    If two vectors in the list has the same direction and length,
    within the distance_tolerance, the one will be removed.

    The returned list will be sorted as a function to length,
    meaning the shortest vector will be first.

    Parameters
    ----------
    vector_list : list of tuples
    distance_tolerance : float
        Positive non-zero number.

    Returns
    -------
    new_vector_list : list of tuples

    Examples
    --------
    >>> vector_list = [(20, 10), (20, 10)]
    >>> import atomap.symmetry_finding as sf
    >>> sf._remove_duplicate_vectors(vector_list, distance_tolerance=1)
    [(20, 10)]

    Changing distance_tolerance

    >>> vector_list = [(20, 12), (20, 10)]
    >>> import atomap.symmetry_finding as sf
    >>> sf._remove_duplicate_vectors(vector_list, distance_tolerance=1)
    [(20, 10), (20, 12)]
    >>> sf._remove_duplicate_vectors(vector_list, distance_tolerance=3)
    [(20, 10)]

    """
    vector_list = _sort_vectors_by_length(vector_list)
    remove_index_list = []
    for zi0, zv0 in enumerate(vector_list):
        for zi1, zv1 in enumerate(vector_list[zi0 + 1 :]):
            distance = math.hypot(zv1[0] - zv0[0], zv1[1] - zv0[1])
            if distance < distance_tolerance:
                remove_index_list.append(zi0 + zi1 + 1)
    new_vector_list = []
    for index, vector in enumerate(vector_list):
        if index not in remove_index_list:
            new_vector_list.append(vector)
    return new_vector_list


def _sort_vectors_by_length(vector_list):
    """Sorts a list of vectors as function of length.

    Parameters
    ----------
    vector_list : list of tuples
        A list containing the vectors, in the form [(x0, y0), (x1, y1), ...]

    Returns
    -------
    new_vector_list : list of tuples

    Examples
    --------
    >>> vector_list = [(20, 10), (0, 10), (10, -10)]
    >>> import atomap.symmetry_finding as sf
    >>> sf._sort_vectors_by_length(vector_list)
    [(0, 10), (10, -10), (20, 10)]

    """
    new_vector_list = copy.deepcopy(vector_list)
    zone_vector_distance_list = []
    for vector in new_vector_list:
        distance = math.hypot(vector[0], vector[1])
        zone_vector_distance_list.append(distance)

    new_vector_list.sort(key=dict(zip(new_vector_list, zone_vector_distance_list)).get)
    return new_vector_list
