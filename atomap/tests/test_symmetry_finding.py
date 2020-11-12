import atomap.symmetry_finding as sf


class TestSortVectorsByLength:
    def test_y_direction(self):
        vector_list = [(0, 3), (0, 1), (0, 2), (0, 100)]
        new_vector_list = sf._sort_vectors_by_length(vector_list)
        assert new_vector_list[0] == (0, 1)
        assert new_vector_list[1] == (0, 2)
        assert new_vector_list[2] == (0, 3)
        assert new_vector_list[3] == (0, 100)

    def test_x_direction(self):
        vector_list = [(3, 0), (1, 0), (2, 0), (100, 0)]
        new_vector_list = sf._sort_vectors_by_length(vector_list)
        assert new_vector_list[0] == (1, 0)
        assert new_vector_list[1] == (2, 0)
        assert new_vector_list[2] == (3, 0)
        assert new_vector_list[3] == (100, 0)

    def test_negative(self):
        vector_list = [(-3, 0), (-1, 0), (-2, 0), (-100, 0)]
        new_vector_list = sf._sort_vectors_by_length(vector_list)
        assert new_vector_list[0] == (-1, 0)
        assert new_vector_list[1] == (-2, 0)
        assert new_vector_list[2] == (-3, 0)
        assert new_vector_list[3] == (-100, 0)

    def test_negative_and_positive(self):
        vector_list = [(-3, 0), (1, 0), (2, 0), (-100, 0)]
        new_vector_list = sf._sort_vectors_by_length(vector_list)
        assert new_vector_list[0] == (1, 0)
        assert new_vector_list[1] == (2, 0)
        assert new_vector_list[2] == (-3, 0)
        assert new_vector_list[3] == (-100, 0)

    def test_xy_negative_positive(self):
        vector_list = [(10, -10), (10, 11), (20, -30), (-300, 200)]
        new_vector_list = sf._sort_vectors_by_length(vector_list)
        assert new_vector_list[0] == (10, -10)
        assert new_vector_list[1] == (10, 11)
        assert new_vector_list[2] == (20, -30)
        assert new_vector_list[3] == (-300, 200)


class TestRemoveDuplicateVectors:
    def test_two_identical_vectors(self):
        vector_list = [(10, 0), (10, 0)]
        new_vector_list = sf._remove_duplicate_vectors(vector_list, 1)
        assert len(new_vector_list) == 1
        assert new_vector_list[0] == (10, 0)

    def test_two_similar_vectors_distance_tolerance(self):
        vector_list = [(11.0, 0), (10, 0)]
        new_vector_list = sf._remove_duplicate_vectors(vector_list, 2)
        assert len(new_vector_list) == 1
        assert new_vector_list[0] == (10, 0)

        new_vector_list = sf._remove_duplicate_vectors(vector_list, 0.5)
        assert len(new_vector_list) == 2
        assert new_vector_list[0] == (10, 0)
        assert new_vector_list[1] == (11, 0)

    def test_two_antiparallel_vectors(self):
        vector_list = [(10, 0), (-10, 0)]
        new_vector_list = sf._remove_duplicate_vectors(vector_list, 1)
        assert len(new_vector_list) == 2

    def test_two_different_vectors(self):
        vector_list = [(10, 0), (-10, 0)]
        new_vector_list = sf._remove_duplicate_vectors(vector_list, 1)
        assert len(new_vector_list) == 2


class TestRemoveParallelVectors:
    def test_parallel_x(self):
        vector_list = [(10, 0), (20, 0)]
        new_vector_list = sf._remove_parallel_vectors(vector_list, 2)
        assert len(new_vector_list) == 1
        assert new_vector_list[0] == (10, 0)

    def test_parallel_y(self):
        vector_list = [(0, 10), (0, 20)]
        new_vector_list = sf._remove_parallel_vectors(vector_list, 2)
        assert len(new_vector_list) == 1
        assert new_vector_list[0] == (0, 10)

    def test_antiparallel_x(self):
        vector_list = [(10, 0), (-10, 0)]
        new_vector_list = sf._remove_parallel_vectors(vector_list, 2)
        assert len(new_vector_list) == 1
        assert new_vector_list[0] == (10, 0)

    def test_antiparallel_y(self):
        vector_list = [(0, 10), (0, -10)]
        new_vector_list = sf._remove_parallel_vectors(vector_list, 2)
        assert len(new_vector_list) == 1
        assert new_vector_list[0] == (0, 10)

    def test_antiparallel_get_positive_x(self):
        vector_list = [(-10, 0), (10, 0), (-20, 0), (30, 0)]
        new_vector_list = sf._remove_parallel_vectors(vector_list, 2)
        assert len(new_vector_list) == 1
        assert new_vector_list[0] == (10, 0)

    def test_antiparallel_get_positive_y(self):
        vector_list = [(0, -10), (0, 10), (0, -20), (0, 30)]
        new_vector_list = sf._remove_parallel_vectors(vector_list, 2)
        assert len(new_vector_list) == 1
        assert new_vector_list[0] == (0, 10)

    def test_parallel_xy(self):
        vector_list = [(10, 10), (20, 20)]
        new_vector_list = sf._remove_parallel_vectors(vector_list, 2)
        assert len(new_vector_list) == 1
        assert new_vector_list[0] == (10, 10)

    def test_antiparallel_xy(self):
        vector_list = [(-10, -10), (10, 10)]
        new_vector_list = sf._remove_parallel_vectors(vector_list, 2)
        assert len(new_vector_list) == 1
        assert new_vector_list[0] == (10, 10)

    def test_only_identical(self):
        vector_list = [(10, 10), (10, 10)]
        new_vector_list = sf._remove_parallel_vectors(vector_list, 1)
        assert len(new_vector_list) == 1
        assert new_vector_list[0] == (10, 10)

    def test_identical(self):
        vector_list = [(10, 10), (10, 10), (-10, -10), (20, 10)]
        new_vector_list = sf._remove_parallel_vectors(vector_list, 1)
        assert len(new_vector_list) == 2
        assert new_vector_list[0] == (10, 10)
        assert new_vector_list[1] == (20, 10)

    def test_distance_tolerance(self):
        vector_list0 = [(10, 0), (10, 1)]
        new_vector_list0 = sf._remove_parallel_vectors(vector_list0, 0.9)
        assert len(new_vector_list0) == 2

        vector_list1 = [(10, 0), (10, 1)]
        new_vector_list1 = sf._remove_parallel_vectors(vector_list1, 1.1)
        assert len(new_vector_list1) == 1

        vector_list2 = [(10, 20), (20, 20)]
        new_vector_list2 = sf._remove_parallel_vectors(vector_list2, 1.1)
        assert len(new_vector_list2) == 2

        vector_list3 = [(10, 20), (20, 20)]
        new_vector_list3 = sf._remove_parallel_vectors(vector_list3, 10.2)
        assert len(new_vector_list3) == 1
