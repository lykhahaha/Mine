import unittest

def rotate_matrix_in_place(mat):
    n = len(mat)
    for row in range(n//2):
        for col in range(row, n-1-row):
            tmp_1 = mat[col][n-1-row]

            mat[col][n-1-row] = mat[row][col]

            tmp_2 = mat[n-1-row][n-1-col]

            mat[n-1-row][n-1-col] = tmp_1

            tmp_1 = mat[n-1-col][row]

            mat[n-1-col][row] = tmp_2

            mat[row][col] = tmp_1

class Test(unittest.TestCase):
    def test_rotate_matrix_in_place(self):
        mat1 = [[1,2],[3,4]]
        mat2 = [[3,1],[4,2]]
        rotate_matrix_in_place(mat1)
        self.assertEqual(mat1, mat2)
        mat5 = [[1,2,3,4],[5,6,7,8],[9,10,11,12],[13,14,15,16]]
        mat6 = [[13, 9, 5, 1], [14, 10, 6, 2], [15, 11, 7, 3], [16, 12, 8, 4]]
        rotate_matrix_in_place(mat5)
        self.assertEqual(mat5, mat6)

if __name__ == '__main__':
    unittest.main()