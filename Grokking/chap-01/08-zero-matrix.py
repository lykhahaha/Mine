import unittest

def zero_out_mat(mat):
    rows = len(mat)
    cols = len(mat[0])

    zero_row_idx, zero_col_idx = [], []

    for row in range(rows):
        for col in range(cols):
            if mat[row][col] == 0:
                zero_row_idx.append(row)
                zero_col_idx.append(col)
                break

    for row_idx, col_idx in zip(zero_row_idx, zero_col_idx):
        mat[row_idx][:] = [0]*cols
        for row in range(rows):
            mat[row][col_idx] = 0

class Test(unittest.TestCase):
    def test_zero_out(self):
        mat1 = [[1,1,1,1,1],[1,0,1,1,1],[1,1,1,1,1],[1,1,1,0,1],[2,3,4,5,6]]
        mat2 = [[1,0,1,0,1],[0,0,0,0,0],[1,0,1,0,1],[0,0,0,0,0],[2,0,4,0,6]]
        zero_out_mat(mat1)
        self.assertEqual(mat1, mat2)

if __name__ == '__main__':
    unittest.main()