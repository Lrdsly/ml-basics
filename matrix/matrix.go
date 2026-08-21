package matrix

import ("errors"
		"math")

type MatrixFunc func(float64) float64

func GenerateMatrix(rows, columns int) [][]float64{
	result := [][]float64{}
	for i:=0; i<rows; i++ {
		result = append(result, []float64{})
	}
	for i := range result {
		for count:=0; count<columns; count ++ {
			result[i] = append(result[i], 0.0)
		}
	}

	return result
}

func ScalerMultipliction(s float64, x [][]float64) [][]float64 {
	rows, columns := len(x), len(x[0])
	result := GenerateMatrix(rows, columns)

	for i := range x {
		for j := range x[i] {
			result[i][j] = x[i][j] * s
		}
	}

	return result
}

func Multipliction(x, y [][]float64) ([][]float64)  {
	row_x, row_y := len(x), len(y)
	column_x, column_y := len(x[0]), len(y[0])
	if column_x != row_y {
		return nil
	}

	// define final matrix values position
	result := GenerateMatrix(row_x, column_y)

	for i, xrow := range x {
		for m:=0; m < column_y; m++ {
			sum := 0.0
			for j, value := range xrow {
				sum = sum + (value * y[j][m])
			}
			result[i][m] = sum
		}		
	}	

	return result
}

func ScalerAddition(s float64, x [][]float64) [][]float64{
	rows, columns := len(x), len(x[0])
	result := GenerateMatrix(rows, columns)

	for i := range x {
		for j := range x[i] {
			result[i][j] = result[i][j] + s
		}
	}

	return result
}

func Addition(x, y [][]float64) ([][]float64) {
	row_x, row_y := len(x), len(y)
	column_x, column_y := len(x[0]), len(y[0])
	if row_x != row_y || column_x != column_y {
		return nil
	}

	// row_x = row_y (column_x = column_y) as both are refering to final matrix rows(columns) count
	result := GenerateMatrix(row_x, column_x)

	for i, _ := range x {
		for j, value := range x[i] {
			result[i][j] = y[i][j] + value
		}
	}

	return result
}

func Transpose(x [][]float64) [][]float64 {
	result := GenerateMatrix(len(x[0]), len(x))
	for i := range x{
		for j := range x[i] {
			result[j][i] = x[i][j]
		}
	}
	return result
}

func SplitColumns(right_columns int, x [][]float64) ([][]float64, [][]float64, error) {
	rows, columns := len(x), len(x[0])
	if right_columns > columns {
		return nil, nil, errors.New("Not a valid spliting")
	}

	left_matrix := GenerateMatrix(rows, columns-right_columns)
	right_matrix := GenerateMatrix(rows, right_columns)

	// assign left matrix
	for i := range left_matrix {
		for j := range left_matrix[i] {
			left_matrix[i][j] = x[i][j]
		}
	}
	// assign right matrix
	for i := range right_matrix {
		for j := range right_matrix[i] {
			right_matrix[i][j] = x[i][columns - right_columns + j]
		}
	}

	return left_matrix, right_matrix, nil
}

func MapElements(x [][]float64, f MatrixFunc) [][]float64 {
	final_matrix := GenerateMatrix(len(x), len(x[0]))
	for i := range x {
		for j:= range x[i] {
			final_matrix[i][j] = f(x[i][j])
		}
	}
	return final_matrix
}

// ---- softmax regression methods ----

func RowSum(x [][]float64) [][]float64 {
	result := GenerateMatrix(len(x), 1)
	for i := range x{
		for j := range x[i] {
			result[i][0] += x[i][j]
		}
	}
	return result
}

func ElementWiseExp(value float64) float64 {
	return math.Exp(value)
}

func RowScalerMultipliction(row []float64, scaler float64) []float64{
	result := make([]float64, len(row))
	for i := range row {
		result[i] = row[i] * scaler
	}
	return result
}

func RowToRowAddition(row1, row2 []float64) []float64 {
	if len(row1) != len(row2) {return nil}

	result := make([]float64, len(row1))
	for i := range row1 {
		result[i] = row1[i] + row2[i]
	}
	return result
}
