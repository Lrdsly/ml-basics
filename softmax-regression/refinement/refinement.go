package refinement

import (m "mlbase/matrix")

// ---- SOFTMAX REGRESSION ----

func Softmax(x [][]float64) [][]float64 {
	result := m.MapElements(x, m.ElementWiseExp)
	row_sum := m.RowSum(result)
	for i := range result {
		result[i] = m.RowScalerMultipliction(result[i], 1.0/row_sum[i][0])
	} 
	return result
}

func RefineWeights(lrate float64, Bios []float64, X,W,Labels [][]float64) ([][]float64, []float64){
	z := m.Multipliction(X, W)
	for i := range z {
		z[i] = m.RowToRowAddition(z[i], Bios)
	}
	z = Softmax(z)

	diff := m.Addition(z, m.ScalerMultipliction(-1, Labels))
	scaler := 1.0/float64(len(X)) // 1/N
	// gradientW = X^T * (Zsoftmax - Labels)   |  W = W - (lrate * gradientW)
	gradientW := m.Multipliction(m.Transpose(X), diff)
	gradientW = m.ScalerMultipliction(scaler, gradientW) // 1/N

	gradientB := m.ColumnSum(diff)

	new_W := m.Addition(W, m.ScalerMultipliction(lrate, m.ScalerMultipliction(-1, gradientW)))
	// Bios is a row matrix and we can consider it as a vector using index 0
	new_Bios := Bios
	for i := range new_Bios {
		new_Bios[i] += ((-lrate * scaler) * gradientB[0][i])
	}

	return new_W, new_Bios
}
