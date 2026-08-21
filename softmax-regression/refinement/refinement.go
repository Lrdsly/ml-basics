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

func RefiningWeights(lrate float64, Bios []float64, X,W,Labels [][]float64) [][]float64 {
	z := m.Multipliction(X, W)
	for i := range z {
		z[i] = m.RowToRowAddition(z[i], Bios)
	}
	z = Softmax(z)

	// gradientW = X^T * (Zsoftmax - Labels)   |  W = W - (lrate * gradientW)
	gradientW := m.Multipliction(m.Transpose(X), (m.Addition(z, m.ScalerMultipliction(-1, Labels))))
	W = m.Addition(W, m.ScalerMultipliction(lrate, m.ScalerMultipliction(-1, gradientW)))
	return W
}
