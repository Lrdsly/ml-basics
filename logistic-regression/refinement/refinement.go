package refinement

import (
		"math"
		matrix "mlbase/matrix")

// ---- REFINEMENT ----

func Sigmoid(z float64) float64 {
	return 1 / (1 + math.Exp(-z))
}
// W is always the second parameter of multipliction and we consider it as a column matrix
// every gradient objects must be similar with W
func GradientDescentStep(lrate, bios float64,Yr, X, W [][]float64) ([][]float64, float64) {
	rows := len(X)
	multipliction := matrix.Multipliction(X, W)
	
	gradient_sumation := matrix.GenerateMatrix(len(W), 1) // sumation of all gradients divided by their count
	bios_sumation := bios

	for i, data_row := range X {
		gradient := [][]float64{}
		Y := Sigmoid(bios + (multipliction[i][0])) // sigmond((X * W) + b)
		// Calculate bios
		bios_sumation = bios_sumation + (Y - Yr[i][0])

		// Calculate gradient
		for _, value := range data_row {
			gradient = append(gradient, []float64{((Y - Yr[i][0])) * (value)}) 
		}
		gradient_sumation = matrix.Addition(gradient_sumation, gradient)
	}

	gradient_sumation = matrix.ScalerMultipliction(1.0/float64(rows), gradient_sumation)
	bios_sumation = bios_sumation / float64(rows)

	new_W := matrix.Addition(W, matrix.ScalerMultipliction(-lrate, gradient_sumation))
	new_bios := bios - (lrate * bios_sumation)

	return new_W, new_bios
}
