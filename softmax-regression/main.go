package main

import (r "mlbase/softmax-regression/refinement"
		m "mlbase/matrix"
		"math/rand/v2"
		"fmt"
	)

// This function will be used to predict when Weigts be confirmed.
func ApplyWeights(Bios []float64, W, X [][]float64) {
	z := m.Multipliction(X, W)
	for i := range z {
		z[i] = m.RowToRowAddition(z[i], Bios)
	}
	z = Softmax(z)
	return a
}

func GenerateRandomWeights(rows, columns int) [][]float64 {
	W := m.GenerateMatrix(rows, columns)
	for i := range W {
		for j := range W[i] {
			W[i][j] = rand.Float64()
		}
	}
	return W
}

func main() {
	X, _ := m.ReadData("dataset/softmax-regression-X.csv")
	Labels, _ := m.ReadData("dataset/softmax-regression-Labels.csv")
	W := GenerateRandomWeights(len(X[0]), len(Labels[0]))
	Bios := make([]float64, len(Labels[0]))
	lrate := 0.01

	fmt.Printf("Starting: %v\n", W)
	
	for i:=0; i<1500; i++ {
		W, Bios = r.RefineWeights(lrate, Bios, X, W, Labels)
	}
	fmt.Printf("Final: %v\n", W)
	fmt.Printf("Bios: %v\n", Bios)
}
