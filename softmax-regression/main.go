package main

import (r "mlbase/softmax-regression/refinement"
		m "mlbase/matrix"
		"math/rand/v2"
		"fmt"
	)

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
		W = r.RefineWeights(lrate, Bios, X, W, Labels)
	}
	fmt.Printf("Final: %v\n", W)
}
