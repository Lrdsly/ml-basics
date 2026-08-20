package main

import ("fmt"
		"math/rand/v2"
		m "mlbase/matrix"
		r "mlbase/refinement")


func GenerateRandomWeights(featureCount int) [][]float64 {
	weights := make([][]float64, featureCount)
	
	for i := 0; i < featureCount; i++ {
		val := rand.Float64()
		weights[i] = []float64{val}
	}
	
	return weights
}

func main() {
	data, _ := r.ReadData("dataset/dataset.csv")
	X, Yr, _ := m.SplitColumns(1, data)
	W := GenerateRandomWeights(len(X[0]))
	b := 0.0
	fmt.Printf("Starting: %v\n", W)
	for i:=0; i<1000; i++ {
		W, b = r.GradientDescentStep(0.1, b, Yr, X, W)
	}
	fmt.Printf("Final: %v\n", W)
	fmt.Printf("Bios: %f\n", b)
}
