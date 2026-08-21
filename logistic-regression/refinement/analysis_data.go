package refinement

import ("os"
		"strconv"
		csv "encoding/csv")

func ReadData(file_path string) ([][]float64, error) {
	file, err := os.Open(file_path)
	if err != nil {
		return nil, err
	}

	defer file.Close()

	reader := csv.NewReader(file)
	data, r_err := reader.ReadAll()
	if r_err != nil {
		return nil, r_err	
	}

	// convert data types from sting to float64
	result := [][]float64{}
	for i := range data {
		result = append(result, []float64{})
		for j := range data[i] {
			result[i] = append(result[i], 0)
			result[i][j], _ = strconv.ParseFloat(data[i][j], 64)
		}
	}
	return result, nil
}
