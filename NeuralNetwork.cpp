#include <iostream>
#include <Windows.h>
#include <vector>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <string>

using namespace std;

//функция активации на втором и третьем слоях
double ReLU(double x) {
	if (x < 0)
		return 0.01 * x;
	else if (x > 1)
		return 1 + 0.01 * (x - 1);
	return x;
}

//функция превращения вектора ответа нейронной сети в вектор вероятностей
void SoftMax(vector<double>& ans) {
	double sumexp = 0;
	for (int i = 0; i != ans.size(); i++) 
		sumexp += exp(ans[i]);

	for (int i = 0; i != ans.size(); i++) 
		ans[i] = exp(ans[i]) / sumexp;
}

//из ожидаемого значения получаем вектор из 0 и 1 на ячейке с индексом ожидаемого значения
vector<double> BoolVector(double trueNumber) {
	vector<double> newVector;
	for (int i = 0; i != 10; i++) {
		if (i == trueNumber)
			newVector.push_back(1.0);
		else
			newVector.push_back(0.0);
	}
	return newVector;
}

//кросс энтропия - получение ошибки
double CrossEntropy(vector<double>& vectorSoftMax, double trueNumber) {
	double ans;
	ans = -log(vectorSoftMax[trueNumber]);
	return ans;
}

//чтение обучающей выборки
void ReadData(vector<double>& trueNumbers, vector<vector<double>>& inputXs) {
	cout << "Чтение данных" << endl;

	ifstream data("lib_MNIST_edit.txt");
	if (data.is_open()) {
		string line;
		while (getline(data, line)) {
			// добавляем размеченное значение
			trueNumbers.push_back(stoi(line));
			vector<double> vectorOfNum;
			for (int lineAfterTrueNumber = 0; lineAfterTrueNumber != 28; lineAfterTrueNumber++) {
				getline(data, line);
				int index = 0;
				for (int i = 0; i != line.size(); i++) {
					if (line[i] == ' ') {
						vectorOfNum.push_back(stof(line.substr(index, i - index)));
						index = i + 1;
					}
				}
			}
			inputXs.push_back(vectorOfNum);
		}
	}
	data.close();
	cout << "Чтение данных прошло" << endl;
}

//чтение тестовой выборки
void ReadDataTest(vector<double>& trueNumbers, vector<vector<double>>& inputXs) {
	cout << "Чтение данных" << endl;

	ifstream data("lib_10k.txt");
	if (data.is_open()) {
		string line;
		while (getline(data, line)) {
			// добавляем размеченное значение
			trueNumbers.push_back(stoi(line));
			vector<double> vectorOfNum;
			for (int lineAfterTrueNumber = 0; lineAfterTrueNumber != 28; lineAfterTrueNumber++) {
				getline(data, line);
				int index = 0;
				for (int i = 0; i != line.size(); i++) {
					if (line[i] == ' ') {
						vectorOfNum.push_back(stof(line.substr(index, i - index)));
						index = i + 1;
					}
				}
			}
			inputXs.push_back(vectorOfNum);
		}
	}
	data.close();
	cout << "Чтение данных прошло" << endl;
}


//поэлементное вычитание
vector<double> elementalSubstraction(vector<double>& vctr1, vector<double>& vctr2) {
	if (vctr1.size() == vctr2.size()) {
		vector<double> ansVector;
		for (int i = 0; i != vctr1.size(); i++)
			ansVector.push_back(vctr1[i] - vctr2[i]);

		return ansVector;
	}
	throw "Векторы разных размерностей!";
}

void elementalSubstractionMatrix(vector<vector<double>>& matrix1, vector<vector<double>>& matrix2) {
	if (matrix1.size() == matrix2.size() && matrix1[0].size() == matrix2[0].size()) {
		for (int n = 0; n != matrix1.size(); n++) {
			for (int m = 0; m != matrix1[0].size(); m++) {
				matrix1[n][m] = matrix1[n][m] - matrix2[n][m];
			}
		}
		return;
	}
	throw "матрицы разных размерностей";
}

void elementalSubstractionVector(vector<double>& v1, vector<double>& v2) {
	if (v1.size() == v2.size()) {
		for (int i = 0; i != v1.size(); i++) {
			v1[i] = v1[i] - v2[i];
		}
		return;
	}
	throw "Векторы разных размерностей";
}


//транспонирование матрицы-строки в матрицу столбец
vector<vector<double>> Transponse(vector<double> matrix) {
	vector<vector<double>> newMatrix;

	for (int i = 0; i != matrix.size(); i++) {
		vector<double> line = { matrix[i] };
		newMatrix.push_back(line);
	}

	return newMatrix;
}

//траспонирование матрицы nxm в mxn
vector<vector<double>> Transponse(vector<vector<double>> matrix) {
	vector<vector<double>> newMatrix;

	for (int i = 0; i != matrix[0].size(); i++) {
		vector<double> line;
		for (int j = 0; j != matrix.size(); j++) {
			line.push_back(matrix[j][i]);
		}
		newMatrix.push_back(line);
	}

	return newMatrix;
}

//матричное умножение nx1 и 1xm
vector<vector<double>> matrixMultiply(vector<vector<double>>& matrix1, vector<double>& matrix2) {
	vector<vector<double>> ansVector;
	if (matrix1[0].size() == 1) {
		for (int n = 0; n != matrix1.size(); n++) {
			vector<double> line;
			for (int m = 0; m != matrix2.size(); m++) {
				line.push_back(matrix1[n][0] * matrix2[m]);
			}
			ansVector.push_back(line);
		}
		return ansVector;
	}
	throw "Невозможно произвести матричное умножение";
}

//матричное умножение 1xn и nxm
vector<double> matrixMultiply(vector<double>& matrix1, vector<vector<double>>& matrix2) {
	
	if (matrix1.size() == matrix2.size()) {
		vector<double> ansVector;
		for (int n = 0; n != matrix2[0].size(); n++) { // n != 16
			double cell = 0;
			for (int m = 0; m != matrix1.size(); m++) { // m != 10
				cell += matrix1[m] * matrix2[m][n];
			}
			ansVector.push_back(cell);
		}

		return ansVector;
	}

	throw "Невозможно выполнить матричное умножение";
}

//произведение поэлементно
vector<double> multiplyAdemara(vector<double>& vctr1, vector<double>& vctr2) {
	if (vctr1.size() == vctr2.size()){
		vector<double> ansVector;

		for (int i = 0; i != vctr1.size(); i++)
			ansVector.push_back(vctr1[i] * vctr2[i]);

		return ansVector;
	}
	throw "Невозможно выполнить поэлементное произведение";
}

//производная ReLU вектор
vector<double> ReLUDerivative(vector<double>& t2) {
	vector<double> ansVector;

	for (int i = 0; i != t2.size(); i++) {
		if ((t2[i] < 0) || (t2[i] > 1))
			ansVector.push_back(0.01);
		else
			ansVector.push_back(1);
	}
	
	return ansVector;
}

//произведение константы на матрицу
void multiplyConstMatrix(double alpha, vector<vector<double>>& matrix) {
	for (int n = 0; n != matrix.size(); n++) {
		for (int m = 0; m != matrix[0].size(); m++) {
			matrix[n][m] = matrix[n][m] * alpha;
		}
	}
}

//произведение константы на вектор
void multiplyConstVector(double LerningRate, vector<double>& v) {
	for (int i = 0; i != v.size(); i++) {
		v[i] = v[i] * LerningRate;
	}
}

//прямой проход
void DirectPassage(vector<double>& ans, vector<double>& inputX, vector<vector<double>>& WeightsFirst, vector<double>& BiasFirst, vector<vector<double>>& WeightsSecond, vector<double>& BiasSecond, vector<vector<double>>& WeightsThird, vector<double>& BiasThird, vector<double>& h1, vector<double>& h2, vector<double>& t1, vector<double>& t2) {

	//получение значения нейронов на втором слое
	for (int numberOfNeuronSecondLayer = 0; numberOfNeuronSecondLayer != 16; numberOfNeuronSecondLayer++) {
		double neuron = BiasFirst[numberOfNeuronSecondLayer];

		for (int i = 0; i != inputX.size(); i++)
			neuron += inputX[i] * WeightsFirst[i][numberOfNeuronSecondLayer];

		t1.push_back(neuron);
		h1.push_back(ReLU(neuron));
	}

	//получение значения нейронов на третьем слое
	for (int numberOfNeuronThirdLayer = 0; numberOfNeuronThirdLayer != 16; numberOfNeuronThirdLayer++) {
		double neuron = BiasSecond[numberOfNeuronThirdLayer];

		for (int i = 0; i != h1.size(); i++)
			neuron += h1[i] * WeightsSecond[i][numberOfNeuronThirdLayer];

		t2.push_back(neuron);
		h2.push_back(ReLU(neuron));
	}

	//получение ответа нейронной сети (значений на последней слое)
	for (int numberOfNeuronFinalLayer = 0; numberOfNeuronFinalLayer != 10; numberOfNeuronFinalLayer++) {
		double neuron = BiasThird[numberOfNeuronFinalLayer];

		for (int i = 0; i != h2.size(); i++)
			neuron += h2[i] * WeightsThird[i][numberOfNeuronFinalLayer];

		ans.push_back(neuron);
	}

	SoftMax(ans);
}

//метод обратного распространения ошибки
void BackPropogation(vector<double>& SoftMaxt3, vector<double>& y, vector<double>& h2, vector<vector<double>>& WeightsThird, vector<double>& t2, vector<double>& h1, vector<vector<double>>& WeightsSecond, vector<double>& t1, vector<double> inputXs, vector<vector<double>>& WeightsFirst, vector<double>& BiasFirst, vector<double>& BiasSecond, vector<double>& BiasThird) {
	//высчитываем градиенты для оптимизации весов нейросети
	
	vector<double> dE_dt3 = elementalSubstraction(SoftMaxt3, y);
	//vector<double> dE_dt3 = MSEderivative(SoftMaxt3, y);

	vector<vector<double>> h2_T = Transponse(h2);
	vector<vector<double>> dE_dw3 = matrixMultiply(h2_T, dE_dt3);

	// dE_db2 = dE_dt3

	vector<vector<double>> w3_T = Transponse(WeightsThird);
	vector<double> dE_dh2 = matrixMultiply(dE_dt3, w3_T);

	vector<double> ReLUDerivative_t2 = ReLUDerivative(t2);
	vector<double> dE_dt2 = multiplyAdemara(dE_dh2, ReLUDerivative_t2);

	vector<vector<double>> h1_T = Transponse(h1);
	vector<vector<double>> dE_dw2 = matrixMultiply(h1_T, dE_dt2);

	//dE_db2 = dE_dt2

	vector<vector<double>> w2_T = Transponse(WeightsSecond);
	vector<double> dE_dh1 = matrixMultiply(dE_dt2, w2_T);

	vector<double> ReLUDerivative_t1 = ReLUDerivative(t1);
	vector<double> dE_dt1 = multiplyAdemara(dE_dh1, ReLUDerivative_t1);

	vector<vector<double>> x_T = Transponse(inputXs);
	vector<vector<double>> dE_dw1 = matrixMultiply(x_T, dE_dt1);

	//dE_db1 = dE_dt1

	//обновление весов нейросети

	//скорость обучения
	double LearningRate = 0.0002;

	//изменение весов между первым и вторым слоем
	multiplyConstMatrix(LearningRate, dE_dw1);
	elementalSubstractionMatrix(WeightsFirst, dE_dw1);
	multiplyConstVector(LearningRate, dE_dt1);
	elementalSubstractionVector(BiasFirst, dE_dt1);

	//обновление весов между вторым и третьим слоем
	multiplyConstMatrix(LearningRate, dE_dw2);
	elementalSubstractionMatrix(WeightsSecond, dE_dw2);
	multiplyConstVector(LearningRate, dE_dt2);
	elementalSubstractionVector(BiasSecond, dE_dt2);

	//обновление весов между третьим и четвертым слоем
	multiplyConstMatrix(LearningRate, dE_dw3);
	elementalSubstractionMatrix(WeightsThird, dE_dw3);
	multiplyConstVector(LearningRate, dE_dt3);
	elementalSubstractionVector(BiasThird, dE_dt3);
}

//функция ничегонеделания)
void pass() {
	return;
}

//прямой проход тестовый
void DirectPassageTest(vector<double>& ans, vector<double>& inputX, vector<vector<double>>& WeightsFirst, vector<double>& BiasFirst, vector<vector<double>>& WeightsSecond, vector<double>& BiasSecond, vector<vector<double>>& WeightsThird, vector<double>& BiasThird, vector<double>& h1, vector<double>& h2) {

	//получение значения нейронов на втором слое
	for (int numberOfNeuronSecondLayer = 0; numberOfNeuronSecondLayer != 16; numberOfNeuronSecondLayer++) {
		double neuron = BiasFirst[numberOfNeuronSecondLayer];

		for (int i = 0; i != inputX.size(); i++)
			neuron += inputX[i] * WeightsFirst[i][numberOfNeuronSecondLayer];

		h1.push_back(ReLU(neuron));
	}

	//получение значения нейронов на третьем слое
	for (int numberOfNeuronThirdLayer = 0; numberOfNeuronThirdLayer != 16; numberOfNeuronThirdLayer++) {
		double neuron = BiasSecond[numberOfNeuronThirdLayer];

		for (int i = 0; i != h1.size(); i++)
			neuron += h1[i] * WeightsSecond[i][numberOfNeuronThirdLayer];

		h2.push_back(ReLU(neuron));
	}

	//получение ответа нейронной сети (значений на последней слое)
	for (int numberOfNeuronFinalLayer = 0; numberOfNeuronFinalLayer != 10; numberOfNeuronFinalLayer++) {
		double neuron = BiasThird[numberOfNeuronFinalLayer];

		for (int i = 0; i != h2.size(); i++)
			neuron += h2[i] * WeightsThird[i][numberOfNeuronFinalLayer];

		ans.push_back(neuron);
	}

	SoftMax(ans);
}

//проверка верный ответ или нет
double isCorrect(vector<double> y_pred, double y) {
	int indexMax = 18768172;
	double maximum = -327865456;
	for (int i = 0; i != y_pred.size(); i++) {
		if (y_pred[i] > maximum) {
			maximum = y_pred[i];
			indexMax = i;
		}
	}
	if (indexMax == y)
		return 1.0;
	return 0.0;
}

//проверка точности на тестовой выборке
void TestTrain() {
	vector<double> trueNumbers;
	vector<vector<double>> inputXs;

	vector<vector<double>> WeightsFirst;
	vector<double> BiasFirst;
	vector<vector<double>> WeightsSecond;
	vector<double> BiasSecond;
	vector<vector<double>> WeightsThird;
	vector<double> BiasThird;

	ReadDataTest(trueNumbers, inputXs);

	//считываем веса между первым и вторым слоями
	ifstream w1("WeightsFirst.txt");
	if (w1.is_open()) {
		for (int i = 0; i != 784; i++) {
			string line;
			getline(w1, line);
			vector<double> vectorOfNum;
			int index = 0;
			for (int j = 0; j != line.size(); j++) {
				if (line[j] == ' ') {
					vectorOfNum.push_back(stof(line.substr(index, j - index)));
					index = j + 1;
				}
			}
			WeightsFirst.push_back(vectorOfNum);
		}
	}
	w1.close();

	//считываем свободные веса между первым и вторым слоями
	ifstream b1("BiasFirst.txt");
	if (b1.is_open()) {
		int index = 0;
		string line;
		getline(b1, line);
		for (int i = 0; i != line.size(); i++) {
			if (line[i] == ' ') {
				BiasFirst.push_back(stof(line.substr(index, i - index)));
				index = i + 1;
			}
		}
	}
	b1.close();
	//BiasFirst = { 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 };

	//считываем веса между вторым и третьим слоями
	ifstream w2("WeightsSecond.txt");
	if (w2.is_open()) {
		for (int i = 0; i != 16; i++) {
			string line;
			getline(w2, line);
			vector<double> vectorOfNum;
			int index = 0;
			for (int j = 0; j != line.size(); j++) {
				if (line[j] == ' ') {
					vectorOfNum.push_back(stof(line.substr(index, j - index)));
					index = j + 1;
				}
			}
			WeightsSecond.push_back(vectorOfNum);
		}
	}
	w2.close();

	//считываем свободные веса между вторым и третьим слоями
	ifstream b2("BiasSecond.txt");
	if (b2.is_open()) {
		int index = 0;
		string line;
		getline(b2, line);
		for (int i = 0; i != line.size(); i++) {
			if (line[i] == ' ') {
				BiasSecond.push_back(stof(line.substr(index, i - index)));
				index = i + 1;
			}
		}
	}
	b2.close();
	//BiasSecond = { 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 };

	//считываем веса между третьим и четвертым слоями
	ifstream w3("WeightsThird.txt");
	if (w3.is_open()) {
		for (int i = 0; i != 16; i++) {
			string line;
			getline(w3, line);
			vector<double> vectorOfNum;
			int index = 0;
			for (int j = 0; j != line.size(); j++) {
				if (line[j] == ' ') {
					vectorOfNum.push_back(stof(line.substr(index, j - index)));
					index = j + 1;
				}
			}
			WeightsThird.push_back(vectorOfNum);
		}
	}
	w3.close();

	//считываем свободные веса между третьим и четвертым слоями
	ifstream b3("BiasThird.txt");
	if (b3.is_open()) {
		int index = 0;
		string line;
		getline(b3, line);
		for (int i = 0; i != line.size(); i++) {
			if (line[i] == ' ') {
				BiasThird.push_back(stof(line.substr(index, i - index)));
				index = i + 1;
			}
		}
	}
	b3.close();
	//BiasThird = { 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 };

	double correct = 0;
	for (int element = 0; element != trueNumbers.size(); element++) {
		vector<double> h1;
		vector<double> h2;

		vector<double> OutputNeuralNetwork;
		DirectPassageTest(OutputNeuralNetwork, inputXs[element], WeightsFirst, BiasFirst, WeightsSecond, BiasSecond, WeightsThird, BiasThird, h1, h2);
		correct += isCorrect(OutputNeuralNetwork, trueNumbers[element]);
	}
	cout << "Точность нейронной сети: " << 100 * correct / trueNumbers.size();
}

// обучение модели
void Train() {
	vector<double> trueNumbers;
	vector<vector<double>> inputXs;

	ReadData(trueNumbers, inputXs);

	vector<vector<double>> WeightsFirst;
	vector<double> BiasFirst;

	for (int i = 0; i != 16; i++) {
		BiasFirst.push_back(1);
	}

	for (int i = 0; i != 784; i++) {
		vector<double> Weights;
		for (int j = 0; j != 16; j++) {
			Weights.push_back((double)rand() / RAND_MAX);
		}
		WeightsFirst.push_back(Weights);
	}

	vector<vector<double>> WeightsSecond;
	vector<double> BiasSecond;

	for (int i = 0; i != 16; i++) {
		BiasSecond.push_back(1);
	}

	for (int i = 0; i != 16; i++) {
		vector<double> Weights;
		for (int j = 0; j != 16; j++) {
			Weights.push_back((double)rand() / RAND_MAX);
		}
		WeightsSecond.push_back(Weights);
	}

	vector<vector<double>> WeightsThird;
	vector<double> BiasThird;

	for (int i = 0; i != 10; i++) {
		BiasThird.push_back(1);
	}

	for (int i = 0; i != 16; i++) {
		vector<double> Weights;
		for (int j = 0; j != 10; j++) {
			Weights.push_back((double)rand() / RAND_MAX);
		}
		WeightsThird.push_back(Weights);
	}
	
	double correct = 0;
	for (int element = 0; element != trueNumbers.size(); element++){
		vector<double> h2;
		vector<double> h1;
		vector<double> t1;
		vector<double> t2;
		
		vector<double> OutputNeuralNetwork;
		DirectPassage(OutputNeuralNetwork, inputXs[element], WeightsFirst, BiasFirst, WeightsSecond, BiasSecond, WeightsThird, BiasThird, h1, h2, t1, t2);
		double error = CrossEntropy(OutputNeuralNetwork, trueNumbers[element]);
		//double error = MSE(OutputNeuralNetwork, trueNumbers[element]);


		if (element % 1000 == 0) {
			cout << "Элемент: " << element << ' ';
			cout << "Ошибка: " << error << endl;
		}

		correct += isCorrect(OutputNeuralNetwork, trueNumbers[element]);
		vector<double> y = BoolVector(trueNumbers[element]);
		BackPropogation(OutputNeuralNetwork, y, h2, WeightsThird, t2, h1, WeightsSecond, t1, inputXs[element], WeightsFirst, BiasFirst, BiasSecond, BiasThird);
	}
	cout << "Точность в конце: " << 100 * correct / trueNumbers.size();

	//сохраняем веса 1
	ofstream w1("WeightsFirst.txt");
	if (w1.is_open()){
		for (int i = 0; i != WeightsFirst.size(); i++) {
			for (int j = 0; j != WeightsFirst[0].size(); j++) {
				w1 << WeightsFirst[i][j] << ' ';
			}
			w1 << endl;
		}
	}
	w1.close();

	//сохраняем свободные веса 1
	ofstream b1("BiasFirst.txt");
	if (b1.is_open()) {
		for (int i = 0; i != BiasFirst.size(); i++) {
			b1 << BiasFirst[i] << ' ';
		}
	}
	b1.close();

	//сохраняем веса 2
	ofstream w2("WeightsSecond.txt");
	if (w2.is_open()) {
		for (int i = 0; i != WeightsSecond.size(); i++) {
			for (int j = 0; j != WeightsSecond[0].size(); j++) {
				w2 << WeightsSecond[i][j] << ' ';
			}
			w2 << endl;
		}
	}
	w2.close();

	//сохраняем свободные веса 2
	ofstream b2("BiasSecond.txt");
	if (b2.is_open()) {
		for (int i = 0; i != BiasSecond.size(); i++) {
			b2 << BiasSecond[i] << ' ';
		}
	}
	b2.close();

	//сохраняем веса 3
	ofstream w3("WeightsThird.txt");
	if (w3.is_open()) {
		for (int i = 0; i != WeightsThird.size(); i++) {
			for (int j = 0; j != WeightsThird[0].size(); j++) {
				w3 << WeightsThird[i][j] << ' ';
			}
			w3 << endl;
		}
	}
	w3.close();

	//сохраняем свободные веса 3
	ofstream b3("BiasThird.txt");
	if (b3.is_open()) {
		for (int i = 0; i != BiasThird.size(); i++) {
			b3 << BiasThird[i] << ' ';
		}
	}
	b3.close();
}

void NextTrain() {
	vector<double> trueNumbers;
	vector<vector<double>> inputXs;

	vector<vector<double>> WeightsFirst;
	vector<double> BiasFirst;
	vector<vector<double>> WeightsSecond;
	vector<double> BiasSecond;
	vector<vector<double>> WeightsThird;
	vector<double> BiasThird;

	ReadData(trueNumbers, inputXs);

	//считываем веса между первым и вторым слоями
	ifstream w1("WeightsFirst.txt");
	if (w1.is_open()) {
		for (int i = 0; i != 784; i++) {
			string line;
			getline(w1, line);
			vector<double> vectorOfNum;
			int index = 0;
			for (int j = 0; j != line.size(); j++) {
				if (line[j] == ' ') {
					vectorOfNum.push_back(stof(line.substr(index, j - index)));
					index = j + 1;
				}
			}
			WeightsFirst.push_back(vectorOfNum);
		}
	}
	w1.close();

	//считываем свободные веса между первым и вторым слоями
	ifstream b1("BiasFirst.txt");
	if (b1.is_open()) {
		int index = 0;
		string line;
		getline(b1, line);
		for (int i = 0; i != line.size(); i++) {
			if (line[i] == ' ') {
				BiasFirst.push_back(stof(line.substr(index, i - index)));
				index = i + 1;
			}
		}
	}
	b1.close();

	//считываем веса между вторым и третьим слоями
	ifstream w2("WeightsSecond.txt");
	if (w2.is_open()) {
		for (int i = 0; i != 16; i++) {
			string line;
			getline(w2, line);
			vector<double> vectorOfNum;
			int index = 0;
			for (int j = 0; j != line.size(); j++) {
				if (line[j] == ' ') {
					vectorOfNum.push_back(stof(line.substr(index, j - index)));
					index = j + 1;
				}
			}
			WeightsSecond.push_back(vectorOfNum);
		}
	}
	w2.close();

	//считываем свободные веса между вторым и третьим слоями
	ifstream b2("BiasSecond.txt");
	if (b2.is_open()) {
		int index = 0;
		string line;
		getline(b2, line);
		for (int i = 0; i != line.size(); i++) {
			if (line[i] == ' ') {
				BiasSecond.push_back(stof(line.substr(index, i - index)));
				index = i + 1;
			}
		}
	}
	b2.close();

	//считываем веса между третьим и четвертым слоями
	ifstream w3("WeightsThird.txt");
	if (w3.is_open()) {
		for (int i = 0; i != 16; i++) {
			string line;
			getline(w3, line);
			vector<double> vectorOfNum;
			int index = 0;
			for (int j = 0; j != line.size(); j++) {
				if (line[j] == ' ') {
					vectorOfNum.push_back(stof(line.substr(index, j - index)));
					index = j + 1;
				}
			}
			WeightsThird.push_back(vectorOfNum);
		}
	}
	w3.close();

	//считываем свободные веса между третьим и четвертым слоями
	ifstream b3("BiasThird.txt");
	if (b3.is_open()) {
		int index = 0;
		string line;
		getline(b3, line);
		for (int i = 0; i != line.size(); i++) {
			if (line[i] == ' ') {
				BiasThird.push_back(stof(line.substr(index, i - index)));
				index = i + 1;
			}
		}
	}
	b3.close();

	double eps = 0.85;
	double accuracy = 0;
	int stage = 3; //эпоха обучения
	while (accuracy < eps){
		accuracy = 0;
		for (int element = 0; element != trueNumbers.size(); element++) {
			vector<double> h2;
			vector<double> h1;
			vector<double> t1;
			vector<double> t2;

			vector<double> OutputNeuralNetwork;
			DirectPassage(OutputNeuralNetwork, inputXs[element], WeightsFirst, BiasFirst, WeightsSecond, BiasSecond, WeightsThird, BiasThird, h1, h2, t1, t2);
			double error = CrossEntropy(OutputNeuralNetwork, trueNumbers[element]);
			//double error = MSE(OutputNeuralNetwork, trueNumbers[element]);
			accuracy += isCorrect(OutputNeuralNetwork, trueNumbers[element]);

			if (element % 1000 == 0) {
				cout << "Элемент: " << element << ' ';
				cout << "Ошибка: " << error << endl;
			}
			
			vector<double> y = BoolVector(trueNumbers[element]);
			BackPropogation(OutputNeuralNetwork, y, h2, WeightsThird, t2, h1, WeightsSecond, t1, inputXs[element], WeightsFirst, BiasFirst, BiasSecond, BiasThird);
		}
	
		//сохраняем веса 1
		ofstream w_1("WeightsFirst.txt");
		if (w_1.is_open()) {
			for (int i = 0; i != WeightsFirst.size(); i++) {
				for (int j = 0; j != WeightsFirst[0].size(); j++) {
					w_1 << WeightsFirst[i][j] << ' ';
				}
				w_1 << endl;
			}
		}
		w_1.close();

		//сохраняем свободные веса 1
		ofstream b_1("BiasFirst.txt");
		if (b_1.is_open()) {
			for (int i = 0; i != BiasFirst.size(); i++) {
				b_1 << BiasFirst[i] << ' ';
			}
		}
		b_1.close();

		//сохраняем веса 2
		ofstream w_2("WeightsSecond.txt");
		if (w_2.is_open()) {
			for (int i = 0; i != WeightsSecond.size(); i++) {
				for (int j = 0; j != WeightsSecond[0].size(); j++) {
					w_2 << WeightsSecond[i][j] << ' ';
				}
				w_2 << endl;
			}
		}
		w_2.close();

		//сохраняем свободные веса 2
		ofstream b_2("BiasSecond.txt");
		if (b_2.is_open()) {
			for (int i = 0; i != BiasSecond.size(); i++) {
				b_2 << BiasSecond[i] << ' ';
			}
		}
		b_2.close();

		//сохраняем веса 3
		ofstream w_3("WeightsThird.txt");
		if (w_3.is_open()) {
			for (int i = 0; i != WeightsThird.size(); i++) {
				for (int j = 0; j != WeightsThird[0].size(); j++) {
					w_3 << WeightsThird[i][j] << ' ';
				}
				w_3 << endl;
			}
		}
		w_3.close();

		//сохраняем свободные веса 3
		ofstream b_3("BiasThird.txt");
		if (b_3.is_open()) {
			for (int i = 0; i != BiasThird.size(); i++) {
				b_3 << BiasThird[i] << ' ';
			}
		}
		b_3.close();

		stage += 1;
		accuracy = accuracy / trueNumbers.size();
		cout << "                        Точность предсказания: " << accuracy * 100 << endl;
	}

	cout << "Прошло эпох: " << stage;
}

void ReadDataPerson(vector<double>& inputX) {
	ifstream data("outputFile.txt");
	if (data.is_open()) {
		string line;
		while(getline(data, line)){
			int index = 0;
			for (int i = 0; i != line.size(); i++) {
				if (line[i] == ' ') {
					inputX.push_back(stof(line.substr(index, i - index)));
					index = i + 1;
				}
			}
		}
	}
	data.close();
}

void Answer(vector<double>& ans) {
	int indexMax = 18768172;
	double maximum = -327865456;
	for (int i = 0; i != ans.size(); i++) {
		if (ans[i] > maximum) {
			maximum = ans[i];
			indexMax = i;
		}
	}
	cout << "Ответ: " << indexMax << endl;
}

void PersonNumbers() {
	while (true) {
		cout << "Для выхода нажмите	quit\nВведите имя файла: ";
		string path;
		cin >> path;

		if (path == "quit")
			break;
		path = "numbers/" + path;

		cv::Mat image = cv::imread(path, cv::IMREAD_GRAYSCALE);
		
		ofstream outputFile("outputFile.txt");

		for (int i = 0; i < image.rows; ++i) {
			for (int j = 0; j < image.cols; ++j) {
				float pixel = 1 - (static_cast<float>(image.at<uchar>(i, j)) / 255.0f);
				outputFile << pixel << " ";
			}
			outputFile << endl;
		}

		vector<double> inputX;

		ReadDataPerson(inputX);

		vector<double> ans;
		vector<double> h1;
		vector<double> h2;
		vector<vector<double>> WeightsFirst;
		vector<double> BiasFirst;
		vector<vector<double>> WeightsSecond;
		vector<double> BiasSecond;
		vector<vector<double>> WeightsThird;
		vector<double> BiasThird;

		//считываем веса между первым и вторым слоями
		ifstream w1("WeightsFirst.txt");
		if (w1.is_open()) {
			for (int i = 0; i != 784; i++) {
				string line;
				getline(w1, line);
				vector<double> vectorOfNum;
				int index = 0;
				for (int j = 0; j != line.size(); j++) {
					if (line[j] == ' ') {
						vectorOfNum.push_back(stof(line.substr(index, j - index)));
						index = j + 1;
					}
				}
				WeightsFirst.push_back(vectorOfNum);
			}
		}
		w1.close();

		//считываем свободные веса между первым и вторым слоями
		ifstream b1("BiasFirst.txt");
		if (b1.is_open()) {
			int index = 0;
			string line;
			getline(b1, line);
			for (int i = 0; i != line.size(); i++) {
				if (line[i] == ' ') {
					BiasFirst.push_back(stof(line.substr(index, i - index)));
					index = i + 1;
				}
			}
		}
		b1.close();

		//считываем веса между вторым и третьим слоями
		ifstream w2("WeightsSecond.txt");
		if (w2.is_open()) {
			for (int i = 0; i != 16; i++) {
				string line;
				getline(w2, line);
				vector<double> vectorOfNum;
				int index = 0;
				for (int j = 0; j != line.size(); j++) {
					if (line[j] == ' ') {
						vectorOfNum.push_back(stof(line.substr(index, j - index)));
						index = j + 1;
					}
				}
				WeightsSecond.push_back(vectorOfNum);
			}
		}
		w2.close();

		//считываем свободные веса между вторым и третьим слоями
		ifstream b2("BiasSecond.txt");
		if (b2.is_open()) {
			int index = 0;
			string line;
			getline(b2, line);
			for (int i = 0; i != line.size(); i++) {
				if (line[i] == ' ') {
					BiasSecond.push_back(stof(line.substr(index, i - index)));
					index = i + 1;
				}
			}
		}
		b2.close();

		//считываем веса между третьим и четвертым слоями
		ifstream w3("WeightsThird.txt");
		if (w3.is_open()) {
			for (int i = 0; i != 16; i++) {
				string line;
				getline(w3, line);
				vector<double> vectorOfNum;
				int index = 0;
				for (int j = 0; j != line.size(); j++) {
					if (line[j] == ' ') {
						vectorOfNum.push_back(stof(line.substr(index, j - index)));
						index = j + 1;
					}
				}
				WeightsThird.push_back(vectorOfNum);
			}
		}
		w3.close();

		//считываем свободные веса между третьим и четвертым слоями
		ifstream b3("BiasThird.txt");
		if (b3.is_open()) {
			int index = 0;
			string line;
			getline(b3, line);
			for (int i = 0; i != line.size(); i++) {
				if (line[i] == ' ') {
					BiasThird.push_back(stof(line.substr(index, i - index)));
					index = i + 1;
				}
			}
		}
		b3.close();

		DirectPassageTest(ans, inputX, WeightsFirst, BiasFirst, WeightsSecond, BiasSecond, WeightsThird, BiasThird, h1, h2);

		Answer(ans);

		// печатаем распределение вероятностей на выходном слое
		for (int i = 0; i != 10; i++) {
			cout << i << ' ' << ans[i] << endl;
		}
	}
}

int main() {
	SetConsoleCP(1251);
	SetConsoleOutputCP(1251);

	// кол-во нейронов в каждом из четырех слоев
	int InputLayer = 784;
	int OutputLayer = 10;
	int HiddenLayerFirst = 16;
	int HiddenLayerSecond = 16;

	//Train();
	//NextTrain();
	//TestTrain();
	PersonNumbers();

	return 0;
}