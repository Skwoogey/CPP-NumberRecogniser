#pragma once

#include <fstream>

class Neuron
{
	int length;
	double* weights;
	double* weight_changes;
	double last_output;

	double bias = 0;

public:
	Neuron(int wl);

	Neuron(std::ifstream* inf, int length);

	void save(std::ofstream* of);

	double predict(double* iv);

	void train(double error, double* new_error, double* input);

	~Neuron();
};

class Layer
{
	double *input, *output;
	int layer_size;
	int weights_length;
	Neuron** neurons;

public:
	Layer(int wl, int ls);

	Layer(std::ifstream *inf);

	void set_input(double *new_input);
	double* get_output();

	void save(std::ofstream *of);

	void predict();

	double* train(double *error);

	~Layer();
};

class ANN
{
	int layers_num;
	Layer** layers;
	double learning_rate = 0.5;
	double* output_layer;
	int ov_length;

public:
	ANN(int ln, int* neuron_num_per_layer);

	ANN(std::string in_file);

	double* predict(double* iv);

	void train(double *iv, double *dov);

	void save(std::string out_file);

	~ANN();
};