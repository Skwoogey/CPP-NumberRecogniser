#include "ANN.h"
#include "stdafx.h"

#include <iostream>
#include <cmath>
#include <string>
#include <fstream>

#define frand() (double)rand()/RAND_MAX
#define ALPHA 0
#define LERNING_RATE 0.1

using namespace std;

double sigmoid(double x)
{
	return 1 / (1 + exp(-x));
}

ANN::ANN(int ln, int* neuron_num_per_layer)
{
	layers_num = ln;
	layers = new Layer*[ln];
	ov_length = neuron_num_per_layer[ln];

	//cout << "ANN with n layers: " << layers_num << endl;
	double* current_input = NULL;
	for (int i = 0; i < ln; i++)
	{
		//cout << "layer: " << i << endl;
		layers[i] = new Layer(neuron_num_per_layer[i], neuron_num_per_layer[i + 1]);
		layers[i]->set_input(current_input);
		current_input = layers[i]->get_output();
	}

	output_layer = current_input;
}

ANN::ANN(std::string in_file)
{
	ifstream inf;
	inf.open(in_file, ios::binary);

	if (!inf.is_open())
	{
		cout << "can't open file\n";
		return;
	}

	inf.read((char*)&layers_num, sizeof(int));
	layers = new Layer*[layers_num];

	inf.read((char*)&ov_length, sizeof(int));

	double* current_input = NULL;
	for (int i = 0; i < layers_num; i++)
	{
		layers[i] = new Layer(&inf);
		layers[i]->set_input(current_input);
		current_input = layers[i]->get_output();
	}

	output_layer = current_input;

	inf.close();
}

double* ANN::predict(double * iv)
{
	layers[0]->set_input(iv);

	for (int i = 0; i < layers_num; i++)
	{
		layers[i]->predict();
	}

	return output_layer;
}

void ANN::train(double * iv, double * dov)
{
	/*
	cout << "ANN input: ";
	for (int i = 0; i < 2; i++)
		cout << iv[i] << " ";
	cout << endl;

	cout << "ANN des output: ";
	for (int i = 0; i < ov_length; i++)
		cout << dov[i] << " ";
	cout << endl;
	*/
	double* delta_temp = new double[ov_length];
	predict(iv);

	for (int i = 0; i < ov_length; i++)
		delta_temp[i] = (dov[i] - output_layer[i]);

	/*
	cout << "ANN prediction: ";
	for (int i = 0; i < ov_length; i++)
		cout << output_layer[i] << " ";
	cout << endl;

	cout << "Error: ";
	for (int i = 0; i < ov_length; i++)
		cout << delta_temp[i] << " ";
	cout << "\n";
	*/

	for (int i = layers_num - 1; i >= 0; i--)
	{
		delta_temp = layers[i]->train(delta_temp);
	}

	delete[] delta_temp;

	//cout << "\n";
}

void ANN::save(string out_file)
{
	ofstream of;
	of.open(out_file, ios::binary);

	if (!of.is_open())
	{
		cout << "can't open file\n";
		return;
	}

	of.write((char*)&layers_num, sizeof(int));
	of.write((char*)&ov_length, sizeof(int));
	for (int i = 0; i < layers_num; i++)
	{
		layers[i]->save(&of);
	}

	of.close();
}

ANN::~ANN()
{
	for (int i = 0; i < layers_num; i++)
	{
		delete layers[i];
	}

	delete[] layers;
}

Layer::Layer(int wl, int ls)
{
	layer_size = ls;
	weights_length = wl;

	//cout << "neurons on this layer: " << layer_size << endl;
	//cout << "weights on this layer: " << wieghts_length << endl;

	neurons = new Neuron*[ls];

	output = new double[layer_size];

	for (int i = 0; i < ls; i++)
	{
		//cout << "neuron: " << i << endl;
		neurons[i] = new Neuron(wl);

	}
}

Layer::Layer(std::ifstream * inf)
{
	inf->read((char*)&weights_length, sizeof(int));
	inf->read((char*)&layer_size, sizeof(int));

	neurons = new Neuron*[layer_size];

	output = new double[layer_size];

	for (int i = 0; i < layer_size; i++)
	{
		neurons[i] = new Neuron(inf, weights_length);

	}
}

void Layer::set_input(double * new_input)
{
	input = new_input;
}

double * Layer::get_output()
{
	return output;
}

void Layer::save(ofstream *of)
{
	of->write((char*)&weights_length, sizeof(int));
	of->write((char*)&layer_size, sizeof(int));
	for (int i = 0; i < layer_size; i++)
	{
		neurons[i]->save(of);
	}


}

void Layer::predict()
{
	for (int i = 0; i < layer_size; i++)
		output[i] = neurons[i]->predict(input);
}

double* Layer::train(double* error)
{
	double* new_error = new double[weights_length];

	for (int i = 0; i < weights_length; i++)
		new_error[i] = 0;

	/*
	cout << "def err value: ";
	for (int i = 0; i < weights_length; i++)
		cout << new_error[i] << " ";
	cout << endl;
	*/
	for (int i = 0; i < layer_size; i++)
	{
		neurons[i]->train(error[i], new_error, input);

		/*
		cout << "new err value: ";
		for (int i = 0; i < weights_length; i++)
			cout << new_error[i] << " ";
		cout << endl;
		*/
	}
	//cout << endl;

	/*
	for (int i = 0; i < weights_length; i++)
		cout << new_error[i] << " ";
	cout << endl;
	*/
	delete[] error;

	return new_error;
}

Layer::~Layer()
{
	for (int i = 0; i < layer_size; i++)
	{
		delete neurons[i];
	}

	delete[] output;
	delete[] neurons;
}

Neuron::Neuron(int wl)
{
	length = wl;

	weights = new double[wl];
	weight_changes = new double[wl];
	//cout << "weights length of this neuron: " << length << endl;

	for (int i = 0; i < length; i++)
	{
		weights[i] = frand() - 0.5;
		weight_changes[i] = 0;
	}
}

Neuron::Neuron(std::ifstream * inf, int wl)
{
	length = wl;

	weights = new double[wl];
	weight_changes = new double[wl];

	inf->read((char*)&bias, sizeof(double));
	for (int i = 0; i < length; i++)
	{
		inf->read((char*)(weights + i), sizeof(double));
		weight_changes[i] = 0;
	}
}

void Neuron::save(ofstream * of)
{
	of->write((char*)&bias, sizeof(double));
	for (int i = 0; i < length; i++)
		of->write((char*)(weights + i), sizeof(double));
}

double Neuron::predict(double * iv)
{
	double temp = bias;
	for (int i = 0; i < length; i++)
	{
		temp += weights[i] * iv[i];
	}

	last_output = sigmoid(temp);

	return last_output;
}

void Neuron::train(double error, double* new_error, double* input)
{
	double true_error = error*last_output * (1 - last_output);
	for (int i = 0; i < length; i++)
	{
		new_error[i] += weights[i] * true_error;
		weight_changes[i] = LERNING_RATE*true_error*input[i];
		weights[i] += weight_changes[i];
	}

	bias += LERNING_RATE * true_error;
}

Neuron::~Neuron()
{
	delete[] weights;
	delete[] weight_changes;
}