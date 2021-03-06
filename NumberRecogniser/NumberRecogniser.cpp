// NumberRecogniser.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"

using namespace std;

int main()
{
	ANN NR("number_recogniser.ann");

	string FileNameIn;

	double iv[784];
	double *ov;

	/*
	ifstream images("images_test.txt", ios::binary);
	images.ignore(16, EOF);
	
	for (int im = 0; im < 100; im++)
	{
		string name = to_string(im) + ".pgm";
		ofstream image(name);

		image << "P5\n28\n28\n255\n";

		unsigned char temp;
		for (int i = 0; i < 784; i++)
		{
				images.read((char*)&temp, 1);
				//iv[i] = (double)temp / 255;
				image.write((char*)&temp, 1);
		}

		image.close();
	}
	
	images.close();
	*/

	cout << "image path:";
	cin >> FileNameIn;
	ifstream fin(FileNameIn, ios::binary);

	unsigned char buf[2];
	fin.read((char*)buf, 2);
	//cout << buf[0] << buf[1] << endl;
	if (buf[0] != 'P' && buf[1] != '5')
	{
		cout << "WRONG FORMAT\n";
		return 0;
	}
		
	int dimension;
	for (int i = 0; i < 2; i++)
	{
		fin >> dimension;
		//cout << dimension << endl;
		if (dimension != 28)
		{
			cout << "WRONG Dimensions\n";
			return 0;
		}
	}
	
	int max_val;
	fin >> max_val;
	//cout << max_val << endl;
	fin.ignore(1);
	for (int i = 0; i < 28 * 28; i++)
	{
		fin.read((char*)buf, 1);

		iv[i] = (double)buf[0] / max_val;
		//cout << iv[i]<< endl;
	}

	ov = NR.predict(iv);
	for (int i = 0; i < 10; i++)
	{
		cout << i << ": " << ov[i]*100 << "%" << endl;
	}
	
	
	system("pause");

    return 0;
}

