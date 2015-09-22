/*
 * cFileHandle.h
 *
 *  Created on: Jul 25, 2015
 *      Author: adas
 */

#ifndef CFILEHANDLE_H_
#define CFILEHANDLE_H_

#include <iostream>
#include <string>
#include <fstream>
#include <vector>


class cFileHandle {
private:
public:
	cFileHandle();
	static void openFile(fstream &file, string path)
	{
		// Open file
		file.open(path.c_str());
		if (!file.is_open()) {
			cout << "Weights-Model file not found at: " + path << endl;
			exit(EXIT_FAILURE);
		}
	}

	static void Tokenize(const string& str, vector<string>& tokens, const string& delimiters)
	{
		// Clear the tokens vector
		tokens.clear();

		// Skip delimiters at beginning.
		string::size_type lastPos = str.find_first_not_of(delimiters, 0);

		// Find first "non-delimiter".
		string::size_type pos     = str.find_first_of(delimiters, lastPos);

		while (string::npos != pos || string::npos != lastPos)
		{
			// Found a token, add it to the vector.
			tokens.push_back(str.substr(lastPos, pos - lastPos));

			// Skip delimiters.  Note the "not_of"
			lastPos = str.find_first_not_of(delimiters, pos);

			// Find next "non-delimiter"
			pos = str.find_first_of(delimiters, lastPos);
		}
	}

};

#endif /* CFILEHANDLE_H_ */
