/*
 * cReadParameters.h
 *
 *  Created on: Jul 23, 2015
 *      Author: adas
 */

#ifndef CPARAMETERS_H_
#define CPARAMETERS_H_

// Includes
#include <iostream>
#include <stdlib.h>
#include <sstream>
#include <string>
#include <fstream>
#include <unistd.h>
#include <ctime>

#include "../common/parameters.h"
#include "fileInOut/IniFileIO.h"

class cParameters {
private:
	parameters m_params;
	string m_parametersFile;
public:
	cParameters();
	void readParameters();
	inline parameters* getParams() { return &m_params; }

};

#endif /* CPARAMETERS_H_ */
