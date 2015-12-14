/*
 * RFclassification.h
 *
 *  Created on: Dec 3, 2015
 *      Author: adas
 */

#ifndef RFCLASSIFICATION_H_
#define RFCLASSIFICATION_H_

#include "applyNode.h"

template<typename T>
void computeROIscore(T *features, T *scores, dataSizes *szs)
{
	for (int i = 0; i < szs->rf.nTrees; i++) {
		for (int j = 0; j < szs->rf.nNodes[i]; j++) {
			// cooretig pointers funcio
			applyNodeSVM(features, szs);

		}
	}
}



template<typename T>
void computeRFscores(T *features, T *outScores, dataSizes *sizes)
{
	for (int i = 0; i < sizes->rf.yROIs_d; i++) {
		for (int j = 0; j < sizes->rf.xROIs_d; ++j) {
			// corretgit pointers de la funcio
			computeROIscore(features, outScores, sizes);

		}

	}

}


#endif /* RFCLASSIFICATION_H_ */
