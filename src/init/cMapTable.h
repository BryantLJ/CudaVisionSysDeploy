/*
 * cMapTable.h
 *
 *  Created on: Jul 24, 2015
 *      Author: adas
 */

#ifndef CMAPTABLE_H_
#define CMAPTABLE_H_

class cMapTable {
private:
	static void fillMappingTable(uint8_t *lutTable)
	{
		int samples = 8, index = 0; // number of neighbors
		int newMax = samples * (samples-1)+3;
		for (unsigned int i = 0; i < unsigned(1<<samples); i++){
			if (transitions(i,samples) <= 2) //uniform
				lutTable[i] = index++;
			else
				lutTable[i] = newMax-1;
		}
	}

	// LBP uniform get number of transitions
	static int transitions(unsigned int c, int bits)
	{
		int base = 1;
		int current = c & base, current2, changes = 0;
		for (int i=1;i<bits;i++) {
				base <<= 1;
				current2 = (c & base) >> i;
				if (current ^ current2) changes++;
				current = current2;
		}
		return changes; //(changes <= 2)? 1 : 0;
	}

public:
	static void generateHostLUT(uint8_t **lut, uint lutSize)
	{
		*lut = (uint8_t*) malloc(sizeof(uint8_t) * lutSize);
		fillMappingTable(*lut);
	}
	static void generateDeviceLut(uint8_t **lut, uint lutSize)
	{
		uint8_t *auxLut = (uint8_t*) malloc(sizeof(uint8_t) * lutSize);
		fillMappingTable(auxLut);
		cudaMallocGen<uint8_t>(lut, lutSize);
		copyHtoD<uint8_t>(*lut, auxLut, lutSize);
	}
};



#endif /* CMAPTABLE_H_ */

