/*
 * LBPcomputeH.h
 *
 *  Created on: Aug 27, 2015
 *      Author: adas
 */

#ifndef LBPCOMPUTEH_H_
#define LBPCOMPUTEH_H_

#include "../common/operators.h"

/* LBP CPU implementation
 *
 *
 */
template<typename T>
void ComputeFast (const cv::Mat &input, cv::Mat &output, const uint8_t *mapTable)
{
    const int predicate = 1;
    int leap = input.cols*predicate;

    // Set up a circularly indexed neighborhood using nine pointers
    T
        *p7 = input.data,
        *p6 = p7 + predicate,
        *p5 = p6 + predicate,
        *p4 = p5 + leap,
        *p3 = p4 + leap,
        *p2 = p3 - predicate,
        *p1 = p2 - predicate,
        *p0 = p1 - leap,
        *pCenter = p0 + predicate,
        *pResult = output.data + predicate + leap;

    int pred2 = predicate << 1;
    uint value, center;
    memset(output.data, 0, (leap+predicate)*sizeof(T));	// Set the first output row to zero
    memset((output.data + ((input.rows-1)*input.cols)), 0, leap * sizeof(T)); // set the last row to zero

    for (int y = predicate; y < (input.rows-predicate); y++){
        for (int x = predicate; x < (input.cols-predicate); x++){
            value = 0;
            center = *pCenter + m_clipTh;
            compab_mask_inc(p0, 0);
            compab_mask_inc(p1, 1);
            compab_mask_inc(p2, 2);
            compab_mask_inc(p3, 3);
            compab_mask_inc(p4, 4);
            compab_mask_inc(p5, 5);
            compab_mask_inc(p6, 6);
            compab_mask_inc(p7, 7);
            pCenter++;
	        *pResult = mapTable[(uchar)value];
			//*pResult = value;  //Result without mapping table
            pResult++;
        }
        p0 += pred2;
        p1 += pred2;
        p2 += pred2;
        p3 += pred2;
        p4 += pred2;
        p5 += pred2;
        p6 += pred2;
        p7 += pred2;
        pCenter += pred2;
        //pResult += pred2;
        *pResult = 0;
        pResult++;
        *pResult = 0;
        pResult++;
    }
}


#endif /* LBPCOMPUTEH_H_ */
